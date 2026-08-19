import importlib.util
import json
from pathlib import Path


def load_checkpoint_test_helpers():
    helper_path = (
        Path(__file__).resolve().parent
        / "test_text_to_movie_checkpoint_resume.py"
    )
    spec = importlib.util.spec_from_file_location(
        "text_to_movie_checkpoint_helpers_for_orphan_reconciliation",
        helper_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_checkpoint(module, helper, db, *, video_provider="videodb"):
    fingerprint = module.build_request_fingerprint(
        collection_id="collection-1",
        engine=video_provider,
        audio_engine="videodb",
        storyline="A quiet reunion",
    )
    checkpoint = module.create_checkpoint(
        request_fingerprint=fingerprint,
        visual_style=json.loads(helper.valid_visual_style()),
        scenes=json.loads(helper.valid_scene_sequence())["scenes"],
        video_provider=video_provider,
        audio_provider="videodb",
    )
    store = module.TextToMovieCheckpointStore(helper.FakeSession(db))
    return checkpoint, store


def run_request(agent, *, engine="videodb"):
    return agent.run(
        collection_id="collection-1",
        engine=engine,
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )


def test_stale_submitting_without_provider_id_blocks_duplicate_submission(monkeypatch):
    helper = load_checkpoint_test_helpers()
    helper.reset_scripts()

    def resume_text_to_video(self, request_id, save_at):
        raise AssertionError("resume should not run without a provider request id")

    monkeypatch.setattr(
        helper.FakeVideoGenerationTool,
        "resume_text_to_video",
        resume_text_to_video,
        raising=False,
    )
    monkeypatch.setenv("KLING_AI_ACCESS_API_KEY", "test-access")
    monkeypatch.setenv("KLING_AI_SECRET_API_KEY", "test-secret")

    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()
    checkpoint, store = make_checkpoint(
        module,
        helper,
        db,
        video_provider="kling",
    )

    operation = checkpoint.scenes[0].operation
    operation.state = "submitting"
    operation.attempt_count = 1
    operation.provider_request_id = None
    operation.updated_at_epoch = 1
    checkpoint.scenes[0].prompt = "existing scene prompt"
    store.save(checkpoint)

    agent, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    response = run_request(agent, engine="kling")

    assert response.status == helper.AgentStatus.ERROR
    assert response.data["stage"] == "reconciliation"
    assert response.data["code"] == "submission_outcome_unknown"
    assert response.data["resumable"] is False
    assert response.data["operation_id"] == operation.operation_id
    assert response.data["reconciliation"]["classification"] == "unknown"
    assert response.data["reconciliation"]["action"] == "manual_review"
    assert helper.FakeVideoGenerationTool.total_calls == 0


def test_materialized_scene_without_durable_artifact_is_orphan_candidate(monkeypatch):
    helper = load_checkpoint_test_helpers()
    helper.reset_scripts()

    monkeypatch.setattr(
        helper.FakeVideoDBTool,
        "get_videos",
        lambda self: [],
        raising=False,
    )

    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()
    checkpoint, store = make_checkpoint(module, helper, db)

    operation = checkpoint.scenes[0].operation
    operation.state = "materialized"
    operation.attempt_count = 1
    operation.updated_at_epoch = 1
    checkpoint.scenes[0].prompt = "existing scene prompt"
    store.save(checkpoint)

    agent, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    response = run_request(agent)

    assert response.status == helper.AgentStatus.ERROR
    assert response.data["stage"] == "reconciliation"
    assert response.data["code"] == "materialized_artifact_not_durable"
    assert response.data["reconciliation"]["classification"] == "orphan_candidate"
    assert response.data["reconciliation"]["action"] == "manual_review"
    assert helper.FakeVideoGenerationTool.total_calls == 0


def test_durable_upload_is_restored_when_checkpoint_commit_was_missing(monkeypatch):
    helper = load_checkpoint_test_helpers()
    helper.reset_scripts()

    durable_videos = []
    lookup_calls = {"video": 0}

    def get_videos(self):
        lookup_calls["video"] += 1
        return list(durable_videos)

    monkeypatch.setattr(
        helper.FakeVideoDBTool,
        "get_videos",
        get_videos,
        raising=False,
    )

    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()
    checkpoint, store = make_checkpoint(module, helper, db)

    first = checkpoint.scenes[0]
    first.operation.state = "materialized"
    first.operation.attempt_count = 1
    first.operation.updated_at_epoch = 1
    first.prompt = "existing scene prompt"
    durable_videos.append(
        {
            "id": "scene-1-recovered",
            "name": module.make_artifact_name(first.operation.operation_id, "video"),
            "collection_id": "collection-1",
            "length": 4,
        }
    )

    for index, scene in enumerate(checkpoint.scenes[1:], start=1):
        media = {"id": f"scene-{index + 1}", "length": 5}
        scene.media = media
        scene.status = "complete"
        scene.operation.state = "persisted"
        scene.operation.artifact = dict(media)
        scene.operation.recoverable = False

    checkpoint.audio_media = {"id": "audio-1", "length": 14}
    checkpoint.audio_operation.state = "persisted"
    checkpoint.audio_operation.artifact = dict(checkpoint.audio_media)
    checkpoint.audio_operation.recoverable = False
    store.save(checkpoint)

    agent, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    response = run_request(agent)

    assert response.status == helper.AgentStatus.SUCCESS
    assert helper.FakeVideoGenerationTool.total_calls == 0
    assert helper.FakeAudioGenerationTool.total_calls == 0
    assert lookup_calls["video"] == 1

    restored = store.get(checkpoint.checkpoint_id)
    restored_scene = restored.scenes[0]
    assert restored_scene.status == "complete"
    assert restored_scene.media["id"] == "scene-1-recovered"
    assert restored_scene.operation.state == "persisted"
    assert restored_scene.operation.artifact["id"] == "scene-1-recovered"
