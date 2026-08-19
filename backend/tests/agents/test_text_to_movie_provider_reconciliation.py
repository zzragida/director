import importlib.util
from pathlib import Path


def load_checkpoint_test_helpers():
    helper_path = (
        Path(__file__).resolve().parent
        / "test_text_to_movie_checkpoint_resume.py"
    )
    spec = importlib.util.spec_from_file_location(
        "text_to_movie_checkpoint_helpers_for_reconciliation",
        helper_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_provider_request_id_is_resumed_without_duplicate_submission(
    monkeypatch,
):
    helper = load_checkpoint_test_helpers()

    request_ids = [None, None, "kling-task-scene-3"]
    submission_script = [
        {"id": "scene-1", "length": 4},
        {"id": "scene-2", "length": 5},
        RuntimeError("polling interrupted after task submission"),
    ]
    resume_script = [None]
    counters = {"submit": 0, "resume": 0}

    def tracked_text_to_video(self, *args, on_request_id=None, **kwargs):
        counters["submit"] += 1
        request_id = request_ids.pop(0)
        if request_id and on_request_id is not None:
            on_request_id(request_id)
        result = submission_script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def resume_text_to_video(self, request_id, save_at):
        counters["resume"] += 1
        assert request_id == "kling-task-scene-3"
        return resume_script.pop(0)

    monkeypatch.setattr(
        helper.FakeVideoGenerationTool,
        "text_to_video",
        tracked_text_to_video,
    )
    monkeypatch.setattr(
        helper.FakeVideoGenerationTool,
        "resume_text_to_video",
        resume_text_to_video,
        raising=False,
    )
    monkeypatch.setenv("KLING_AI_ACCESS_API_KEY", "test-access")
    monkeypatch.setenv("KLING_AI_SECRET_API_KEY", "test-secret")

    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()

    first_agent, _ = helper.make_agent(module, db, helper.first_run_llm())
    first_response = first_agent.run(
        collection_id="collection-1",
        engine="kling",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )

    assert first_response.status == helper.AgentStatus.ERROR
    assert first_response.data["stage"] == "scene_generation"
    assert first_response.data["failed_scene_index"] == 2
    assert first_response.data["operation_id"].startswith("genop:scene_video:2:")
    assert counters == {"submit": 3, "resume": 0}

    store = module.TextToMovieCheckpointStore(helper.FakeSession(db))
    checkpoint = store.get(first_response.data["checkpoint_id"])
    failed_operation = checkpoint.scenes[2].operation
    assert failed_operation.provider_request_id == "kling-task-scene-3"
    assert failed_operation.has_provider_resume_token is True
    assert failed_operation.attempt_count == 1

    helper.FakeVideoDBTool.upload_script = [
        {"id": "scene-3", "length": 5}
    ]
    helper.FakeAudioGenerationTool.script = [
        {"id": "audio-1", "length": 14}
    ]
    retry_llm = helper.FakeLLM(
        [helper.FakeLLMResponse(content="soft piano rising to warmth")]
    )
    retry_agent, _ = helper.make_agent(module, db, retry_llm)
    retry_response = retry_agent.run(
        collection_id="collection-1",
        engine="kling",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )

    assert retry_response.status == helper.AgentStatus.SUCCESS
    assert counters == {"submit": 3, "resume": 1}
    assert retry_llm.calls == 1

    completed = store.get(first_response.data["checkpoint_id"])
    scene_three = completed.scenes[2]
    assert scene_three.operation.attempt_count == 2
    assert scene_three.operation.state == "persisted"
    assert scene_three.operation.artifact["id"] == "scene-3"
    assert scene_three.media["id"] == "scene-3"


def test_legacy_checkpoint_gets_lifecycle_backfilled_without_losing_media(monkeypatch):
    helper = load_checkpoint_test_helpers()
    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()

    fingerprint = module.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
    )
    checkpoint = module.create_checkpoint(
        request_fingerprint=fingerprint,
        visual_style={"camera_setup": "35mm"},
        scenes=[
            {
                "story_beat": "arrival",
                "scene_description": "wide entrance",
                "suggested_duration": 4,
            }
        ],
    )
    checkpoint.version = 1
    checkpoint.generation_run_id = None
    checkpoint.scenes[0].media = {"id": "existing-scene", "length": 4}
    checkpoint.scenes[0].status = "complete"
    checkpoint.scenes[0].operation = None

    changed = checkpoint.ensure_lifecycle(
        video_provider="videodb",
        audio_provider="videodb",
    )

    assert changed is True
    assert checkpoint.generation_run_id.startswith("genrun:")
    assert checkpoint.scenes[0].operation.state == "persisted"
    assert checkpoint.scenes[0].operation.artifact["id"] == "existing-scene"
    assert checkpoint.audio_operation.operation_id.startswith(
        "genop:background_audio:"
    )
