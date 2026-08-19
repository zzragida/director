import importlib.util
import json
from pathlib import Path


def load_checkpoint_test_helpers():
    helper_path = (
        Path(__file__).resolve().parent
        / "test_text_to_movie_checkpoint_resume.py"
    )
    spec = importlib.util.spec_from_file_location(
        "text_to_movie_checkpoint_helpers_for_generation_lease",
        helper_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_request(agent):
    return agent.run(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )


def request_fingerprint(module):
    return module.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
    )


def test_planning_lease_blocks_duplicate_llm_planning(monkeypatch):
    helper = load_checkpoint_test_helpers()
    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()

    fingerprint = request_fingerprint(module)
    planning_operation_id = module.make_operation_id(
        module.make_generation_run_id(fingerprint),
        kind="planning",
    )
    lease_store = module.GenerationLeaseStore(helper.FakeSession(db))
    held = lease_store.acquire(
        planning_operation_id,
        owner_id="worker-a",
        ttl_seconds=300,
    )
    assert held.acquired is True

    blocked_llm = helper.FakeLLM([])
    worker_b, _ = helper.make_agent(module, db, blocked_llm)
    response = run_request(worker_b)

    assert response.status == helper.AgentStatus.ERROR
    assert response.data["stage"] == "concurrency"
    assert response.data["code"] == "operation_locked"
    assert response.data["operation_id"] == planning_operation_id
    assert blocked_llm.calls == 0
    assert helper.FakeVideoGenerationTool.total_calls == 0
    assert "lease_token" not in str(response.data)
    assert "owner_id" not in str(response.data)


def test_scene_lease_allows_only_one_worker_to_submit_provider_work(monkeypatch):
    helper = load_checkpoint_test_helpers()
    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()

    fingerprint = request_fingerprint(module)
    checkpoint = module.create_checkpoint(
        request_fingerprint=fingerprint,
        visual_style=json.loads(helper.valid_visual_style()),
        scenes=json.loads(helper.valid_scene_sequence())["scenes"],
        video_provider="videodb",
        audio_provider="videodb",
    )
    checkpoint.status = "generating"
    first_scene = checkpoint.scenes[0]
    first_scene.prompt = "existing scene prompt"

    for index, scene in enumerate(checkpoint.scenes[1:], start=2):
        media = {"id": f"scene-{index}", "length": 5}
        scene.media = media
        scene.status = "complete"
        scene.operation.state = "persisted"
        scene.operation.artifact = dict(media)
        scene.operation.recoverable = False

    checkpoint.audio_media = {"id": "audio-1", "length": 14}
    checkpoint.audio_operation.state = "persisted"
    checkpoint.audio_operation.artifact = dict(checkpoint.audio_media)
    checkpoint.audio_operation.recoverable = False

    store = module.TextToMovieCheckpointStore(helper.FakeSession(db))
    store.save(checkpoint)

    lease_store = module.GenerationLeaseStore(helper.FakeSession(db))
    held = lease_store.acquire(
        first_scene.operation.operation_id,
        owner_id="worker-a",
        ttl_seconds=300,
    )
    assert held.acquired is True

    worker_b, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    blocked = run_request(worker_b)

    assert blocked.status == helper.AgentStatus.ERROR
    assert blocked.data["stage"] == "concurrency"
    assert blocked.data["code"] == "operation_locked"
    assert blocked.data["operation_id"] == first_scene.operation.operation_id
    assert helper.FakeVideoGenerationTool.total_calls == 0
    assert helper.FakeAudioGenerationTool.total_calls == 0

    assert lease_store.release(held.lease) is True

    helper.FakeVideoGenerationTool.script = [
        {"id": "scene-1", "length": 4}
    ]
    worker_c, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    completed = run_request(worker_c)

    assert completed.status == helper.AgentStatus.SUCCESS
    assert helper.FakeVideoGenerationTool.total_calls == 1
    assert helper.FakeAudioGenerationTool.total_calls == 0
    assert completed.data["video_url"] == "https://stream.example/final.m3u8"

    persisted = store.get(checkpoint.checkpoint_id)
    assert persisted.scenes[0].status == "complete"
    assert persisted.scenes[0].media["id"] == "scene-1"
    assert persisted.status == "complete"
