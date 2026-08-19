import importlib.util
import json
from pathlib import Path


def load_helpers():
    helper_path = Path(__file__).resolve().parent / "test_text_to_movie_checkpoint_resume.py"
    spec = importlib.util.spec_from_file_location(
        "text_to_movie_checkpoint_helpers_for_budget_guard",
        helper_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(module):
    return module.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
    )


def run_request(agent):
    return agent.run(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )


def rate_card(unit_price=100):
    return json.dumps(
        {
            "rate_card_id": "director-test-rates",
            "rate_card_version": "2026-08",
            "rates": [
                {
                    "provider": "videodb",
                    "media_type": "video",
                    "basis": "per_submission",
                    "unit_price_micros": unit_price,
                    "currency": "USD",
                },
                {
                    "provider": "videodb",
                    "media_type": "audio",
                    "basis": "per_submission",
                    "unit_price_micros": unit_price,
                    "currency": "USD",
                },
            ],
        }
    )


def budget(max_total=100, max_retry=100, max_retry_submissions=1):
    return json.dumps(
        {
            "policy_id": "director-test-budget",
            "policy_version": "1",
            "currency": "USD",
            "max_total_micros": max_total,
            "max_retry_micros": max_retry,
            "max_submissions_per_operation": 2,
            "max_retry_submissions_per_operation": max_retry_submissions,
        }
    )


def seed_one_pending_scene(helper, module, db, *, failed_once=False):
    checkpoint = module.create_checkpoint(
        request_fingerprint=fingerprint(module),
        visual_style=json.loads(helper.valid_visual_style()),
        scenes=json.loads(helper.valid_scene_sequence())["scenes"],
        video_provider="videodb",
        audio_provider="videodb",
    )
    checkpoint.status = "generating"
    first = checkpoint.scenes[0]
    first.prompt = "existing scene prompt"
    if failed_once:
        first.operation.submission_count = 1
        first.operation.attempt_count = 1
        first.operation.state = "failed"
        first.operation.last_error_code = "scene_generation_failed"
        first.operation.recoverable = True

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
    return checkpoint, store


def test_total_budget_blocks_provider_before_submission(monkeypatch):
    helper = load_helpers()
    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()
    checkpoint, store = seed_one_pending_scene(helper, module, db)

    monkeypatch.setenv("DIRECTOR_GENERATION_BUDGET_POLICY_JSON", budget(max_total=50))
    monkeypatch.setenv("DIRECTOR_GENERATION_RATE_CARD_JSON", rate_card(unit_price=100))

    agent, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    response = run_request(agent)

    assert response.status == helper.AgentStatus.ERROR
    assert response.data["stage"] == "budget"
    assert response.data["code"] == "budget_total_exceeded"
    assert response.data["operation_id"] == checkpoint.scenes[0].operation.operation_id
    assert helper.FakeVideoGenerationTool.total_calls == 0
    assert helper.FakeAudioGenerationTool.total_calls == 0

    persisted = store.get(checkpoint.checkpoint_id)
    assert persisted.scenes[0].operation.submission_count == 0
    assert persisted.failure_stage == "budget"
    assert "__generation_budget_reservations__" not in db.context["session-1"]


def test_allowed_budget_creates_reservation_and_generation_completes(monkeypatch):
    helper = load_helpers()
    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()
    checkpoint, store = seed_one_pending_scene(helper, module, db)

    monkeypatch.setenv("DIRECTOR_GENERATION_BUDGET_POLICY_JSON", budget(max_total=100))
    monkeypatch.setenv("DIRECTOR_GENERATION_RATE_CARD_JSON", rate_card(unit_price=100))
    helper.FakeVideoGenerationTool.script = [{"id": "scene-1", "length": 4}]

    agent, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    response = run_request(agent)

    assert response.status == helper.AgentStatus.SUCCESS
    assert helper.FakeVideoGenerationTool.total_calls == 1
    assert helper.FakeAudioGenerationTool.total_calls == 0

    context = db.context["session-1"]
    messages = context["__generation_budget_reservations__"]
    reservations = json.loads(messages[-1]["content"])
    assert len(reservations) == 1
    reservation = next(iter(reservations.values()))
    assert reservation["estimated_amount_micros"] == 100
    assert reservation["submission_ordinal"] == 1
    assert reservation["retry_submission"] is False

    persisted = store.get(checkpoint.checkpoint_id)
    assert persisted.status == "complete"
    assert persisted.scenes[0].operation.submission_count == 1


def test_retry_submission_limit_blocks_retry_without_provider_call(monkeypatch):
    helper = load_helpers()
    helper.reset_scripts()
    module = helper.load_text_to_movie_module(monkeypatch)
    db = helper.FakeDB()
    checkpoint, store = seed_one_pending_scene(helper, module, db, failed_once=True)

    monkeypatch.setenv(
        "DIRECTOR_GENERATION_BUDGET_POLICY_JSON",
        budget(max_total=1000, max_retry=1000, max_retry_submissions=0),
    )
    monkeypatch.setenv("DIRECTOR_GENERATION_RATE_CARD_JSON", rate_card(unit_price=100))

    agent, _ = helper.make_agent(module, db, helper.FakeLLM([]))
    response = run_request(agent)

    assert response.status == helper.AgentStatus.ERROR
    assert response.data["stage"] == "budget"
    assert response.data["code"] == "budget_retry_limit_exceeded"
    assert helper.FakeVideoGenerationTool.total_calls == 0

    persisted = store.get(checkpoint.checkpoint_id)
    assert persisted.scenes[0].operation.submission_count == 1
    assert persisted.failure_stage == "budget"
