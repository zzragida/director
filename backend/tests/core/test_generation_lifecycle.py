from director.core.generation_lifecycle import (
    GenerationOperationState,
    begin_resume,
    begin_submission,
    create_operation,
    make_generation_run_id,
    make_operation_id,
    record_failure,
    record_materialized,
    record_persisted,
    record_provider_request,
    safe_operation_summary,
)


def test_generation_and_operation_ids_are_deterministic():
    run_id = make_generation_run_id("abcdef0123456789abcdef0123456789")
    assert run_id == make_generation_run_id("abcdef0123456789abcdef0123456789")

    first = make_operation_id(run_id, kind="scene_video", index=2)
    second = make_operation_id(run_id, kind="scene_video", index=2)
    other = make_operation_id(run_id, kind="scene_video", index=1)

    assert first == second
    assert first != other
    assert first.startswith("genop:scene_video:2:")


def test_operation_records_submission_provider_id_and_persistence():
    operation = create_operation(
        "genrun:test",
        kind="scene_video",
        provider="kling",
        index=0,
    )

    begin_submission(operation)
    assert operation.state == GenerationOperationState.submitting
    assert operation.attempt_count == 1

    record_provider_request(operation, "task-123")
    assert operation.state == GenerationOperationState.submitted
    assert operation.provider_request_id == "task-123"
    assert operation.has_provider_resume_token is True

    record_materialized(operation)
    assert operation.state == GenerationOperationState.materialized

    record_persisted(operation, {"id": "video-1", "length": 5})
    assert operation.state == GenerationOperationState.persisted
    assert operation.is_persisted is True
    assert operation.has_provider_resume_token is False
    assert operation.artifact["id"] == "video-1"


def test_failed_submitted_operation_keeps_resume_token():
    operation = create_operation(
        "genrun:test",
        kind="scene_video",
        provider="stabilityai",
        index=1,
    )
    begin_submission(operation)
    record_provider_request(operation, "generation-77")
    record_failure(operation, code="scene_generation_failed")

    assert operation.state == GenerationOperationState.failed
    assert operation.provider_request_id == "generation-77"
    assert operation.has_provider_resume_token is True
    assert operation.last_error_code == "scene_generation_failed"

    begin_resume(operation)
    assert operation.state == GenerationOperationState.submitted
    assert operation.attempt_count == 2
    assert operation.provider_request_id == "generation-77"


def test_safe_summary_does_not_expose_provider_request_id():
    operation = create_operation(
        "genrun:test",
        kind="scene_video",
        provider="kling",
        index=0,
    )
    begin_submission(operation)
    record_provider_request(operation, "private-provider-task-id")

    summary = safe_operation_summary(operation)

    assert summary["provider_request_known"] is True
    assert "provider_request_id" not in summary
    assert "private-provider-task-id" not in str(summary)
