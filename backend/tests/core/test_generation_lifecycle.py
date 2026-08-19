from director.core.generation_lifecycle import (
    GenerationOperation,
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
    assert operation.submission_count == 1
    assert operation.resume_count == 0
    assert operation.accounting_known is True

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


def test_failed_submitted_operation_keeps_resume_token_without_new_submission():
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
    assert operation.submission_count == 1
    assert operation.resume_count == 1
    assert operation.provider_request_id == "generation-77"


def test_second_new_submission_is_counted_separately_from_resume():
    operation = create_operation(
        "genrun:test",
        kind="scene_video",
        provider="videodb",
        index=0,
    )
    begin_submission(operation)
    record_failure(operation, code="temporary_failure")
    begin_submission(operation)

    assert operation.attempt_count == 2
    assert operation.submission_count == 2
    assert operation.resume_count == 0


def test_legacy_v2_operation_does_not_invent_historical_submission_split():
    operation = GenerationOperation.model_validate(
        {
            "version": 2,
            "operation_id": "genop:scene_video:0:legacy",
            "kind": "scene_video",
            "provider": "kling",
            "state": "failed",
            "attempt_count": 3,
            "provider_request_id": None,
            "artifact": None,
            "last_error_code": "legacy_failure",
            "recoverable": True,
            "updated_at_epoch": 1000,
        }
    )

    assert operation.submission_count is None
    assert operation.resume_count is None
    assert operation.accounting_known is False

    begin_submission(operation)
    assert operation.version == 3
    assert operation.attempt_count == 4
    assert operation.submission_count == 1
    assert operation.resume_count == 0
    assert operation.accounting_known is False


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
    assert summary["submission_count"] == 1
    assert summary["resume_count"] == 0
    assert summary["accounting_known"] is True
    assert "provider_request_id" not in summary
    assert "private-provider-task-id" not in str(summary)
