from director.core.generation_lifecycle import GenerationOperation
from director.core.generation_reconciliation import (
    classify_generation_operation,
    safe_reconciliation_summary,
)


def make_operation(state="planned", provider="kling"):
    return GenerationOperation(
        operation_id="genop:scene_video:0:test",
        kind="scene_video",
        provider=provider,
        state=state,
        updated_at_epoch=1000,
    )


def test_planned_operation_can_submit():
    operation = make_operation()
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1010,
        stale_after_seconds=300,
    )
    assert decision.classification == "healthy"
    assert decision.action == "retry_submission"
    assert decision.auto_execute is True


def test_fresh_submitting_without_provider_id_waits_instead_of_resubmitting():
    operation = make_operation(state="submitting")
    operation.attempt_count = 1
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1100,
        stale_after_seconds=300,
    )
    assert decision.classification == "healthy"
    assert decision.action == "wait"
    assert decision.reason_code == "submission_in_flight"
    assert decision.auto_execute is False
    assert decision.stale is False


def test_stale_submitting_without_provider_id_requires_manual_reconciliation():
    operation = make_operation(state="submitting")
    operation.attempt_count = 1
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    assert decision.classification == "unknown"
    assert decision.action == "manual_review"
    assert decision.reason_code == "submission_outcome_unknown"
    assert decision.auto_execute is False
    assert decision.stale is True


def test_submitted_provider_task_is_resumed_when_supported():
    operation = make_operation(state="submitted")
    operation.attempt_count = 1
    operation.provider_request_id = "provider-task-1"
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    assert decision.classification == "reconcile_required"
    assert decision.action == "resume_provider"
    assert decision.reason_code == "submitted_provider_task_available"
    assert decision.auto_execute is True


def test_materialized_without_durable_artifact_becomes_orphan_candidate():
    operation = make_operation(state="materialized", provider="elevenlabs")
    operation.attempt_count = 1
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=False,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    assert decision.classification == "orphan_candidate"
    assert decision.action == "manual_review"
    assert decision.reason_code == "materialized_artifact_not_durable"


def test_discovered_durable_artifact_wins_over_provider_retry():
    operation = make_operation(state="materialized")
    operation.attempt_count = 1
    operation.provider_request_id = "provider-task-1"
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        durable_artifact_found=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    assert decision.classification == "reconcile_required"
    assert decision.action == "restore_persisted_artifact"
    assert decision.reason_code == "durable_artifact_checkpoint_missing"
    assert decision.auto_execute is True


def test_failed_resumable_provider_without_request_id_is_not_blindly_resubmitted():
    operation = make_operation(state="failed")
    operation.attempt_count = 1
    operation.recoverable = True
    operation.last_error_code = "scene_generation_failed"
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    assert decision.classification == "unknown"
    assert decision.action == "manual_review"
    assert decision.reason_code == "provider_submission_may_have_succeeded"


def test_failed_non_resumable_provider_preserves_existing_retry_contract():
    operation = make_operation(state="failed", provider="videodb")
    operation.attempt_count = 1
    operation.recoverable = True
    operation.last_error_code = "scene_generation_failed"
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=False,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    assert decision.classification == "healthy"
    assert decision.action == "retry_submission"
    assert decision.auto_execute is True


def test_safe_summary_does_not_expose_provider_request_id():
    operation = make_operation(state="submitted")
    operation.provider_request_id = "sensitive-provider-task-id"
    decision = classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    summary = safe_reconciliation_summary(decision)
    assert "provider_request_id" not in summary
    assert "sensitive-provider-task-id" not in str(summary)
