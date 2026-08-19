import importlib.util
from pathlib import Path


def load_modules():
    core_dir = Path(__file__).resolve().parents[2] / "director" / "core"

    lifecycle_spec = importlib.util.spec_from_file_location(
        "director_generation_lifecycle_reconciliation_test",
        core_dir / "generation_lifecycle.py",
    )
    lifecycle = importlib.util.module_from_spec(lifecycle_spec)
    lifecycle_spec.loader.exec_module(lifecycle)

    import sys

    sys.modules["director.core.generation_lifecycle"] = lifecycle
    reconciliation_spec = importlib.util.spec_from_file_location(
        "director_generation_reconciliation_contract_test",
        core_dir / "generation_reconciliation.py",
    )
    reconciliation = importlib.util.module_from_spec(reconciliation_spec)
    reconciliation_spec.loader.exec_module(reconciliation)
    return lifecycle, reconciliation


def make_operation(lifecycle, state="planned", provider="kling"):
    operation = lifecycle.GenerationOperation(
        operation_id="genop:scene_video:0:test",
        kind="scene_video",
        provider=provider,
        state=state,
        updated_at_epoch=1000,
    )
    return operation


def test_planned_operation_can_submit():
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle)

    decision = reconciliation.classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1010,
        stale_after_seconds=300,
    )

    assert decision.classification == "healthy"
    assert decision.action == "retry_submission"
    assert decision.auto_execute is True


def test_fresh_submitting_without_provider_id_waits_instead_of_resubmitting():
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="submitting")
    operation.attempt_count = 1

    decision = reconciliation.classify_generation_operation(
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
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="submitting")
    operation.attempt_count = 1

    decision = reconciliation.classify_generation_operation(
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
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="submitted")
    operation.attempt_count = 1
    operation.provider_request_id = "provider-task-1"

    decision = reconciliation.classify_generation_operation(
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
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="materialized", provider="elevenlabs")
    operation.attempt_count = 1

    decision = reconciliation.classify_generation_operation(
        operation,
        provider_supports_resume=False,
        now_epoch=1400,
        stale_after_seconds=300,
    )

    assert decision.classification == "orphan_candidate"
    assert decision.action == "manual_review"
    assert decision.reason_code == "materialized_artifact_not_durable"


def test_discovered_durable_artifact_wins_over_provider_retry():
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="materialized")
    operation.attempt_count = 1
    operation.provider_request_id = "provider-task-1"

    decision = reconciliation.classify_generation_operation(
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
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="failed")
    operation.attempt_count = 1
    operation.recoverable = True
    operation.last_error_code = "scene_generation_failed"

    decision = reconciliation.classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )

    assert decision.classification == "unknown"
    assert decision.action == "manual_review"
    assert decision.reason_code == "provider_submission_may_have_succeeded"


def test_failed_non_resumable_provider_preserves_existing_retry_contract():
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="failed", provider="videodb")
    operation.attempt_count = 1
    operation.recoverable = True
    operation.last_error_code = "scene_generation_failed"

    decision = reconciliation.classify_generation_operation(
        operation,
        provider_supports_resume=False,
        now_epoch=1400,
        stale_after_seconds=300,
    )

    assert decision.classification == "healthy"
    assert decision.action == "retry_submission"
    assert decision.auto_execute is True


def test_safe_summary_does_not_expose_provider_request_id():
    lifecycle, reconciliation = load_modules()
    operation = make_operation(lifecycle, state="submitted")
    operation.provider_request_id = "sensitive-provider-task-id"

    decision = reconciliation.classify_generation_operation(
        operation,
        provider_supports_resume=True,
        now_epoch=1400,
        stale_after_seconds=300,
    )
    summary = reconciliation.safe_reconciliation_summary(decision)

    assert "provider_request_id" not in summary
    assert "sensitive-provider-task-id" not in str(summary)
