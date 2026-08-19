import os
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict

from director.core.generation_lifecycle import (
    GenerationOperation,
    GenerationOperationState,
)


DEFAULT_STALE_SECONDS = 300


class ReconciliationClassification(str, Enum):
    healthy = "healthy"
    persisted = "persisted"
    reconcile_required = "reconcile_required"
    orphan_candidate = "orphan_candidate"
    unknown = "unknown"


class ReconciliationAction(str, Enum):
    none = "none"
    wait = "wait"
    retry_submission = "retry_submission"
    resume_provider = "resume_provider"
    restore_persisted_artifact = "restore_persisted_artifact"
    manual_review = "manual_review"


class ReconciliationDecision(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    classification: ReconciliationClassification
    action: ReconciliationAction
    reason_code: str
    auto_execute: bool = False
    stale: Optional[bool] = None


def configured_stale_seconds() -> int:
    raw = os.getenv("GENERATION_RECONCILIATION_STALE_SECONDS")
    if raw is None:
        return DEFAULT_STALE_SECONDS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_STALE_SECONDS


def operation_age_seconds(
    operation: GenerationOperation,
    *,
    now_epoch: Optional[int] = None,
) -> Optional[int]:
    if operation.updated_at_epoch is None:
        return None
    current = int(now_epoch if now_epoch is not None else time.time())
    return max(0, current - int(operation.updated_at_epoch))


def _is_stale(
    operation: GenerationOperation,
    *,
    now_epoch: Optional[int],
    stale_after_seconds: int,
) -> Optional[bool]:
    age = operation_age_seconds(operation, now_epoch=now_epoch)
    if age is None:
        return None
    return age >= stale_after_seconds


def classify_generation_operation(
    operation: GenerationOperation,
    *,
    provider_supports_resume: bool,
    durable_artifact_found: bool = False,
    now_epoch: Optional[int] = None,
    stale_after_seconds: Optional[int] = None,
) -> ReconciliationDecision:
    """Classify whether a generation operation can execute automatically.

    The policy deliberately avoids resubmitting work when Director cannot prove
    whether a resumable provider accepted an earlier request. A durable artifact
    discovered by deterministic name always wins over provider retry/resume.
    """

    stale_after = stale_after_seconds or configured_stale_seconds()
    stale = _is_stale(
        operation,
        now_epoch=now_epoch,
        stale_after_seconds=stale_after,
    )

    if operation.is_persisted:
        return ReconciliationDecision(
            classification=ReconciliationClassification.persisted,
            action=ReconciliationAction.none,
            reason_code="artifact_already_persisted",
            auto_execute=False,
            stale=stale,
        )

    if durable_artifact_found:
        return ReconciliationDecision(
            classification=ReconciliationClassification.reconcile_required,
            action=ReconciliationAction.restore_persisted_artifact,
            reason_code="durable_artifact_checkpoint_missing",
            auto_execute=True,
            stale=stale,
        )

    state = operation.state

    if state == GenerationOperationState.planned:
        return ReconciliationDecision(
            classification=ReconciliationClassification.healthy,
            action=ReconciliationAction.retry_submission,
            reason_code="operation_not_started",
            auto_execute=True,
            stale=stale,
        )

    if state == GenerationOperationState.submitting:
        if operation.provider_request_id and provider_supports_resume:
            return ReconciliationDecision(
                classification=ReconciliationClassification.reconcile_required,
                action=ReconciliationAction.resume_provider,
                reason_code="provider_request_known",
                auto_execute=True,
                stale=stale,
            )
        if operation.provider_request_id:
            return ReconciliationDecision(
                classification=ReconciliationClassification.reconcile_required,
                action=ReconciliationAction.manual_review,
                reason_code="provider_request_not_resumable",
                auto_execute=False,
                stale=stale,
            )
        if stale is False:
            return ReconciliationDecision(
                classification=ReconciliationClassification.healthy,
                action=ReconciliationAction.wait,
                reason_code="submission_in_flight",
                auto_execute=False,
                stale=False,
            )
        return ReconciliationDecision(
            classification=ReconciliationClassification.unknown,
            action=ReconciliationAction.manual_review,
            reason_code="submission_outcome_unknown",
            auto_execute=False,
            stale=stale,
        )

    if state == GenerationOperationState.submitted:
        if operation.provider_request_id and provider_supports_resume:
            return ReconciliationDecision(
                classification=ReconciliationClassification.reconcile_required,
                action=ReconciliationAction.resume_provider,
                reason_code="submitted_provider_task_available",
                auto_execute=True,
                stale=stale,
            )
        return ReconciliationDecision(
            classification=ReconciliationClassification.unknown,
            action=ReconciliationAction.manual_review,
            reason_code=(
                "provider_request_not_resumable"
                if operation.provider_request_id
                else "submitted_without_provider_request_id"
            ),
            auto_execute=False,
            stale=stale,
        )

    if state == GenerationOperationState.materialized:
        if operation.provider_request_id and provider_supports_resume:
            return ReconciliationDecision(
                classification=ReconciliationClassification.reconcile_required,
                action=ReconciliationAction.resume_provider,
                reason_code="materialized_artifact_not_persisted",
                auto_execute=True,
                stale=stale,
            )
        return ReconciliationDecision(
            classification=ReconciliationClassification.orphan_candidate,
            action=ReconciliationAction.manual_review,
            reason_code="materialized_artifact_not_durable",
            auto_execute=False,
            stale=stale,
        )

    if state == GenerationOperationState.failed:
        if operation.provider_request_id and provider_supports_resume and operation.recoverable:
            return ReconciliationDecision(
                classification=ReconciliationClassification.reconcile_required,
                action=ReconciliationAction.resume_provider,
                reason_code="failed_provider_task_resumable",
                auto_execute=True,
                stale=stale,
            )
        if provider_supports_resume and operation.attempt_count > 0 and not operation.provider_request_id:
            return ReconciliationDecision(
                classification=ReconciliationClassification.unknown,
                action=ReconciliationAction.manual_review,
                reason_code="provider_submission_may_have_succeeded",
                auto_execute=False,
                stale=stale,
            )
        if operation.recoverable:
            return ReconciliationDecision(
                classification=ReconciliationClassification.healthy,
                action=ReconciliationAction.retry_submission,
                reason_code="recoverable_failure_without_resume_token",
                auto_execute=True,
                stale=stale,
            )
        return ReconciliationDecision(
            classification=ReconciliationClassification.reconcile_required,
            action=ReconciliationAction.manual_review,
            reason_code="non_recoverable_generation_failure",
            auto_execute=False,
            stale=stale,
        )

    return ReconciliationDecision(
        classification=ReconciliationClassification.unknown,
        action=ReconciliationAction.manual_review,
        reason_code="unknown_operation_state",
        auto_execute=False,
        stale=stale,
    )


def safe_reconciliation_summary(decision: ReconciliationDecision) -> dict:
    return {
        "classification": str(decision.classification),
        "action": str(decision.action),
        "reason_code": decision.reason_code,
        "auto_execute": decision.auto_execute,
        "stale": decision.stale,
    }
