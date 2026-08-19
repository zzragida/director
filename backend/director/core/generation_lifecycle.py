import hashlib
import time
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerationOperationState(str, Enum):
    planned = "planned"
    submitting = "submitting"
    submitted = "submitted"
    materialized = "materialized"
    persisted = "persisted"
    failed = "failed"


class GenerationOperation(BaseModel):
    """Durable lifecycle metadata for one external generation operation.

    Director's operation identity is intentionally separate from any provider
    request ID. Provider request IDs enable resume/reconciliation when a
    provider exposes a fetch/poll API, but they are not proof of exactly-once
    submission.

    ``attempt_count`` is retained for backwards compatibility and counts both
    new provider submissions and provider resumes. Cost attribution must use
    ``submission_count`` instead because a resume/poll is not necessarily a new
    billable generation request. Legacy v2 checkpoints intentionally keep
    ``accounting_complete=False`` rather than guessing their historical split.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    version: int = 3
    operation_id: str
    kind: str
    provider: str
    state: GenerationOperationState = GenerationOperationState.planned
    attempt_count: int = Field(default=0, ge=0)
    submission_count: Optional[int] = Field(default=None, ge=0)
    resume_count: Optional[int] = Field(default=None, ge=0)
    accounting_complete: bool = False
    provider_request_id: Optional[str] = None
    artifact: Optional[Dict[str, Any]] = None
    last_error_code: Optional[str] = None
    recoverable: bool = True
    updated_at_epoch: Optional[int] = None

    @property
    def is_persisted(self) -> bool:
        return self.state == GenerationOperationState.persisted and bool(self.artifact)

    @property
    def has_provider_resume_token(self) -> bool:
        return bool(self.provider_request_id) and not self.is_persisted

    @property
    def accounting_known(self) -> bool:
        return bool(
            self.accounting_complete
            and self.submission_count is not None
            and self.resume_count is not None
        )


def _now_epoch() -> int:
    return int(time.time())


def touch_operation(operation: GenerationOperation, *, now_epoch: Optional[int] = None) -> None:
    operation.updated_at_epoch = int(now_epoch if now_epoch is not None else _now_epoch())


def make_generation_run_id(request_fingerprint: str) -> str:
    return f"genrun:{request_fingerprint[:24]}"


def make_operation_id(
    generation_run_id: str,
    *,
    kind: str,
    index: Optional[int] = None,
) -> str:
    payload = f"{generation_run_id}|{kind}|{'' if index is None else index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    suffix = f":{index}" if index is not None else ""
    return f"genop:{kind}{suffix}:{digest}"


def make_artifact_name(operation_id: str, media_type: str) -> str:
    """Return a deterministic VideoDB name for recoverable persisted artifacts."""

    digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:24]
    return f"director-{media_type}-{digest}"


def create_operation(
    generation_run_id: str,
    *,
    kind: str,
    provider: str,
    index: Optional[int] = None,
) -> GenerationOperation:
    operation = GenerationOperation(
        operation_id=make_operation_id(
            generation_run_id,
            kind=kind,
            index=index,
        ),
        kind=kind,
        provider=provider,
        submission_count=0,
        resume_count=0,
        accounting_complete=True,
    )
    touch_operation(operation)
    return operation


def _ensure_v3_accounting(operation: GenerationOperation) -> None:
    """Start exact counters from now without inventing legacy history."""

    if operation.version < 3:
        operation.version = 3
    if operation.submission_count is None:
        operation.submission_count = 0
    if operation.resume_count is None:
        operation.resume_count = 0
    # Do not set accounting_complete here. A legacy operation can start exact
    # counters from this call onward while its earlier attempt history remains
    # unknowable.


def begin_submission(operation: GenerationOperation) -> None:
    _ensure_v3_accounting(operation)
    operation.attempt_count += 1
    operation.submission_count += 1
    operation.state = GenerationOperationState.submitting
    operation.last_error_code = None
    operation.recoverable = True
    touch_operation(operation)


def begin_resume(operation: GenerationOperation) -> None:
    _ensure_v3_accounting(operation)
    operation.attempt_count += 1
    operation.resume_count += 1
    operation.state = GenerationOperationState.submitted
    operation.last_error_code = None
    operation.recoverable = True
    touch_operation(operation)


def record_provider_request(
    operation: GenerationOperation,
    provider_request_id: str,
) -> None:
    if not provider_request_id:
        return
    operation.provider_request_id = str(provider_request_id)
    operation.state = GenerationOperationState.submitted
    touch_operation(operation)


def record_materialized(operation: GenerationOperation) -> None:
    operation.state = GenerationOperationState.materialized
    touch_operation(operation)


def record_persisted(
    operation: GenerationOperation,
    artifact: Dict[str, Any],
) -> None:
    operation.artifact = dict(artifact)
    operation.state = GenerationOperationState.persisted
    operation.last_error_code = None
    operation.recoverable = False
    touch_operation(operation)


def record_failure(
    operation: GenerationOperation,
    *,
    code: str,
    recoverable: bool = True,
) -> None:
    operation.state = GenerationOperationState.failed
    operation.last_error_code = code
    operation.recoverable = recoverable
    touch_operation(operation)


def safe_operation_summary(operation: Optional[GenerationOperation]) -> Dict[str, Any]:
    if operation is None:
        return {}
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "provider": operation.provider,
        "state": str(operation.state),
        "attempt_count": operation.attempt_count,
        "submission_count": operation.submission_count,
        "resume_count": operation.resume_count,
        "accounting_known": operation.accounting_known,
        "provider_request_known": bool(operation.provider_request_id),
        "artifact_persisted": operation.is_persisted,
        "recoverable": operation.recoverable,
        "last_error_code": operation.last_error_code,
        "updated_at_epoch": operation.updated_at_epoch,
    }
