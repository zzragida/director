import hashlib
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

    This model tracks Director's own operation identity separately from any
    provider request ID. A provider request ID enables resume/reconciliation
    when the provider exposes a fetch/poll API, but it is not treated as proof
    of exactly-once submission.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    version: int = 1
    operation_id: str
    kind: str
    provider: str
    state: GenerationOperationState = GenerationOperationState.planned
    attempt_count: int = Field(default=0, ge=0)
    provider_request_id: Optional[str] = None
    artifact: Optional[Dict[str, Any]] = None
    last_error_code: Optional[str] = None
    recoverable: bool = True

    @property
    def is_persisted(self) -> bool:
        return self.state == GenerationOperationState.persisted and bool(self.artifact)

    @property
    def has_provider_resume_token(self) -> bool:
        return bool(self.provider_request_id) and not self.is_persisted


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


def create_operation(
    generation_run_id: str,
    *,
    kind: str,
    provider: str,
    index: Optional[int] = None,
) -> GenerationOperation:
    return GenerationOperation(
        operation_id=make_operation_id(
            generation_run_id,
            kind=kind,
            index=index,
        ),
        kind=kind,
        provider=provider,
    )


def begin_submission(operation: GenerationOperation) -> None:
    operation.attempt_count += 1
    operation.state = GenerationOperationState.submitting
    operation.last_error_code = None
    operation.recoverable = True


def begin_resume(operation: GenerationOperation) -> None:
    operation.attempt_count += 1
    operation.state = GenerationOperationState.submitted
    operation.last_error_code = None
    operation.recoverable = True


def record_provider_request(
    operation: GenerationOperation,
    provider_request_id: str,
) -> None:
    if not provider_request_id:
        return
    operation.provider_request_id = str(provider_request_id)
    operation.state = GenerationOperationState.submitted


def record_materialized(operation: GenerationOperation) -> None:
    operation.state = GenerationOperationState.materialized


def record_persisted(
    operation: GenerationOperation,
    artifact: Dict[str, Any],
) -> None:
    operation.artifact = dict(artifact)
    operation.state = GenerationOperationState.persisted
    operation.last_error_code = None
    operation.recoverable = False


def record_failure(
    operation: GenerationOperation,
    *,
    code: str,
    recoverable: bool = True,
) -> None:
    operation.state = GenerationOperationState.failed
    operation.last_error_code = code
    operation.recoverable = recoverable


def safe_operation_summary(operation: Optional[GenerationOperation]) -> Dict[str, Any]:
    if operation is None:
        return {}
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "provider": operation.provider,
        "state": str(operation.state),
        "attempt_count": operation.attempt_count,
        "provider_request_known": bool(operation.provider_request_id),
        "artifact_persisted": operation.is_persisted,
        "recoverable": operation.recoverable,
        "last_error_code": operation.last_error_code,
    }
