import json
import os
import time
import uuid
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.context_cas import atomic_update_context


LEASE_CONTEXT_KEY = "__generation_operation_leases__"
FENCE_CONTEXT_KEY = "__generation_fence_counters__"
LEASE_VERSION = 2
DEFAULT_LEASE_TTL_SECONDS = 900
MIN_LEASE_TTL_SECONDS = 30


class GenerationLeaseLostError(RuntimeError):
    """Raised when an execution no longer owns its generation lease."""

    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


class GenerationLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = LEASE_VERSION
    operation_id: str
    owner_id: str
    lease_token: str
    fencing_token: int = Field(default=0, ge=0)
    acquired_at_epoch: int = Field(ge=0)
    heartbeat_at_epoch: int = Field(ge=0)
    expires_at_epoch: int = Field(ge=0)

    def is_expired(self, now_epoch: int) -> bool:
        return int(now_epoch) >= self.expires_at_epoch


class LeaseAcquireResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    acquired: bool
    lease: Optional[GenerationLease] = None
    reason_code: str
    replaced_expired_lease: bool = False


def make_lease_owner_id() -> str:
    worker_id = os.getenv("DIRECTOR_WORKER_ID", "worker").strip() or "worker"
    return f"{worker_id}:{uuid.uuid4().hex[:16]}"


def get_lease_ttl_seconds() -> int:
    raw = os.getenv("GENERATION_LEASE_TTL_SECONDS")
    if raw is None:
        return DEFAULT_LEASE_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LEASE_TTL_SECONDS
    return max(MIN_LEASE_TTL_SECONDS, value)


def _read_json_document(context: Dict, key: str) -> Dict[str, object]:
    messages = context.get(key, [])
    if not messages or not isinstance(messages, list):
        return {}
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if not isinstance(content, str) or not content:
        return {}
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_document(context: Dict, key: str, document: Dict[str, object]) -> Dict:
    context[key] = [
        {
            "role": "system",
            "content": json.dumps(
                document,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ),
        }
    ]
    return context


def validate_fencing_lease_in_context(
    context: Dict,
    lease: GenerationLease,
    *,
    now_epoch: int,
) -> Optional[str]:
    """Return a stable failure code when a lease is no longer the current fence."""

    lease_document = _read_json_document(context, LEASE_CONTEXT_KEY)
    fence_document = _read_json_document(context, FENCE_CONTEXT_KEY)
    raw = lease_document.get(lease.operation_id)
    if raw is None:
        return "lease_missing"
    try:
        current = GenerationLease.model_validate(raw)
    except Exception:
        return "lease_record_invalid"

    current_fence = fence_document.get(lease.operation_id)
    try:
        current_fence = int(current_fence)
    except (TypeError, ValueError):
        return "fence_counter_missing"

    if current_fence != lease.fencing_token:
        return "stale_fencing_token"
    if current.fencing_token != lease.fencing_token:
        return "stale_fencing_token"
    if current.lease_token != lease.lease_token or current.owner_id != lease.owner_id:
        return "lease_token_mismatch"
    if current.is_expired(now_epoch):
        return "lease_expired"
    return None


def safe_lease_summary(
    lease: Optional[GenerationLease],
    *,
    now_epoch: Optional[int] = None,
) -> Dict:
    if lease is None:
        return {"locked": False}
    now = int(time.time()) if now_epoch is None else int(now_epoch)
    return {
        "locked": not lease.is_expired(now),
        "operation_id": lease.operation_id,
        "expires_at_epoch": lease.expires_at_epoch,
        "heartbeat_age_seconds": max(0, now - lease.heartbeat_at_epoch),
        "fencing_enabled": lease.fencing_token > 0,
    }


class GenerationLeaseStore:
    """CAS-backed durable operation leases with DB-time fencing tokens."""

    def __init__(self, session, *, max_cas_attempts: int = 8):
        self.session = session
        self.max_cas_attempts = max_cas_attempts

    @staticmethod
    def _read_document(context: Dict) -> Dict[str, dict]:
        return _read_json_document(context, LEASE_CONTEXT_KEY)

    @staticmethod
    def _write_document(context: Dict, document: Dict[str, dict]) -> Dict:
        return _write_json_document(context, LEASE_CONTEXT_KEY, document)

    @staticmethod
    def _read_fence_document(context: Dict) -> Dict[str, int]:
        return _read_json_document(context, FENCE_CONTEXT_KEY)

    @staticmethod
    def _write_fence_document(context: Dict, document: Dict[str, int]) -> Dict:
        return _write_json_document(context, FENCE_CONTEXT_KEY, document)

    def _authoritative_now(self, now_epoch: Optional[int] = None) -> int:
        if now_epoch is not None:
            return int(now_epoch)
        current_epoch = getattr(self.session.db, "current_epoch", None)
        if callable(current_epoch):
            try:
                return int(current_epoch())
            except (AttributeError, NotImplementedError):
                pass
        # Compatibility fallback for non-production test doubles/backends.
        return int(time.time())

    def get(self, operation_id: str) -> Optional[GenerationLease]:
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        raw = self._read_document(context).get(operation_id)
        if raw is None:
            return None
        try:
            return GenerationLease.model_validate(raw)
        except Exception:
            return None

    def get_fencing_token(self, operation_id: str) -> int:
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        raw = self._read_fence_document(context).get(operation_id, 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def acquire(
        self,
        operation_id: str,
        *,
        owner_id: str,
        ttl_seconds: Optional[int] = None,
        now_epoch: Optional[int] = None,
    ) -> LeaseAcquireResult:
        now = self._authoritative_now(now_epoch)
        ttl = max(
            MIN_LEASE_TTL_SECONDS,
            int(ttl_seconds if ttl_seconds is not None else get_lease_ttl_seconds()),
        )
        result = LeaseAcquireResult(acquired=False, reason_code="operation_locked")

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal result
            document = self._read_document(context)
            fence_document = self._read_fence_document(context)
            raw_existing = document.get(operation_id)
            existing = None
            if raw_existing is not None:
                try:
                    existing = GenerationLease.model_validate(raw_existing)
                except Exception:
                    result = LeaseAcquireResult(
                        acquired=False,
                        reason_code="lease_record_invalid",
                    )
                    return None

            if existing is not None and not existing.is_expired(now):
                result = LeaseAcquireResult(
                    acquired=False,
                    lease=existing,
                    reason_code="operation_locked",
                )
                return None

            try:
                previous_fence = int(fence_document.get(operation_id, 0))
            except (TypeError, ValueError):
                result = LeaseAcquireResult(
                    acquired=False,
                    reason_code="fence_counter_invalid",
                )
                return None
            if existing is not None:
                previous_fence = max(previous_fence, int(existing.fencing_token or 0))
            fencing_token = previous_fence + 1

            lease = GenerationLease(
                operation_id=operation_id,
                owner_id=owner_id,
                lease_token=uuid.uuid4().hex,
                fencing_token=fencing_token,
                acquired_at_epoch=now,
                heartbeat_at_epoch=now,
                expires_at_epoch=now + ttl,
            )
            document[operation_id] = lease.model_dump(mode="json")
            fence_document[operation_id] = fencing_token
            self._write_fence_document(context, fence_document)
            result = LeaseAcquireResult(
                acquired=True,
                lease=lease,
                reason_code=(
                    "expired_lease_replaced" if existing is not None else "lease_acquired"
                ),
                replaced_expired_lease=existing is not None,
            )
            return self._write_document(context, document)

        atomic_update_context(
            self.session.db,
            self.session.session_id,
            mutate,
            max_attempts=self.max_cas_attempts,
        )
        return result

    def assert_current(
        self,
        lease: GenerationLease,
        *,
        now_epoch: Optional[int] = None,
    ) -> None:
        now = self._authoritative_now(now_epoch)
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        reason = validate_fencing_lease_in_context(context, lease, now_epoch=now)
        if reason is not None:
            raise GenerationLeaseLostError(reason)

    def heartbeat(
        self,
        lease: GenerationLease,
        *,
        ttl_seconds: Optional[int] = None,
        now_epoch: Optional[int] = None,
    ) -> GenerationLease:
        now = self._authoritative_now(now_epoch)
        ttl = max(
            MIN_LEASE_TTL_SECONDS,
            int(ttl_seconds if ttl_seconds is not None else get_lease_ttl_seconds()),
        )
        refreshed: Optional[GenerationLease] = None
        lost_reason: Optional[str] = None

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal refreshed, lost_reason
            reason = validate_fencing_lease_in_context(context, lease, now_epoch=now)
            if reason is not None:
                lost_reason = reason
                return None

            document = self._read_document(context)
            current = GenerationLease.model_validate(document[lease.operation_id])
            current.version = LEASE_VERSION
            current.heartbeat_at_epoch = now
            current.expires_at_epoch = now + ttl
            document[lease.operation_id] = current.model_dump(mode="json")
            refreshed = current
            return self._write_document(context, document)

        atomic_update_context(
            self.session.db,
            self.session.session_id,
            mutate,
            max_attempts=self.max_cas_attempts,
        )
        if refreshed is None:
            raise GenerationLeaseLostError(lost_reason or "lease_lost")
        return refreshed

    def release(self, lease: GenerationLease) -> bool:
        released = False
        now = self._authoritative_now()

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal released
            reason = validate_fencing_lease_in_context(context, lease, now_epoch=now)
            if reason is not None and reason != "lease_expired":
                return None
            document = self._read_document(context)
            raw = document.get(lease.operation_id)
            if raw is None:
                return None
            try:
                current = GenerationLease.model_validate(raw)
            except Exception:
                return None
            if (
                current.lease_token != lease.lease_token
                or current.owner_id != lease.owner_id
                or current.fencing_token != lease.fencing_token
            ):
                return None
            document.pop(lease.operation_id, None)
            released = True
            # Deliberately retain FENCE_CONTEXT_KEY so the next owner receives N+1.
            return self._write_document(context, document)

        atomic_update_context(
            self.session.db,
            self.session.session_id,
            mutate,
            max_attempts=self.max_cas_attempts,
        )
        return released
