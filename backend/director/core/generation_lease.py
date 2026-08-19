import json
import os
import time
import uuid
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.context_cas import atomic_update_context


LEASE_CONTEXT_KEY = "__generation_operation_leases__"
LEASE_VERSION = 1
DEFAULT_LEASE_TTL_SECONDS = 900
MIN_LEASE_TTL_SECONDS = 30


class GenerationLeaseLostError(RuntimeError):
    """Raised when an execution no longer owns its generation lease."""


class GenerationLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = LEASE_VERSION
    operation_id: str
    owner_id: str
    lease_token: str
    acquired_at_epoch: int = Field(ge=0)
    heartbeat_at_epoch: int = Field(ge=0)
    expires_at_epoch: int = Field(ge=0)

    def is_expired(self, now_epoch: Optional[int] = None) -> bool:
        now = int(time.time()) if now_epoch is None else int(now_epoch)
        return now >= self.expires_at_epoch


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
    }


class GenerationLeaseStore:
    """CAS-backed durable operation leases stored in the session context row."""

    def __init__(self, session, *, max_cas_attempts: int = 8):
        self.session = session
        self.max_cas_attempts = max_cas_attempts

    @staticmethod
    def _read_document(context: Dict) -> Dict[str, dict]:
        messages = context.get(LEASE_CONTEXT_KEY, [])
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

    @staticmethod
    def _write_document(context: Dict, document: Dict[str, dict]) -> Dict:
        content = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        context[LEASE_CONTEXT_KEY] = [
            {"role": "system", "content": content}
        ]
        return context

    def get(self, operation_id: str) -> Optional[GenerationLease]:
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        raw = self._read_document(context).get(operation_id)
        if raw is None:
            return None
        try:
            return GenerationLease.model_validate(raw)
        except Exception:
            return None

    def acquire(
        self,
        operation_id: str,
        *,
        owner_id: str,
        ttl_seconds: Optional[int] = None,
        now_epoch: Optional[int] = None,
    ) -> LeaseAcquireResult:
        now = int(time.time()) if now_epoch is None else int(now_epoch)
        ttl = max(
            MIN_LEASE_TTL_SECONDS,
            int(ttl_seconds if ttl_seconds is not None else get_lease_ttl_seconds()),
        )
        result = LeaseAcquireResult(
            acquired=False,
            reason_code="operation_locked",
        )

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal result
            document = self._read_document(context)
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

            lease = GenerationLease(
                operation_id=operation_id,
                owner_id=owner_id,
                lease_token=uuid.uuid4().hex,
                acquired_at_epoch=now,
                heartbeat_at_epoch=now,
                expires_at_epoch=now + ttl,
            )
            document[operation_id] = lease.model_dump(mode="json")
            result = LeaseAcquireResult(
                acquired=True,
                lease=lease,
                reason_code=(
                    "expired_lease_replaced"
                    if existing is not None
                    else "lease_acquired"
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

    def heartbeat(
        self,
        lease: GenerationLease,
        *,
        ttl_seconds: Optional[int] = None,
        now_epoch: Optional[int] = None,
    ) -> GenerationLease:
        now = int(time.time()) if now_epoch is None else int(now_epoch)
        ttl = max(
            MIN_LEASE_TTL_SECONDS,
            int(ttl_seconds if ttl_seconds is not None else get_lease_ttl_seconds()),
        )
        refreshed: Optional[GenerationLease] = None
        lost_reason: Optional[str] = None

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal refreshed, lost_reason
            document = self._read_document(context)
            raw = document.get(lease.operation_id)
            if raw is None:
                lost_reason = "lease_missing"
                return None
            try:
                current = GenerationLease.model_validate(raw)
            except Exception:
                lost_reason = "lease_record_invalid"
                return None

            if current.lease_token != lease.lease_token or current.owner_id != lease.owner_id:
                lost_reason = "lease_token_mismatch"
                return None
            if current.is_expired(now):
                lost_reason = "lease_expired"
                return None

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

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal released
            document = self._read_document(context)
            raw = document.get(lease.operation_id)
            if raw is None:
                return None
            try:
                current = GenerationLease.model_validate(raw)
            except Exception:
                return None
            if current.lease_token != lease.lease_token or current.owner_id != lease.owner_id:
                return None
            document.pop(lease.operation_id, None)
            released = True
            return self._write_document(context, document)

        atomic_update_context(
            self.session.db,
            self.session.session_id,
            mutate,
            max_attempts=self.max_cas_attempts,
        )
        return released
