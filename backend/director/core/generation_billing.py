import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.context_cas import atomic_update_context
from director.core.generation_provenance import stable_digest


BILLING_CONTEXT_KEY = "__generation_billing_events__"
BILLING_EVENT_VERSION = 1
BILLING_RECONCILIATION_VERSION = 1


class BillingEventKind(str, Enum):
    charge = "charge"
    credit = "credit"


class BillingEventStatus(str, Enum):
    pending = "pending"
    finalized = "finalized"


class BillingEventSource(str, Enum):
    provider_usage = "provider_usage"
    invoice = "invoice"
    manual_adjustment = "manual_adjustment"


class ProviderBillingEvent(BaseModel):
    """Normalized provider usage/invoice event.

    Director does not fetch provider invoices itself in this contract. Provider
    adapters/importers can normalize authoritative billing records into this
    model without storing raw invoice payloads in session context.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    version: int = BILLING_EVENT_VERSION
    event_id: str
    provider: str
    generation_run_id: str
    operation_id: Optional[str] = None
    reservation_id: Optional[str] = None
    source: BillingEventSource = BillingEventSource.provider_usage
    status: BillingEventStatus = BillingEventStatus.finalized
    kind: BillingEventKind = BillingEventKind.charge
    amount_micros: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    invoice_id: Optional[str] = None
    occurred_at_epoch: Optional[int] = None

    @property
    def event_key(self) -> str:
        return f"{self.provider}:{self.event_id}"

    @property
    def signed_amount_micros(self) -> int:
        return self.amount_micros if self.kind == BillingEventKind.charge else -self.amount_micros


class BillingLedgerConflictError(RuntimeError):
    """Raised when the same provider event ID is reused with different data."""


class GenerationBillingReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = BILLING_RECONCILIATION_VERSION
    generation_run_id: str
    currency: str
    finalized_event_count: int = Field(default=0, ge=0)
    pending_event_count: int = Field(default=0, ge=0)
    matched_reservation_count: int = Field(default=0, ge=0)
    unresolved_reservation_count: int = Field(default=0, ge=0)
    unmatched_finalized_event_count: int = Field(default=0, ge=0)
    estimated_reserved_micros: int = 0
    actual_finalized_micros: int = 0
    unmatched_actual_micros: int = 0
    effective_exposure_micros: int = 0
    reconciliation_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"reconciliation_digest"})
        self.reconciliation_digest = stable_digest(payload)


def _read_document(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    messages = context.get(BILLING_CONTEXT_KEY, [])
    if not messages:
        return {}
    last = messages[-1] if isinstance(messages[-1], dict) else None
    content = last.get("content") if isinstance(last, dict) else None
    if not isinstance(content, str) or not content:
        return {}
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_document(context: Dict[str, Any], document: Dict[str, Any]) -> None:
    content = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    context[BILLING_CONTEXT_KEY] = [{"role": "system", "content": content}]


def billing_events_from_context(context: Dict[str, Any]) -> List[ProviderBillingEvent]:
    events: List[ProviderBillingEvent] = []
    for raw in _read_document(context).values():
        try:
            events.append(ProviderBillingEvent.model_validate(raw))
        except Exception:
            continue
    return events


class GenerationBillingLedgerStore:
    """CAS-backed idempotent store for normalized billing events."""

    def __init__(self, session, *, max_cas_attempts: int = 8):
        self.session = session
        self.max_cas_attempts = max_cas_attempts

    def record(self, event: ProviderBillingEvent) -> bool:
        inserted = False
        conflict = False

        def mutate(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            nonlocal inserted, conflict
            document = _read_document(context)
            existing = document.get(event.event_key)
            serialized = event.model_dump(mode="json")
            if existing is not None:
                if existing != serialized:
                    conflict = True
                return None
            document[event.event_key] = serialized
            _write_document(context, document)
            inserted = True
            return context

        atomic_update_context(
            self.session.db,
            self.session.session_id,
            mutate,
            max_attempts=self.max_cas_attempts,
        )
        if conflict:
            raise BillingLedgerConflictError("billing_event_conflict")
        return inserted

    def events_for_run(self, generation_run_id: str) -> List[ProviderBillingEvent]:
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        return [
            event
            for event in billing_events_from_context(context)
            if event.generation_run_id == generation_run_id
        ]


def reconcile_generation_billing(
    *,
    generation_run_id: str,
    currency: str,
    reservations: Iterable[Any],
    events: Iterable[ProviderBillingEvent],
) -> GenerationBillingReconciliation:
    normalized_currency = currency.upper()
    run_events = [
        event
        for event in events
        if event.generation_run_id == generation_run_id
        and event.currency.upper() == normalized_currency
    ]
    finalized = [event for event in run_events if event.status == BillingEventStatus.finalized]
    pending = [event for event in run_events if event.status == BillingEventStatus.pending]

    reservations_list = [
        reservation
        for reservation in reservations
        if getattr(reservation, "generation_run_id", None) == generation_run_id
        and str(getattr(reservation, "currency", "")).upper() == normalized_currency
    ]
    reservation_ids = {
        getattr(reservation, "reservation_id", None)
        for reservation in reservations_list
        if getattr(reservation, "reservation_id", None)
    }

    actual_by_reservation: Dict[str, int] = {}
    for event in finalized:
        if event.reservation_id:
            actual_by_reservation[event.reservation_id] = (
                actual_by_reservation.get(event.reservation_id, 0)
                + event.signed_amount_micros
            )

    estimated_reserved = 0
    effective_exposure = 0
    matched = 0
    unresolved = 0
    for reservation in reservations_list:
        estimated = getattr(reservation, "estimated_amount_micros", None)
        if estimated is not None:
            estimated_reserved += int(estimated)
        reservation_id = getattr(reservation, "reservation_id", None)
        if reservation_id and reservation_id in actual_by_reservation:
            matched += 1
            effective_exposure += actual_by_reservation[reservation_id]
        else:
            unresolved += 1
            effective_exposure += int(estimated or 0)

    unmatched_events = [
        event
        for event in finalized
        if not event.reservation_id or event.reservation_id not in reservation_ids
    ]
    unmatched_actual = sum(event.signed_amount_micros for event in unmatched_events)
    actual_finalized = sum(event.signed_amount_micros for event in finalized)
    effective_exposure += unmatched_actual

    reconciliation = GenerationBillingReconciliation(
        generation_run_id=generation_run_id,
        currency=normalized_currency,
        finalized_event_count=len(finalized),
        pending_event_count=len(pending),
        matched_reservation_count=matched,
        unresolved_reservation_count=unresolved,
        unmatched_finalized_event_count=len(unmatched_events),
        estimated_reserved_micros=estimated_reserved,
        actual_finalized_micros=actual_finalized,
        unmatched_actual_micros=unmatched_actual,
        effective_exposure_micros=max(effective_exposure, 0),
    )
    reconciliation.refresh_digest()
    return reconciliation


def safe_billing_summary(reconciliation: GenerationBillingReconciliation) -> Dict[str, Any]:
    return {
        "generation_run_id": reconciliation.generation_run_id,
        "currency": reconciliation.currency,
        "finalized_event_count": reconciliation.finalized_event_count,
        "pending_event_count": reconciliation.pending_event_count,
        "matched_reservation_count": reconciliation.matched_reservation_count,
        "unresolved_reservation_count": reconciliation.unresolved_reservation_count,
        "unmatched_finalized_event_count": reconciliation.unmatched_finalized_event_count,
        "actual_finalized_micros": reconciliation.actual_finalized_micros,
        "effective_exposure_micros": reconciliation.effective_exposure_micros,
        "reconciliation_digest": reconciliation.reconciliation_digest,
    }
