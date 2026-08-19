import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_billing import (
    BillingEventStatus,
    ProviderBillingEvent,
    billing_events_from_context,
    reconcile_generation_billing,
)
from director.core.generation_budget import (
    BudgetConfigurationError,
    BudgetReservation,
    GenerationBudgetPolicy,
    budget_reservations_from_context,
    load_budget_configuration_from_env,
)
from director.core.generation_provenance import stable_digest


SPEND_STATUS_VERSION = 1
_CHECKPOINT_CONTEXT_KEY = "__text_to_movie_checkpoints__"


class AppliedPolicyIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    policy_version: str
    policy_digest: Optional[str] = None
    rate_card_id: Optional[str] = None
    rate_card_version: Optional[str] = None
    rate_card_digest: Optional[str] = None
    first_seen_epoch: int = Field(ge=0)
    last_seen_epoch: int = Field(ge=0)
    reservation_count: int = Field(ge=1)


class CurrentBudgetPolicyStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    configuration_error: Optional[str] = None
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    policy_digest: Optional[str] = None
    currency: Optional[str] = None
    max_total_micros: Optional[int] = Field(default=None, ge=0)
    max_retry_micros: Optional[int] = Field(default=None, ge=0)
    max_submissions_per_operation: Optional[int] = Field(default=None, ge=1)
    max_retry_submissions_per_operation: Optional[int] = Field(default=None, ge=0)
    require_preflight_pricing: Optional[bool] = None
    allow_legacy_accounting: Optional[bool] = None
    rate_card_id: Optional[str] = None
    rate_card_version: Optional[str] = None
    rate_card_digest: Optional[str] = None


class ProviderSpendSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    reservation_count: int = Field(default=0, ge=0)
    submission_count: int = Field(default=0, ge=0)
    retry_submission_count: int = Field(default=0, ge=0)
    estimated_reserved_micros: int = Field(default=0, ge=0)
    finalized_actual_micros: int = 0
    pending_billing_event_count: int = Field(default=0, ge=0)
    finalized_billing_event_count: int = Field(default=0, ge=0)


class SpendCircuitBreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    reason_codes: List[str] = Field(default_factory=list)


class GenerationSpendRunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = SPEND_STATUS_VERSION
    generation_run_id: str
    checkpoint_id: Optional[str] = None
    checkpoint_status: Optional[str] = None
    last_block_reason: Optional[str] = None
    currency: Optional[str] = None
    reservation_count: int = Field(default=0, ge=0)
    submission_count: int = Field(default=0, ge=0)
    retry_submission_count: int = Field(default=0, ge=0)
    estimated_reserved_micros: int = Field(default=0, ge=0)
    actual_finalized_micros: int = 0
    effective_exposure_micros: int = Field(default=0, ge=0)
    retry_exposure_micros: int = Field(default=0, ge=0)
    remaining_total_micros: Optional[int] = Field(default=None, ge=0)
    remaining_retry_micros: Optional[int] = Field(default=None, ge=0)
    actual_minus_estimated_micros: int = 0
    variance_complete: bool = False
    finalized_billing_event_count: int = Field(default=0, ge=0)
    pending_billing_event_count: int = Field(default=0, ge=0)
    unmatched_finalized_event_count: int = Field(default=0, ge=0)
    unresolved_reservation_count: int = Field(default=0, ge=0)
    policy: CurrentBudgetPolicyStatus
    policy_history: List[AppliedPolicyIdentity] = Field(default_factory=list)
    providers: List[ProviderSpendSummary] = Field(default_factory=list)
    circuit_breaker: SpendCircuitBreaker
    status_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"status_digest"})
        self.status_digest = stable_digest(payload)


class GenerationSpendSessionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = SPEND_STATUS_VERSION
    session_id: str
    run_count: int = Field(default=0, ge=0)
    runs: List[GenerationSpendRunStatus] = Field(default_factory=list)
    status_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"status_digest"})
        self.status_digest = stable_digest(payload)


def _read_checkpoint_document(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    messages = context.get(_CHECKPOINT_CONTEXT_KEY, [])
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


def _checkpoint_for_run(
    context: Dict[str, Any], generation_run_id: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    for checkpoint_id, raw in _read_checkpoint_document(context).items():
        if isinstance(raw, dict) and raw.get("generation_run_id") == generation_run_id:
            return checkpoint_id, raw
    return None, None


def _run_ids(context: Dict[str, Any]) -> List[str]:
    run_ids = set()
    for raw in _read_checkpoint_document(context).values():
        if isinstance(raw, dict) and raw.get("generation_run_id"):
            run_ids.add(str(raw["generation_run_id"]))
    for reservation in budget_reservations_from_context(context):
        run_ids.add(reservation.generation_run_id)
    for event in billing_events_from_context(context):
        run_ids.add(event.generation_run_id)
    return sorted(run_ids)


def _load_current_policy_status() -> Tuple[CurrentBudgetPolicyStatus, Optional[GenerationBudgetPolicy]]:
    try:
        policy, rate_card = load_budget_configuration_from_env()
    except BudgetConfigurationError as exc:
        return CurrentBudgetPolicyStatus(
            enabled=True,
            configuration_error=str(exc),
        ), None
    if policy is None:
        return CurrentBudgetPolicyStatus(enabled=False), None
    return (
        CurrentBudgetPolicyStatus(
            enabled=True,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            policy_digest=policy.policy_digest,
            currency=policy.currency.upper(),
            max_total_micros=policy.max_total_micros,
            max_retry_micros=policy.max_retry_micros,
            max_submissions_per_operation=policy.max_submissions_per_operation,
            max_retry_submissions_per_operation=policy.max_retry_submissions_per_operation,
            require_preflight_pricing=policy.require_preflight_pricing,
            allow_legacy_accounting=policy.allow_legacy_accounting,
            rate_card_id=rate_card.rate_card_id if rate_card else None,
            rate_card_version=rate_card.rate_card_version if rate_card else None,
            rate_card_digest=rate_card.rate_card_digest if rate_card else None,
        ),
        policy,
    )


def _policy_history(reservations: Iterable[BudgetReservation]) -> List[AppliedPolicyIdentity]:
    grouped: Dict[Tuple[Any, ...], List[BudgetReservation]] = {}
    for reservation in reservations:
        key = (
            reservation.policy_id,
            reservation.policy_version,
            reservation.policy_digest,
            reservation.rate_card_id,
            reservation.rate_card_version,
            reservation.rate_card_digest,
        )
        grouped.setdefault(key, []).append(reservation)

    history: List[AppliedPolicyIdentity] = []
    for key, items in grouped.items():
        times = [int(item.created_at_epoch) for item in items]
        history.append(
            AppliedPolicyIdentity(
                policy_id=key[0],
                policy_version=key[1],
                policy_digest=key[2],
                rate_card_id=key[3],
                rate_card_version=key[4],
                rate_card_digest=key[5],
                first_seen_epoch=min(times),
                last_seen_epoch=max(times),
                reservation_count=len(items),
            )
        )
    return sorted(history, key=lambda entry: (entry.first_seen_epoch, entry.policy_id))


def _retry_exposure(
    reservations: List[BudgetReservation],
    events: List[ProviderBillingEvent],
    *,
    generation_run_id: str,
    currency: str,
) -> int:
    retry_reservations = {
        reservation.reservation_id: reservation
        for reservation in reservations
        if reservation.generation_run_id == generation_run_id
        and reservation.retry_submission
        and reservation.currency.upper() == currency.upper()
    }
    actual_by_reservation: Dict[str, int] = {}
    for event in events:
        if (
            event.generation_run_id == generation_run_id
            and event.currency.upper() == currency.upper()
            and event.status == BillingEventStatus.finalized
            and event.reservation_id in retry_reservations
        ):
            actual_by_reservation[event.reservation_id] = (
                actual_by_reservation.get(event.reservation_id, 0)
                + event.signed_amount_micros
            )
    exposure = 0
    for reservation_id, reservation in retry_reservations.items():
        exposure += actual_by_reservation.get(
            reservation_id, int(reservation.estimated_amount_micros or 0)
        )
    return max(exposure, 0)


def _provider_summaries(
    reservations: List[BudgetReservation],
    events: List[ProviderBillingEvent],
    generation_run_id: str,
) -> List[ProviderSpendSummary]:
    providers = {
        reservation.provider
        for reservation in reservations
        if reservation.generation_run_id == generation_run_id
    }
    providers.update(
        event.provider for event in events if event.generation_run_id == generation_run_id
    )
    summaries: List[ProviderSpendSummary] = []
    for provider in sorted(providers):
        provider_reservations = [
            reservation
            for reservation in reservations
            if reservation.generation_run_id == generation_run_id
            and reservation.provider == provider
        ]
        provider_events = [
            event
            for event in events
            if event.generation_run_id == generation_run_id
            and event.provider == provider
        ]
        summaries.append(
            ProviderSpendSummary(
                provider=provider,
                reservation_count=len(provider_reservations),
                submission_count=len(provider_reservations),
                retry_submission_count=sum(
                    1 for reservation in provider_reservations if reservation.retry_submission
                ),
                estimated_reserved_micros=sum(
                    int(reservation.estimated_amount_micros or 0)
                    for reservation in provider_reservations
                ),
                finalized_actual_micros=sum(
                    event.signed_amount_micros
                    for event in provider_events
                    if event.status == BillingEventStatus.finalized
                ),
                pending_billing_event_count=sum(
                    1 for event in provider_events if event.status == BillingEventStatus.pending
                ),
                finalized_billing_event_count=sum(
                    1 for event in provider_events if event.status == BillingEventStatus.finalized
                ),
            )
        )
    return summaries


def _circuit_breaker(
    policy_status: CurrentBudgetPolicyStatus,
    *,
    effective_exposure_micros: int,
    retry_exposure_micros: int,
) -> SpendCircuitBreaker:
    if not policy_status.enabled:
        return SpendCircuitBreaker(state="disabled")
    if policy_status.configuration_error:
        return SpendCircuitBreaker(
            state="configuration_error",
            reason_codes=["budget_configuration_invalid"],
        )
    reasons: List[str] = []
    if (
        policy_status.max_total_micros is not None
        and effective_exposure_micros >= policy_status.max_total_micros
    ):
        reasons.append("total_budget_reached")
    if (
        policy_status.max_retry_micros is not None
        and retry_exposure_micros >= policy_status.max_retry_micros
    ):
        reasons.append("retry_budget_reached")
    return SpendCircuitBreaker(
        state="open" if reasons else "closed",
        reason_codes=reasons,
    )


def build_generation_spend_run_status(
    context: Dict[str, Any],
    generation_run_id: str,
) -> GenerationSpendRunStatus:
    reservations = [
        reservation
        for reservation in budget_reservations_from_context(context)
        if reservation.generation_run_id == generation_run_id
    ]
    events = [
        event
        for event in billing_events_from_context(context)
        if event.generation_run_id == generation_run_id
    ]
    checkpoint_id, checkpoint = _checkpoint_for_run(context, generation_run_id)
    policy_status, current_policy = _load_current_policy_status()

    currency = policy_status.currency
    if currency is None and reservations:
        currency = reservations[0].currency.upper()
    if currency is None and events:
        currency = events[0].currency.upper()

    if currency:
        reconciliation = reconcile_generation_billing(
            generation_run_id=generation_run_id,
            currency=currency,
            reservations=reservations,
            events=events,
        )
        retry_exposure = _retry_exposure(
            reservations,
            events,
            generation_run_id=generation_run_id,
            currency=currency,
        )
        estimated = reconciliation.estimated_reserved_micros
        actual = reconciliation.actual_finalized_micros
        effective = reconciliation.effective_exposure_micros
        finalized_count = reconciliation.finalized_event_count
        pending_count = reconciliation.pending_event_count
        unmatched_count = reconciliation.unmatched_finalized_event_count
        unresolved_count = reconciliation.unresolved_reservation_count
    else:
        retry_exposure = 0
        estimated = 0
        actual = 0
        effective = 0
        finalized_count = 0
        pending_count = 0
        unmatched_count = 0
        unresolved_count = 0

    remaining_total = None
    remaining_retry = None
    if current_policy is not None and current_policy.max_total_micros is not None:
        remaining_total = max(current_policy.max_total_micros - effective, 0)
    if current_policy is not None and current_policy.max_retry_micros is not None:
        remaining_retry = max(current_policy.max_retry_micros - retry_exposure, 0)

    last_block_reason = None
    checkpoint_status = None
    if checkpoint is not None:
        checkpoint_status = checkpoint.get("status")
        if checkpoint.get("failure_stage") == "budget":
            last_block_reason = checkpoint.get("failure_code")

    status = GenerationSpendRunStatus(
        generation_run_id=generation_run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_status=checkpoint_status,
        last_block_reason=last_block_reason,
        currency=currency,
        reservation_count=len(reservations),
        submission_count=len(reservations),
        retry_submission_count=sum(1 for item in reservations if item.retry_submission),
        estimated_reserved_micros=estimated,
        actual_finalized_micros=actual,
        effective_exposure_micros=effective,
        retry_exposure_micros=retry_exposure,
        remaining_total_micros=remaining_total,
        remaining_retry_micros=remaining_retry,
        actual_minus_estimated_micros=actual - estimated,
        variance_complete=(
            pending_count == 0
            and unresolved_count == 0
            and bool(reservations or events)
        ),
        finalized_billing_event_count=finalized_count,
        pending_billing_event_count=pending_count,
        unmatched_finalized_event_count=unmatched_count,
        unresolved_reservation_count=unresolved_count,
        policy=policy_status,
        policy_history=_policy_history(reservations),
        providers=_provider_summaries(reservations, events, generation_run_id),
        circuit_breaker=_circuit_breaker(
            policy_status,
            effective_exposure_micros=effective,
            retry_exposure_micros=retry_exposure,
        ),
    )
    status.refresh_digest()
    return status


def build_generation_spend_session_status(
    context: Dict[str, Any],
    *,
    session_id: str,
) -> GenerationSpendSessionStatus:
    runs = [
        build_generation_spend_run_status(context, generation_run_id)
        for generation_run_id in _run_ids(context)
    ]
    status = GenerationSpendSessionStatus(
        session_id=session_id,
        run_count=len(runs),
        runs=runs,
    )
    status.refresh_digest()
    return status


class GenerationSpendStatusService:
    def __init__(self, db):
        self.db = db

    def session_status(self, session_id: str) -> GenerationSpendSessionStatus:
        context = self.db.get_context_messages(session_id) or {}
        return build_generation_spend_session_status(context, session_id=session_id)

    def run_status(
        self,
        session_id: str,
        generation_run_id: str,
    ) -> Optional[GenerationSpendRunStatus]:
        context = self.db.get_context_messages(session_id) or {}
        if generation_run_id not in _run_ids(context):
            return None
        return build_generation_spend_run_status(context, generation_run_id)
