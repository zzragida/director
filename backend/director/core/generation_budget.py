import json
import os
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_billing import (
    billing_events_from_context,
    reconcile_generation_billing,
)
from director.core.generation_cost import (
    GenerationCostRate,
    GenerationCostRateCard,
    PricingBasis,
    extract_model_hint,
)
from director.core.generation_provenance import stable_digest


BUDGET_CONTEXT_KEY = "__generation_budget_reservations__"
BUDGET_POLICY_ENV = "DIRECTOR_GENERATION_BUDGET_POLICY_JSON"
BUDGET_RATE_CARD_ENV = "DIRECTOR_GENERATION_RATE_CARD_JSON"
BUDGET_POLICY_VERSION = 1
BUDGET_RESERVATION_VERSION = 1


class GenerationBudgetPolicy(BaseModel):
    """Operator-supplied guardrails applied before new provider submissions."""

    model_config = ConfigDict(extra="forbid")

    version: int = BUDGET_POLICY_VERSION
    policy_id: str
    policy_version: str = "1"
    currency: str = Field(min_length=3, max_length=3)
    max_total_micros: Optional[int] = Field(default=None, ge=0)
    max_retry_micros: Optional[int] = Field(default=None, ge=0)
    max_submissions_per_operation: Optional[int] = Field(default=None, ge=1)
    max_retry_submissions_per_operation: Optional[int] = Field(default=None, ge=0)
    require_preflight_pricing: bool = True
    allow_legacy_accounting: bool = False
    policy_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"policy_digest"})
        self.policy_digest = stable_digest(payload)

    @property
    def has_monetary_limit(self) -> bool:
        return self.max_total_micros is not None or self.max_retry_micros is not None


class BudgetReservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = BUDGET_RESERVATION_VERSION
    reservation_id: str
    generation_run_id: str
    operation_id: str
    submission_ordinal: int = Field(ge=1)
    retry_submission: bool = False
    media_type: str
    provider: str
    model: Optional[str] = None
    currency: str = Field(min_length=3, max_length=3)
    pricing_basis: Optional[str] = None
    estimated_amount_micros: Optional[int] = Field(default=None, ge=0)
    estimated: bool = False
    policy_id: str
    policy_version: str
    policy_digest: Optional[str] = None
    rate_card_id: Optional[str] = None
    rate_card_version: Optional[str] = None
    rate_card_digest: Optional[str] = None
    created_at_epoch: int


class BudgetGuardViolation(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        operation_id: Optional[str] = None,
        scene_index: Optional[int] = None,
    ):
        super().__init__(code)
        self.code = code
        self.operation_id = operation_id
        self.scene_index = scene_index


class BudgetConfigurationError(RuntimeError):
    pass


def _read_document(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    messages = context.get(BUDGET_CONTEXT_KEY, [])
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
    context[BUDGET_CONTEXT_KEY] = [{"role": "system", "content": content}]


def budget_reservations_from_context(context: Dict[str, Any]) -> List[BudgetReservation]:
    reservations: List[BudgetReservation] = []
    for raw in _read_document(context).values():
        try:
            reservations.append(BudgetReservation.model_validate(raw))
        except Exception:
            continue
    return reservations


def load_budget_configuration_from_env() -> Tuple[
    Optional[GenerationBudgetPolicy], Optional[GenerationCostRateCard]
]:
    raw_policy = os.getenv(BUDGET_POLICY_ENV)
    if not raw_policy:
        return None, None
    try:
        policy = GenerationBudgetPolicy.model_validate(json.loads(raw_policy))
    except Exception as exc:
        raise BudgetConfigurationError("invalid_budget_policy") from exc
    policy.currency = policy.currency.upper()
    policy.refresh_digest()

    raw_rate_card = os.getenv(BUDGET_RATE_CARD_ENV)
    if not raw_rate_card:
        if policy.has_monetary_limit and policy.require_preflight_pricing:
            raise BudgetConfigurationError("budget_rate_card_required")
        return policy, None
    try:
        rate_card = GenerationCostRateCard.model_validate(json.loads(raw_rate_card))
    except Exception as exc:
        raise BudgetConfigurationError("invalid_budget_rate_card") from exc
    for rate in rate_card.rates:
        rate.currency = rate.currency.upper()
    rate_card.refresh_digest()
    return policy, rate_card


def make_budget_reservation_id(
    generation_run_id: str,
    operation_id: str,
    submission_ordinal: int,
) -> str:
    digest = stable_digest(
        {
            "generation_run_id": generation_run_id,
            "operation_id": operation_id,
            "submission_ordinal": submission_ordinal,
        }
    )[:24]
    return f"genbudget:{digest}"


def _money(quantity: float, unit_price_micros: int) -> int:
    amount = Decimal(str(quantity)) * Decimal(unit_price_micros)
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _estimate_submission_amount(
    *,
    rate: Optional[GenerationCostRate],
    planned_duration_seconds: Optional[float],
) -> Tuple[Optional[int], Optional[str], bool]:
    if rate is None:
        return None, None, False
    basis = str(rate.basis)
    if basis == PricingBasis.per_submission:
        return rate.unit_price_micros, basis, False
    if basis == PricingBasis.per_planned_second_submission:
        if planned_duration_seconds is None:
            return None, basis, True
        return (
            _money(planned_duration_seconds, rate.unit_price_micros),
            basis,
            True,
        )
    if basis == PricingBasis.per_persisted_second:
        return None, basis, False
    return None, basis, False


def _old_scene_operation(raw_checkpoint: Optional[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_checkpoint, dict):
        return None
    for scene in raw_checkpoint.get("scenes", []) or []:
        if isinstance(scene, dict) and scene.get("index") == index:
            operation = scene.get("operation")
            return operation if isinstance(operation, dict) else None
    return None


def _old_audio_operation(raw_checkpoint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_checkpoint, dict):
        return None
    operation = raw_checkpoint.get("audio_operation")
    return operation if isinstance(operation, dict) else None


def _new_submission_ordinals(
    *,
    old_operation: Optional[Dict[str, Any]],
    new_operation: Any,
    policy: GenerationBudgetPolicy,
    scene_index: Optional[int],
) -> List[int]:
    new_count = getattr(new_operation, "submission_count", None)
    if new_count is None or int(new_count) <= 0:
        return []

    old_count_raw = old_operation.get("submission_count") if old_operation else 0
    old_count = int(old_count_raw) if old_count_raw is not None else 0
    if int(new_count) <= old_count:
        return []

    accounting_complete = bool(getattr(new_operation, "accounting_complete", False))
    if not accounting_complete and not policy.allow_legacy_accounting:
        raise BudgetGuardViolation(
            "budget_legacy_accounting_unknown",
            operation_id=getattr(new_operation, "operation_id", None),
            scene_index=scene_index,
        )
    return list(range(old_count + 1, int(new_count) + 1))


def _operation_limit_check(
    policy: GenerationBudgetPolicy,
    *,
    operation: Any,
    submission_ordinal: int,
    scene_index: Optional[int],
) -> None:
    if (
        policy.max_submissions_per_operation is not None
        and submission_ordinal > policy.max_submissions_per_operation
    ):
        raise BudgetGuardViolation(
            "budget_submission_limit_exceeded",
            operation_id=getattr(operation, "operation_id", None),
            scene_index=scene_index,
        )
    retry_ordinal = max(submission_ordinal - 1, 0)
    if (
        policy.max_retry_submissions_per_operation is not None
        and retry_ordinal > policy.max_retry_submissions_per_operation
    ):
        raise BudgetGuardViolation(
            "budget_retry_limit_exceeded",
            operation_id=getattr(operation, "operation_id", None),
            scene_index=scene_index,
        )


def _retry_exposure(
    *,
    reservations: List[BudgetReservation],
    events: List[Any],
    generation_run_id: str,
    currency: str,
) -> int:
    reservation_by_id = {
        reservation.reservation_id: reservation
        for reservation in reservations
        if reservation.generation_run_id == generation_run_id
        and reservation.currency.upper() == currency.upper()
        and reservation.retry_submission
    }
    actual_by_reservation: Dict[str, int] = {}
    for event in events:
        if (
            event.generation_run_id == generation_run_id
            and event.currency.upper() == currency.upper()
            and str(event.status) == "finalized"
            and event.reservation_id in reservation_by_id
        ):
            actual_by_reservation[event.reservation_id] = (
                actual_by_reservation.get(event.reservation_id, 0)
                + event.signed_amount_micros
            )
    exposure = 0
    for reservation_id, reservation in reservation_by_id.items():
        if reservation_id in actual_by_reservation:
            exposure += actual_by_reservation[reservation_id]
        else:
            exposure += int(reservation.estimated_amount_micros or 0)
    return max(exposure, 0)


def _provider_config_for_operation(checkpoint: Any, media_type: str) -> Dict[str, Any]:
    provenance = getattr(checkpoint, "provenance", None)
    request = getattr(provenance, "request", None)
    if request is None:
        return {}
    config = (
        getattr(request, "video_config", {})
        if media_type == "video"
        else getattr(request, "audio_config", {})
    )
    return config if isinstance(config, dict) else {}


def _planned_duration_for_scene(scene: Any) -> Optional[float]:
    plan = getattr(scene, "plan", None)
    if not isinstance(plan, dict):
        return None
    value = plan.get("suggested_duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if float(value) >= 0 else None


def authorize_checkpoint_budget_transition(
    *,
    context: Dict[str, Any],
    previous_checkpoint: Optional[Dict[str, Any]],
    checkpoint: Any,
    policy: GenerationBudgetPolicy,
    rate_card: Optional[GenerationCostRateCard],
    now_epoch: int,
) -> List[BudgetReservation]:
    """Reserve budget for every newly-recorded provider submission.

    The caller invokes this inside the same context CAS mutation that persists
    the checkpoint. If this raises, neither the checkpoint submission state nor
    the reservation document is committed.
    """

    if not getattr(checkpoint, "generation_run_id", None):
        raise BudgetGuardViolation("budget_generation_run_missing")

    generation_run_id = checkpoint.generation_run_id
    document = _read_document(context)
    reservations = budget_reservations_from_context(context)
    events = billing_events_from_context(context)
    new_reservations: List[BudgetReservation] = []

    reconciliation = reconcile_generation_billing(
        generation_run_id=generation_run_id,
        currency=policy.currency,
        reservations=reservations,
        events=events,
    )
    total_exposure = reconciliation.effective_exposure_micros
    retry_exposure = _retry_exposure(
        reservations=reservations,
        events=events,
        generation_run_id=generation_run_id,
        currency=policy.currency,
    )

    candidates: List[Tuple[Any, str, Optional[int], Optional[float], Optional[Dict[str, Any]]]] = []
    for scene in getattr(checkpoint, "scenes", []) or []:
        operation = getattr(scene, "operation", None)
        if operation is None:
            continue
        candidates.append(
            (
                operation,
                "video",
                getattr(scene, "index", None),
                _planned_duration_for_scene(scene),
                _old_scene_operation(previous_checkpoint, getattr(scene, "index", 0)),
            )
        )
    audio_operation = getattr(checkpoint, "audio_operation", None)
    if audio_operation is not None:
        candidates.append(
            (
                audio_operation,
                "audio",
                None,
                None,
                _old_audio_operation(previous_checkpoint),
            )
        )

    for operation, media_type, scene_index, planned_duration, old_operation in candidates:
        ordinals = _new_submission_ordinals(
            old_operation=old_operation,
            new_operation=operation,
            policy=policy,
            scene_index=scene_index,
        )
        if not ordinals:
            continue

        config = _provider_config_for_operation(checkpoint, media_type)
        model = extract_model_hint(config)
        rate = (
            rate_card.find_rate(
                provider=getattr(operation, "provider", None),
                media_type=media_type,
                model=model,
            )
            if rate_card is not None
            else None
        )

        for ordinal in ordinals:
            _operation_limit_check(
                policy,
                operation=operation,
                submission_ordinal=ordinal,
                scene_index=scene_index,
            )
            reservation_id = make_budget_reservation_id(
                generation_run_id,
                operation.operation_id,
                ordinal,
            )
            if reservation_id in document:
                continue

            amount, basis, estimated = _estimate_submission_amount(
                rate=rate,
                planned_duration_seconds=planned_duration,
            )
            if rate is None and policy.has_monetary_limit and policy.require_preflight_pricing:
                raise BudgetGuardViolation(
                    "budget_unpriced_operation",
                    operation_id=operation.operation_id,
                    scene_index=scene_index,
                )
            if (
                rate is not None
                and rate.currency.upper() != policy.currency.upper()
                and policy.has_monetary_limit
            ):
                raise BudgetGuardViolation(
                    "budget_currency_mismatch",
                    operation_id=operation.operation_id,
                    scene_index=scene_index,
                )
            if (
                rate is not None
                and amount is None
                and policy.has_monetary_limit
                and policy.require_preflight_pricing
            ):
                raise BudgetGuardViolation(
                    "budget_unpriceable_operation",
                    operation_id=operation.operation_id,
                    scene_index=scene_index,
                )

            reservation = BudgetReservation(
                reservation_id=reservation_id,
                generation_run_id=generation_run_id,
                operation_id=operation.operation_id,
                submission_ordinal=ordinal,
                retry_submission=ordinal > 1,
                media_type=media_type,
                provider=operation.provider,
                model=model,
                currency=policy.currency.upper(),
                pricing_basis=basis,
                estimated_amount_micros=amount,
                estimated=estimated,
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
                policy_digest=policy.policy_digest,
                rate_card_id=rate_card.rate_card_id if rate_card else None,
                rate_card_version=rate_card.rate_card_version if rate_card else None,
                rate_card_digest=rate_card.rate_card_digest if rate_card else None,
                created_at_epoch=int(now_epoch),
            )
            next_total = total_exposure + int(amount or 0)
            next_retry = retry_exposure + (
                int(amount or 0) if reservation.retry_submission else 0
            )
            if policy.max_total_micros is not None and next_total > policy.max_total_micros:
                raise BudgetGuardViolation(
                    "budget_total_exceeded",
                    operation_id=operation.operation_id,
                    scene_index=scene_index,
                )
            if policy.max_retry_micros is not None and next_retry > policy.max_retry_micros:
                raise BudgetGuardViolation(
                    "budget_retry_exceeded",
                    operation_id=operation.operation_id,
                    scene_index=scene_index,
                )

            document[reservation_id] = reservation.model_dump(mode="json")
            reservations.append(reservation)
            new_reservations.append(reservation)
            total_exposure = next_total
            retry_exposure = next_retry

    if new_reservations:
        _write_document(context, document)
    return new_reservations


def safe_budget_reservation_summary(reservation: BudgetReservation) -> Dict[str, Any]:
    return {
        "reservation_id": reservation.reservation_id,
        "generation_run_id": reservation.generation_run_id,
        "operation_id": reservation.operation_id,
        "submission_ordinal": reservation.submission_ordinal,
        "retry_submission": reservation.retry_submission,
        "media_type": reservation.media_type,
        "provider": reservation.provider,
        "currency": reservation.currency,
        "estimated_amount_micros": reservation.estimated_amount_micros,
        "estimated": reservation.estimated,
        "policy_id": reservation.policy_id,
        "policy_version": reservation.policy_version,
        "rate_card_id": reservation.rate_card_id,
        "rate_card_version": reservation.rate_card_version,
    }
