import json
import os
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


RATE_CARD_VERSION = 1
ACCOUNTING_VERSION = 1
RATE_CARD_ENV = "GENERATION_RATE_CARD_JSON"


class CostStatus(str, Enum):
    unknown = "unknown"
    estimated = "estimated"
    actual = "actual"


class UsageMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str
    quantity: float = Field(ge=0)


class RateCardLine(BaseModel):
    """One explicit pricing rule supplied by an operator, not by Director defaults."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    provider: str
    operation_kind: str
    unit: str
    price_per_unit: Decimal = Field(ge=0)
    model: Optional[str] = None
    model_version: Optional[str] = None

    @field_validator("provider", "operation_kind", "unit")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RateCardSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = RATE_CARD_VERSION
    rate_card_id: str
    currency: str = "USD"
    effective_at: Optional[str] = None
    source_reference: Optional[str] = None
    lines: List[RateCardLine] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a 3-letter code")
        return value


class OperationCostAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    operation_id: str
    operation_kind: str
    provider: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    attempt_count: int = Field(default=0, ge=0)
    usage: List[UsageMetric] = Field(default_factory=list)
    cost_status: CostStatus = CostStatus.unknown
    currency: Optional[str] = None
    amount: Optional[Decimal] = Field(default=None, ge=0)
    pricing_unit: Optional[str] = None
    rate_card_id: Optional[str] = None
    retry_waste_amount: Optional[Decimal] = Field(default=None, ge=0)


class GenerationCostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = ACCOUNTING_VERSION
    rate_card: Optional[RateCardSnapshot] = None
    operations: List[OperationCostAttribution] = Field(default_factory=list)
    currency: Optional[str] = None
    total_amount: Optional[Decimal] = Field(default=None, ge=0)
    retry_waste_amount: Optional[Decimal] = Field(default=None, ge=0)
    estimated_operation_count: int = Field(default=0, ge=0)
    actual_operation_count: int = Field(default=0, ge=0)
    unknown_operation_count: int = Field(default=0, ge=0)


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid decimal value") from exc


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def load_rate_card_from_environment() -> Optional[RateCardSnapshot]:
    """Load an operator-supplied versioned rate card. Director ships no prices."""

    raw = os.getenv(RATE_CARD_ENV)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return RateCardSnapshot.model_validate(payload)
    except Exception:
        return None


def infer_model_metadata(provider_config: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    config = provider_config or {}
    model = config.get("model") or config.get("model_id")
    version = config.get("model_version") or config.get("version")
    return (
        str(model) if model not in (None, "") else None,
        str(version) if version not in (None, "") else None,
    )


def _usage_for_lineage(entry: Any, *, operation_kind: str) -> List[UsageMetric]:
    metrics = [UsageMetric(unit="operation", quantity=1.0)]
    attempts = int(getattr(entry, "attempt_count", 0) or 0)
    if attempts:
        metrics.append(UsageMetric(unit="attempt", quantity=float(attempts)))
    length = getattr(entry, "artifact_length", None)
    if length is not None:
        try:
            quantity = float(length)
            if quantity >= 0:
                metrics.append(UsageMetric(unit="output_second", quantity=quantity))
        except (TypeError, ValueError):
            pass
    return metrics


def _match_rate(
    rate_card: Optional[RateCardSnapshot],
    *,
    provider: Optional[str],
    operation_kind: str,
    model: Optional[str],
    model_version: Optional[str],
    usage: List[UsageMetric],
) -> Optional[RateCardLine]:
    if rate_card is None or not provider:
        return None
    usage_units = {item.unit for item in usage}
    candidates = []
    for line in rate_card.lines:
        if line.provider != provider or line.operation_kind != operation_kind:
            continue
        if line.unit not in usage_units:
            continue
        if line.model is not None and line.model != model:
            continue
        if line.model_version is not None and line.model_version != model_version:
            continue
        specificity = int(line.model is not None) + int(line.model_version is not None)
        candidates.append((specificity, line))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _quantity_for_unit(usage: List[UsageMetric], unit: str) -> Optional[Decimal]:
    for item in usage:
        if item.unit == unit:
            return _decimal(item.quantity)
    return None


def _build_operation_cost(
    entry: Any,
    *,
    operation_kind: str,
    rate_card: Optional[RateCardSnapshot],
) -> Optional[OperationCostAttribution]:
    operation_id = getattr(entry, "operation_id", None)
    if not operation_id:
        return None
    provider = getattr(entry, "provider", None)
    provider_config = getattr(entry, "provider_config", None) or {}
    model, model_version = infer_model_metadata(provider_config)
    usage = _usage_for_lineage(entry, operation_kind=operation_kind)
    attempts = int(getattr(entry, "attempt_count", 0) or 0)
    attribution = OperationCostAttribution(
        operation_id=operation_id,
        operation_kind=operation_kind,
        provider=provider,
        model=model,
        model_version=model_version,
        attempt_count=attempts,
        usage=usage,
    )
    line = _match_rate(
        rate_card,
        provider=provider,
        operation_kind=operation_kind,
        model=model,
        model_version=model_version,
        usage=usage,
    )
    if line is None or rate_card is None:
        return attribution

    quantity = _quantity_for_unit(usage, line.unit)
    if quantity is None:
        return attribution
    amount = _money(quantity * line.price_per_unit)
    attribution.cost_status = CostStatus.estimated
    attribution.currency = rate_card.currency
    attribution.amount = amount
    attribution.pricing_unit = line.unit
    attribution.rate_card_id = rate_card.rate_card_id
    if line.unit == "attempt" and attempts > 1:
        attribution.retry_waste_amount = _money(
            Decimal(attempts - 1) * line.price_per_unit
        )
    return attribution


def build_cost_summary(
    manifest: Any,
    *,
    rate_card: Optional[RateCardSnapshot] = None,
) -> GenerationCostSummary:
    """Build cost attribution without inventing provider prices or usage."""

    effective_rate_card = rate_card
    prior = getattr(manifest, "accounting", None)
    if effective_rate_card is None and prior is not None:
        effective_rate_card = getattr(prior, "rate_card", None)

    operations: List[OperationCostAttribution] = []
    for scene in getattr(manifest, "scenes", []) or []:
        cost = _build_operation_cost(
            scene,
            operation_kind="scene_video",
            rate_card=effective_rate_card,
        )
        if cost is not None:
            operations.append(cost)

    audio = getattr(manifest, "audio", None)
    if audio is not None:
        cost = _build_operation_cost(
            audio,
            operation_kind="background_audio",
            rate_card=effective_rate_card,
        )
        if cost is not None:
            operations.append(cost)

    final = getattr(manifest, "final", None)
    if (
        final is not None
        and getattr(final, "operation_id", None)
        and getattr(final, "stream_url", None)
    ):
        final_entry = type("FinalCostEntry", (), {
            "operation_id": final.operation_id,
            "provider": "director",
            "provider_config": {},
            "attempt_count": 1,
            "artifact_length": None,
        })()
        cost = _build_operation_cost(
            final_entry,
            operation_kind="combine",
            rate_card=effective_rate_card,
        )
        if cost is not None:
            operations.append(cost)

    known = [item for item in operations if item.amount is not None and item.currency]
    currencies = {item.currency for item in known if item.currency}
    currency = next(iter(currencies)) if len(currencies) == 1 else None
    total = None
    retry_waste = None
    if currency is not None:
        total = _money(sum((item.amount or Decimal("0")) for item in known))
        wastes = [item.retry_waste_amount for item in known if item.retry_waste_amount is not None]
        if wastes:
            retry_waste = _money(sum(wastes, Decimal("0")))

    return GenerationCostSummary(
        rate_card=effective_rate_card,
        operations=operations,
        currency=currency,
        total_amount=total,
        retry_waste_amount=retry_waste,
        estimated_operation_count=sum(
            1 for item in operations if item.cost_status == CostStatus.estimated
        ),
        actual_operation_count=sum(
            1 for item in operations if item.cost_status == CostStatus.actual
        ),
        unknown_operation_count=sum(
            1 for item in operations if item.cost_status == CostStatus.unknown
        ),
    )
