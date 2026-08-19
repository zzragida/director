from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_provenance import (
    GenerationProvenanceManifest,
    stable_digest,
)


COST_ATTRIBUTION_VERSION = 1
RATE_CARD_VERSION = 1


class PricingBasis(str, Enum):
    per_submission = "per_submission"
    per_planned_second_submission = "per_planned_second_submission"
    per_persisted_second = "per_persisted_second"


class GenerationCostRate(BaseModel):
    """Explicit caller-supplied price for one provider/media/model combination.

    Director deliberately ships no provider prices. Monetary attribution only
    occurs when an explicit rate card is supplied by the operator.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    provider: str
    media_type: str
    basis: PricingBasis
    unit_price_micros: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    model: Optional[str] = None
    label: Optional[str] = None


class GenerationCostRateCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = RATE_CARD_VERSION
    rate_card_id: str
    rate_card_version: str
    rates: List[GenerationCostRate] = Field(default_factory=list)
    rate_card_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"rate_card_digest"})
        self.rate_card_digest = stable_digest(payload)

    def find_rate(
        self,
        *,
        provider: Optional[str],
        media_type: str,
        model: Optional[str],
    ) -> Optional[GenerationCostRate]:
        if not provider:
            return None
        exact = [
            rate
            for rate in self.rates
            if rate.provider == provider
            and rate.media_type == media_type
            and rate.model is not None
            and model is not None
            and rate.model == model
        ]
        if len(exact) > 1:
            raise ValueError("ambiguous exact cost rate")
        if exact:
            return exact[0]

        fallback = [
            rate
            for rate in self.rates
            if rate.provider == provider
            and rate.media_type == media_type
            and rate.model is None
        ]
        if len(fallback) > 1:
            raise ValueError("ambiguous provider cost rate")
        return fallback[0] if fallback else None


class OperationCostAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: Optional[str] = None
    media_type: str
    provider: Optional[str] = None
    model: Optional[str] = None
    accounting_complete: bool = False
    submission_count: Optional[int] = Field(default=None, ge=0)
    resume_count: Optional[int] = Field(default=None, ge=0)
    retry_submission_count: Optional[int] = Field(default=None, ge=0)
    planned_duration_seconds: Optional[float] = Field(default=None, ge=0)
    persisted_duration_seconds: Optional[float] = Field(default=None, ge=0)
    pricing_status: str = "unpriced"
    basis: Optional[str] = None
    quantity: Optional[float] = Field(default=None, ge=0)
    unit_price_micros: Optional[int] = Field(default=None, ge=0)
    amount_micros: Optional[int] = Field(default=None, ge=0)
    retry_cost_micros: Optional[int] = Field(default=None, ge=0)
    currency: Optional[str] = None
    estimated: bool = False


class GenerationCostAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = COST_ATTRIBUTION_VERSION
    pricing_status: str = "unpriced"
    rate_card_id: Optional[str] = None
    rate_card_version: Optional[str] = None
    rate_card_digest: Optional[str] = None
    lines: List[OperationCostAttribution] = Field(default_factory=list)
    totals_micros_by_currency: Dict[str, int] = Field(default_factory=dict)
    retry_cost_micros_by_currency: Dict[str, int] = Field(default_factory=dict)
    known_submission_count: int = Field(default=0, ge=0)
    known_resume_count: int = Field(default=0, ge=0)
    known_retry_submission_count: int = Field(default=0, ge=0)
    unknown_accounting_operations: int = Field(default=0, ge=0)
    estimated_priced_lines: int = Field(default=0, ge=0)
    attribution_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"attribution_digest"})
        self.attribution_digest = stable_digest(payload)


def extract_model_hint(config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    for key in ("model", "model_name", "model_id", "model_version", "version"):
        value = config.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return None


def _money(quantity: float, unit_price_micros: int) -> int:
    amount = Decimal(str(quantity)) * Decimal(unit_price_micros)
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _price_line(
    line: OperationCostAttribution,
    rate_card: Optional[GenerationCostRateCard],
) -> OperationCostAttribution:
    if not line.accounting_complete:
        line.pricing_status = "legacy_accounting_unknown"
        return line
    if rate_card is None:
        line.pricing_status = "unpriced"
        return line

    rate = rate_card.find_rate(
        provider=line.provider,
        media_type=line.media_type,
        model=line.model,
    )
    if rate is None:
        line.pricing_status = "unpriced"
        return line

    basis = str(rate.basis)
    quantity: Optional[float] = None
    estimated = False
    retry_quantity: Optional[float] = None

    if basis == PricingBasis.per_submission:
        quantity = float(line.submission_count or 0)
        retry_quantity = float(line.retry_submission_count or 0)
    elif basis == PricingBasis.per_planned_second_submission:
        if line.planned_duration_seconds is None:
            line.pricing_status = "insufficient_usage"
            line.basis = basis
            return line
        quantity = float(line.planned_duration_seconds) * float(line.submission_count or 0)
        retry_quantity = float(line.planned_duration_seconds) * float(
            line.retry_submission_count or 0
        )
        estimated = True
    elif basis == PricingBasis.per_persisted_second:
        if line.persisted_duration_seconds is None:
            line.pricing_status = "insufficient_usage"
            line.basis = basis
            return line
        quantity = float(line.persisted_duration_seconds)
        # Persisted duration does not reveal how much failed/retried work the
        # provider actually billed, so retry monetary attribution stays unknown.
        retry_quantity = None
    else:
        line.pricing_status = "unpriced"
        return line

    line.pricing_status = "priced"
    line.basis = basis
    line.quantity = quantity
    line.unit_price_micros = rate.unit_price_micros
    line.amount_micros = _money(quantity, rate.unit_price_micros)
    line.retry_cost_micros = (
        _money(retry_quantity, rate.unit_price_micros)
        if retry_quantity is not None
        else None
    )
    line.currency = rate.currency.upper()
    line.estimated = estimated
    return line


def _scene_line(scene: Any) -> OperationCostAttribution:
    submission_count = getattr(scene, "submission_count", None)
    resume_count = getattr(scene, "resume_count", None)
    accounting_complete = bool(getattr(scene, "accounting_complete", False))
    retry_count = (
        max(int(submission_count) - 1, 0)
        if accounting_complete and submission_count is not None
        else None
    )
    return OperationCostAttribution(
        operation_id=getattr(scene, "operation_id", None),
        media_type="video",
        provider=getattr(scene, "provider", None),
        model=extract_model_hint(getattr(scene, "provider_config", {}) or {}),
        accounting_complete=accounting_complete,
        submission_count=submission_count,
        resume_count=resume_count,
        retry_submission_count=retry_count,
        planned_duration_seconds=getattr(scene, "planned_duration_seconds", None),
        persisted_duration_seconds=getattr(scene, "artifact_length", None),
    )


def _audio_line(audio: Any) -> OperationCostAttribution:
    submission_count = getattr(audio, "submission_count", None)
    resume_count = getattr(audio, "resume_count", None)
    accounting_complete = bool(getattr(audio, "accounting_complete", False))
    retry_count = (
        max(int(submission_count) - 1, 0)
        if accounting_complete and submission_count is not None
        else None
    )
    return OperationCostAttribution(
        operation_id=getattr(audio, "operation_id", None),
        media_type="audio",
        provider=getattr(audio, "provider", None),
        model=extract_model_hint(getattr(audio, "provider_config", {}) or {}),
        accounting_complete=accounting_complete,
        submission_count=submission_count,
        resume_count=resume_count,
        retry_submission_count=retry_count,
        planned_duration_seconds=None,
        persisted_duration_seconds=getattr(audio, "artifact_length", None),
    )


def build_generation_cost_attribution(
    manifest: GenerationProvenanceManifest,
    *,
    rate_card: Optional[GenerationCostRateCard] = None,
) -> GenerationCostAttribution:
    if rate_card is not None:
        rate_card.refresh_digest()

    lines = [_price_line(_scene_line(scene), rate_card) for scene in manifest.scenes]
    if manifest.audio is not None:
        lines.append(_price_line(_audio_line(manifest.audio), rate_card))

    totals: Dict[str, int] = {}
    retry_totals: Dict[str, int] = {}
    for line in lines:
        if line.amount_micros is not None and line.currency:
            totals[line.currency] = totals.get(line.currency, 0) + line.amount_micros
        if line.retry_cost_micros is not None and line.currency:
            retry_totals[line.currency] = (
                retry_totals.get(line.currency, 0) + line.retry_cost_micros
            )

    priced = sum(1 for line in lines if line.pricing_status == "priced")
    if lines and priced == len(lines):
        pricing_status = "priced"
    elif priced:
        pricing_status = "partial"
    else:
        pricing_status = "unpriced"

    attribution = GenerationCostAttribution(
        pricing_status=pricing_status,
        rate_card_id=rate_card.rate_card_id if rate_card else None,
        rate_card_version=rate_card.rate_card_version if rate_card else None,
        rate_card_digest=rate_card.rate_card_digest if rate_card else None,
        lines=lines,
        totals_micros_by_currency=totals,
        retry_cost_micros_by_currency=retry_totals,
        known_submission_count=sum(
            int(line.submission_count or 0) for line in lines if line.accounting_complete
        ),
        known_resume_count=sum(
            int(line.resume_count or 0) for line in lines if line.accounting_complete
        ),
        known_retry_submission_count=sum(
            int(line.retry_submission_count or 0)
            for line in lines
            if line.accounting_complete
        ),
        unknown_accounting_operations=sum(
            1 for line in lines if not line.accounting_complete
        ),
        estimated_priced_lines=sum(
            1 for line in lines if line.pricing_status == "priced" and line.estimated
        ),
    )
    attribution.refresh_digest()
    return attribution


def safe_cost_summary(attribution: GenerationCostAttribution) -> Dict[str, Any]:
    return {
        "pricing_status": attribution.pricing_status,
        "rate_card_id": attribution.rate_card_id,
        "rate_card_version": attribution.rate_card_version,
        "totals_micros_by_currency": dict(attribution.totals_micros_by_currency),
        "retry_cost_micros_by_currency": dict(attribution.retry_cost_micros_by_currency),
        "known_submission_count": attribution.known_submission_count,
        "known_resume_count": attribution.known_resume_count,
        "known_retry_submission_count": attribution.known_retry_submission_count,
        "unknown_accounting_operations": attribution.unknown_accounting_operations,
        "estimated_priced_lines": attribution.estimated_priced_lines,
        "attribution_digest": attribution.attribution_digest,
    }
