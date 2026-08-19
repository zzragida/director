import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_provenance import stable_digest


POLICY_AUDIT_CONTEXT_KEY = "__generation_budget_policy_audit__"
POLICY_AUDIT_VERSION = 1


class BudgetPolicyAuditEntry(BaseModel):
    """Safe snapshot of the policy/rate-card pair that authorized submissions.

    The audit entry stores limits and digests, never raw rate lines, provider
    credentials, prompts, or invoice payloads.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = POLICY_AUDIT_VERSION
    audit_id: str
    generation_run_id: str
    policy_id: str
    policy_version: str
    policy_digest: Optional[str] = None
    currency: str = Field(min_length=3, max_length=3)
    max_total_micros: Optional[int] = Field(default=None, ge=0)
    max_retry_micros: Optional[int] = Field(default=None, ge=0)
    max_submissions_per_operation: Optional[int] = Field(default=None, ge=1)
    max_retry_submissions_per_operation: Optional[int] = Field(default=None, ge=0)
    require_preflight_pricing: bool
    allow_legacy_accounting: bool
    rate_card_id: Optional[str] = None
    rate_card_version: Optional[str] = None
    rate_card_digest: Optional[str] = None
    applied_at_epoch: int = Field(ge=0)


def _read_document(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    messages = context.get(POLICY_AUDIT_CONTEXT_KEY, [])
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
    context[POLICY_AUDIT_CONTEXT_KEY] = [{"role": "system", "content": content}]


def budget_policy_audit_from_context(context: Dict[str, Any]) -> List[BudgetPolicyAuditEntry]:
    entries: List[BudgetPolicyAuditEntry] = []
    for raw in _read_document(context).values():
        try:
            entries.append(BudgetPolicyAuditEntry.model_validate(raw))
        except Exception:
            continue
    return sorted(entries, key=lambda entry: (entry.applied_at_epoch, entry.audit_id))


def record_applied_policy_in_context(
    context: Dict[str, Any],
    *,
    generation_run_id: str,
    policy: Any,
    rate_card: Optional[Any],
    applied_at_epoch: int,
) -> BudgetPolicyAuditEntry:
    payload = {
        "generation_run_id": generation_run_id,
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "policy_digest": policy.policy_digest,
        "rate_card_id": rate_card.rate_card_id if rate_card else None,
        "rate_card_version": rate_card.rate_card_version if rate_card else None,
        "rate_card_digest": rate_card.rate_card_digest if rate_card else None,
    }
    audit_id = f"genpolicy:{stable_digest(payload)[:24]}"
    entry = BudgetPolicyAuditEntry(
        audit_id=audit_id,
        generation_run_id=generation_run_id,
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
        applied_at_epoch=int(applied_at_epoch),
    )

    document = _read_document(context)
    existing = document.get(audit_id)
    serialized = entry.model_dump(mode="json")
    if existing is None:
        document[audit_id] = serialized
        _write_document(context, document)
    elif existing != serialized:
        # Same policy/rate-card identity can be applied repeatedly. Preserve the
        # first authoritative application timestamp rather than rewriting history.
        return BudgetPolicyAuditEntry.model_validate(existing)
    return entry


def safe_policy_audit_summary(entry: BudgetPolicyAuditEntry) -> Dict[str, Any]:
    return {
        "audit_id": entry.audit_id,
        "generation_run_id": entry.generation_run_id,
        "policy_id": entry.policy_id,
        "policy_version": entry.policy_version,
        "policy_digest": entry.policy_digest,
        "currency": entry.currency,
        "max_total_micros": entry.max_total_micros,
        "max_retry_micros": entry.max_retry_micros,
        "max_submissions_per_operation": entry.max_submissions_per_operation,
        "max_retry_submissions_per_operation": entry.max_retry_submissions_per_operation,
        "require_preflight_pricing": entry.require_preflight_pricing,
        "allow_legacy_accounting": entry.allow_legacy_accounting,
        "rate_card_id": entry.rate_card_id,
        "rate_card_version": entry.rate_card_version,
        "rate_card_digest": entry.rate_card_digest,
        "applied_at_epoch": entry.applied_at_epoch,
    }
