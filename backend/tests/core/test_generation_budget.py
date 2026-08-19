from types import SimpleNamespace

import pytest

from director.core.generation_billing import ProviderBillingEvent
from director.core.generation_budget import (
    BUDGET_CONTEXT_KEY,
    BudgetGuardViolation,
    GenerationBudgetPolicy,
    authorize_checkpoint_budget_transition,
    budget_reservations_from_context,
)
from director.core.generation_cost import GenerationCostRateCard


def operation(operation_id, provider="kling", submission_count=1, accounting_complete=True):
    return SimpleNamespace(
        operation_id=operation_id,
        provider=provider,
        submission_count=submission_count,
        accounting_complete=accounting_complete,
    )


def checkpoint(*, operations, video_config=None, generation_run_id="genrun:test"):
    scenes = []
    for index, op in enumerate(operations):
        scenes.append(
            SimpleNamespace(
                index=index,
                plan={"suggested_duration": 4},
                operation=op,
            )
        )
    request = SimpleNamespace(video_config=video_config or {}, audio_config={})
    return SimpleNamespace(
        generation_run_id=generation_run_id,
        scenes=scenes,
        audio_operation=None,
        provenance=SimpleNamespace(request=request),
    )


def policy(**updates):
    values = {
        "policy_id": "policy-1",
        "policy_version": "1",
        "currency": "USD",
        "max_total_micros": 1000,
        "max_retry_micros": 1000,
        "max_submissions_per_operation": 3,
        "max_retry_submissions_per_operation": 2,
    }
    values.update(updates)
    result = GenerationBudgetPolicy(**values)
    result.refresh_digest()
    return result


def rate_card(unit_price=400, basis="per_submission"):
    card = GenerationCostRateCard(
        rate_card_id="rates-1",
        rate_card_version="2026-08",
        rates=[
            {
                "provider": "kling",
                "media_type": "video",
                "basis": basis,
                "unit_price_micros": unit_price,
                "currency": "USD",
            }
        ],
    )
    card.refresh_digest()
    return card


def test_new_submission_is_reserved_before_checkpoint_commit():
    context = {}
    current = checkpoint(operations=[operation("op-1")])

    reservations = authorize_checkpoint_budget_transition(
        context=context,
        previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": 0}}]},
        checkpoint=current,
        policy=policy(),
        rate_card=rate_card(),
        now_epoch=1000,
    )

    assert len(reservations) == 1
    assert reservations[0].estimated_amount_micros == 400
    assert reservations[0].submission_ordinal == 1
    assert reservations[0].retry_submission is False
    assert BUDGET_CONTEXT_KEY in context
    assert len(budget_reservations_from_context(context)) == 1


def test_second_submission_is_retry_and_retry_limit_is_enforced():
    current = checkpoint(operations=[operation("op-1", submission_count=2)])
    with pytest.raises(BudgetGuardViolation) as exc:
        authorize_checkpoint_budget_transition(
            context={},
            previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": 1}}]},
            checkpoint=current,
            policy=policy(max_retry_submissions_per_operation=0),
            rate_card=rate_card(),
            now_epoch=1000,
        )
    assert exc.value.code == "budget_retry_limit_exceeded"


def test_monetary_budget_fails_closed_without_rate_card():
    current = checkpoint(operations=[operation("op-1")])
    with pytest.raises(BudgetGuardViolation) as exc:
        authorize_checkpoint_budget_transition(
            context={},
            previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": 0}}]},
            checkpoint=current,
            policy=policy(),
            rate_card=None,
            now_epoch=1000,
        )
    assert exc.value.code == "budget_unpriced_operation"


def test_count_only_policy_allows_unpriced_submission():
    current = checkpoint(operations=[operation("op-1")])
    count_policy = policy(max_total_micros=None, max_retry_micros=None)

    reservations = authorize_checkpoint_budget_transition(
        context={},
        previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": 0}}]},
        checkpoint=current,
        policy=count_policy,
        rate_card=None,
        now_epoch=1000,
    )

    assert reservations[0].estimated_amount_micros is None


def test_persisted_second_pricing_cannot_preflight_strict_monetary_budget():
    current = checkpoint(operations=[operation("op-1")])
    with pytest.raises(BudgetGuardViolation) as exc:
        authorize_checkpoint_budget_transition(
            context={},
            previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": 0}}]},
            checkpoint=current,
            policy=policy(),
            rate_card=rate_card(basis="per_persisted_second"),
            now_epoch=1000,
        )
    assert exc.value.code == "budget_unpriceable_operation"


def test_actual_billing_overage_blocks_next_submission():
    context = {}
    first = checkpoint(operations=[operation("op-1")])
    first_reservations = authorize_checkpoint_budget_transition(
        context=context,
        previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": 0}}]},
        checkpoint=first,
        policy=policy(max_total_micros=1000),
        rate_card=rate_card(400),
        now_epoch=1000,
    )
    reservation = first_reservations[0]

    billing_document = {
        "kling:actual-1": ProviderBillingEvent(
            event_id="actual-1",
            provider="kling",
            generation_run_id="genrun:test",
            operation_id="op-1",
            reservation_id=reservation.reservation_id,
            amount_micros=700,
            currency="USD",
            source="invoice",
        ).model_dump(mode="json")
    }
    import json
    context["__generation_billing_events__"] = [
        {
            "role": "system",
            "content": json.dumps(billing_document, sort_keys=True, separators=(",", ":")),
        }
    ]

    second = checkpoint(operations=[operation("op-1", submission_count=1), operation("op-2")])
    previous = {
        "scenes": [
            {"index": 0, "operation": {"submission_count": 1}},
            {"index": 1, "operation": {"submission_count": 0}},
        ]
    }
    with pytest.raises(BudgetGuardViolation) as exc:
        authorize_checkpoint_budget_transition(
            context=context,
            previous_checkpoint=previous,
            checkpoint=second,
            policy=policy(max_total_micros=1000),
            rate_card=rate_card(400),
            now_epoch=1001,
        )
    assert exc.value.code == "budget_total_exceeded"


def test_legacy_accounting_is_blocked_in_strict_policy():
    current = checkpoint(
        operations=[operation("op-legacy", submission_count=1, accounting_complete=False)]
    )
    with pytest.raises(BudgetGuardViolation) as exc:
        authorize_checkpoint_budget_transition(
            context={},
            previous_checkpoint={"scenes": [{"index": 0, "operation": {"submission_count": None}}]},
            checkpoint=current,
            policy=policy(),
            rate_card=rate_card(),
            now_epoch=1000,
        )
    assert exc.value.code == "budget_legacy_accounting_unknown"
