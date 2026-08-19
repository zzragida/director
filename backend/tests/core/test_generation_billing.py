import copy

import pytest

from director.core.generation_billing import (
    BillingLedgerConflictError,
    GenerationBillingLedgerStore,
    ProviderBillingEvent,
    reconcile_generation_billing,
)
from director.core.generation_budget import BudgetReservation


class FakeDB:
    def __init__(self):
        self.context = {}

    def get_context_messages(self, session_id):
        return copy.deepcopy(self.context.get(session_id, {}))

    def compare_and_swap_context_msg(self, session_id, expected_context, context_messages):
        current = self.context.get(session_id, {})
        if current != expected_context:
            return False
        self.context[session_id] = copy.deepcopy(context_messages)
        return True


class FakeSession:
    def __init__(self, db):
        self.db = db
        self.session_id = "session-1"


def reservation(amount=400):
    return BudgetReservation(
        reservation_id="reservation-1",
        generation_run_id="genrun:test",
        operation_id="genop:scene:1",
        submission_ordinal=1,
        media_type="video",
        provider="kling",
        currency="USD",
        estimated_amount_micros=amount,
        policy_id="policy-1",
        policy_version="1",
        created_at_epoch=1000,
    )


def test_billing_event_store_is_idempotent_and_conflicting_duplicates_fail():
    db = FakeDB()
    store = GenerationBillingLedgerStore(FakeSession(db))
    event = ProviderBillingEvent(
        event_id="invoice-line-1",
        provider="kling",
        generation_run_id="genrun:test",
        operation_id="genop:scene:1",
        reservation_id="reservation-1",
        source="invoice",
        amount_micros=450,
        currency="USD",
        invoice_id="invoice-2026-08",
    )

    assert store.record(event) is True
    assert store.record(event) is False
    assert len(store.events_for_run("genrun:test")) == 1

    changed = event.model_copy(update={"amount_micros": 999})
    with pytest.raises(BillingLedgerConflictError):
        store.record(changed)


def test_finalized_actual_replaces_reservation_estimate_for_effective_exposure():
    estimated = reservation(400)
    event = ProviderBillingEvent(
        event_id="actual-1",
        provider="kling",
        generation_run_id="genrun:test",
        operation_id=estimated.operation_id,
        reservation_id=estimated.reservation_id,
        amount_micros=700,
        currency="USD",
    )

    reconciliation = reconcile_generation_billing(
        generation_run_id="genrun:test",
        currency="USD",
        reservations=[estimated],
        events=[event],
    )

    assert reconciliation.estimated_reserved_micros == 400
    assert reconciliation.actual_finalized_micros == 700
    assert reconciliation.matched_reservation_count == 1
    assert reconciliation.effective_exposure_micros == 700


def test_unmatched_invoice_charge_is_added_to_effective_exposure():
    estimated = reservation(400)
    event = ProviderBillingEvent(
        event_id="unmatched-1",
        provider="kling",
        generation_run_id="genrun:test",
        amount_micros=250,
        currency="USD",
        source="invoice",
    )

    reconciliation = reconcile_generation_billing(
        generation_run_id="genrun:test",
        currency="USD",
        reservations=[estimated],
        events=[event],
    )

    assert reconciliation.unmatched_finalized_event_count == 1
    assert reconciliation.unmatched_actual_micros == 250
    assert reconciliation.effective_exposure_micros == 650


def test_credit_reduces_actual_exposure_but_never_below_zero():
    estimated = reservation(400)
    events = [
        ProviderBillingEvent(
            event_id="charge-1",
            provider="kling",
            generation_run_id="genrun:test",
            reservation_id=estimated.reservation_id,
            amount_micros=300,
            currency="USD",
        ),
        ProviderBillingEvent(
            event_id="credit-1",
            provider="kling",
            generation_run_id="genrun:test",
            reservation_id=estimated.reservation_id,
            amount_micros=500,
            currency="USD",
            kind="credit",
        ),
    ]

    reconciliation = reconcile_generation_billing(
        generation_run_id="genrun:test",
        currency="USD",
        reservations=[estimated],
        events=events,
    )

    assert reconciliation.actual_finalized_micros == -200
    assert reconciliation.effective_exposure_micros == 0
