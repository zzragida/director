import json

from director.core.generation_billing import ProviderBillingEvent
from director.core.generation_budget import BudgetReservation
from director.core.generation_spend import (
    GenerationSpendStatusService,
    build_generation_spend_run_status,
    build_generation_spend_session_status,
)


CHECKPOINT_KEY = "__text_to_movie_checkpoints__"
BUDGET_KEY = "__generation_budget_reservations__"
BILLING_KEY = "__generation_billing_events__"


def _message(document):
    return [{"role": "system", "content": json.dumps(document)}]


def _reservation(
    reservation_id,
    *,
    ordinal,
    amount,
    provider="kling",
    created=100,
):
    return BudgetReservation(
        reservation_id=reservation_id,
        generation_run_id="genrun:test",
        operation_id=f"genop:{ordinal}",
        submission_ordinal=ordinal,
        retry_submission=ordinal > 1,
        media_type="video",
        provider=provider,
        model="kling-v1",
        currency="USD",
        pricing_basis="per_submission",
        estimated_amount_micros=amount,
        estimated=False,
        policy_id="movie-budget",
        policy_version="3",
        policy_digest="policy-digest-3",
        rate_card_id="provider-rates",
        rate_card_version="2026-08",
        rate_card_digest="rate-digest",
        created_at_epoch=created,
    )


def _event(
    event_id,
    amount,
    *,
    reservation_id=None,
    status="finalized",
    provider="kling",
):
    return ProviderBillingEvent(
        event_id=event_id,
        provider=provider,
        generation_run_id="genrun:test",
        operation_id="private-operation",
        reservation_id=reservation_id,
        source="invoice",
        status=status,
        kind="charge",
        amount_micros=amount,
        currency="USD",
        invoice_id="private-invoice-id",
        occurred_at_epoch=200,
    )


def _context(reservations=None, events=None, *, checkpoint_failure=None):
    checkpoint = {
        "checkpoint_id": "text_to_movie:test",
        "generation_run_id": "genrun:test",
        "status": "failed" if checkpoint_failure else "generating",
        "failure_stage": "budget" if checkpoint_failure else None,
        "failure_code": checkpoint_failure,
    }
    context = {
        CHECKPOINT_KEY: _message({"text_to_movie:test": checkpoint}),
    }
    if reservations:
        context[BUDGET_KEY] = _message(
            {item.reservation_id: item.model_dump(mode="json") for item in reservations}
        )
    if events:
        context[BILLING_KEY] = _message(
            {item.event_key: item.model_dump(mode="json") for item in events}
        )
    return context


def _set_policy(monkeypatch, *, retry_limit=100, total_limit=250):
    monkeypatch.setenv(
        "DIRECTOR_GENERATION_BUDGET_POLICY_JSON",
        json.dumps(
            {
                "policy_id": "movie-budget",
                "policy_version": "3",
                "currency": "USD",
                "max_total_micros": total_limit,
                "max_retry_micros": retry_limit,
                "max_submissions_per_operation": 3,
                "max_retry_submissions_per_operation": 2,
                "require_preflight_pricing": True,
            }
        ),
    )
    monkeypatch.setenv(
        "DIRECTOR_GENERATION_RATE_CARD_JSON",
        json.dumps(
            {
                "rate_card_id": "provider-rates",
                "rate_card_version": "2026-08",
                "rates": [
                    {
                        "provider": "kling",
                        "media_type": "video",
                        "model": "kling-v1",
                        "basis": "per_submission",
                        "unit_price_micros": 100,
                        "currency": "USD",
                    }
                ],
            }
        ),
    )


def test_status_reconciles_actual_with_estimates_and_marks_variance_incomplete(monkeypatch):
    _set_policy(monkeypatch)
    first = _reservation("res-1", ordinal=1, amount=100, created=100)
    retry = _reservation("res-2", ordinal=2, amount=50, created=110)
    actual = _event("invoice-line-1", 120, reservation_id="res-1")

    status = build_generation_spend_run_status(
        _context([first, retry], [actual]),
        "genrun:test",
    )

    assert status.estimated_reserved_micros == 150
    assert status.actual_finalized_micros == 120
    assert status.effective_exposure_micros == 170
    assert status.retry_exposure_micros == 50
    assert status.remaining_total_micros == 80
    assert status.remaining_retry_micros == 50
    assert status.actual_minus_estimated_micros == -30
    assert status.variance_complete is False
    assert status.unresolved_reservation_count == 1
    assert status.circuit_breaker.state == "closed"
    assert status.policy.policy_id == "movie-budget"
    assert status.policy.rate_card_id == "provider-rates"
    assert len(status.policy_history) == 1
    assert status.policy_history[0].reservation_count == 2
    assert status.providers[0].submission_count == 2
    assert status.providers[0].retry_submission_count == 1


def test_retry_actual_can_open_retry_circuit_breaker(monkeypatch):
    _set_policy(monkeypatch, retry_limit=60, total_limit=250)
    first = _reservation("res-1", ordinal=1, amount=100)
    retry = _reservation("res-2", ordinal=2, amount=50)
    events = [
        _event("invoice-line-1", 120, reservation_id="res-1"),
        _event("invoice-line-2", 70, reservation_id="res-2"),
    ]

    status = build_generation_spend_run_status(
        _context([first, retry], events),
        "genrun:test",
    )

    assert status.effective_exposure_micros == 190
    assert status.retry_exposure_micros == 70
    assert status.remaining_retry_micros == 0
    assert status.variance_complete is True
    assert status.actual_minus_estimated_micros == 40
    assert status.circuit_breaker.state == "open"
    assert status.circuit_breaker.reason_codes == ["retry_budget_reached"]


def test_unmatched_actual_is_visible_and_budget_block_reason_is_safe(monkeypatch):
    _set_policy(monkeypatch, total_limit=180)
    reservation = _reservation("res-1", ordinal=1, amount=100)
    events = [
        _event("matched", 120, reservation_id="res-1"),
        _event("unmatched", 80, reservation_id=None),
    ]
    status = build_generation_spend_run_status(
        _context([reservation], events, checkpoint_failure="budget_total_exceeded"),
        "genrun:test",
    )

    assert status.effective_exposure_micros == 200
    assert status.unmatched_finalized_event_count == 1
    assert status.remaining_total_micros == 0
    assert status.last_block_reason == "budget_total_exceeded"
    assert status.circuit_breaker.state == "open"
    assert "total_budget_reached" in status.circuit_breaker.reason_codes

    rendered = json.dumps(status.model_dump(mode="json"))
    assert "private-invoice-id" not in rendered
    assert "invoice-line" not in rendered
    assert "res-1" not in rendered
    assert "private-operation" not in rendered


def test_status_is_disabled_without_policy_but_still_discovers_checkpoint_run(monkeypatch):
    monkeypatch.delenv("DIRECTOR_GENERATION_BUDGET_POLICY_JSON", raising=False)
    monkeypatch.delenv("DIRECTOR_GENERATION_RATE_CARD_JSON", raising=False)

    session_status = build_generation_spend_session_status(
        _context(),
        session_id="session-1",
    )

    assert session_status.run_count == 1
    run = session_status.runs[0]
    assert run.generation_run_id == "genrun:test"
    assert run.policy.enabled is False
    assert run.circuit_breaker.state == "disabled"
    assert run.effective_exposure_micros == 0


def test_invalid_budget_configuration_is_reported_in_read_model(monkeypatch):
    monkeypatch.setenv("DIRECTOR_GENERATION_BUDGET_POLICY_JSON", "{invalid")
    status = build_generation_spend_run_status(_context(), "genrun:test")

    assert status.policy.enabled is True
    assert status.policy.configuration_error == "invalid_budget_policy"
    assert status.circuit_breaker.state == "configuration_error"
    assert status.circuit_breaker.reason_codes == ["budget_configuration_invalid"]


def test_status_service_reads_context_without_mutating_it(monkeypatch):
    monkeypatch.delenv("DIRECTOR_GENERATION_BUDGET_POLICY_JSON", raising=False)
    context = _context()

    class FakeDB:
        def __init__(self):
            self.context = context
            self.reads = 0

        def get_context_messages(self, session_id):
            self.reads += 1
            return self.context

    db = FakeDB()
    service = GenerationSpendStatusService(db)
    session = service.session_status("session-1")
    run = service.run_status("session-1", "genrun:test")
    missing = service.run_status("session-1", "genrun:missing")

    assert session.run_count == 1
    assert run.generation_run_id == "genrun:test"
    assert missing is None
    assert db.reads == 3
