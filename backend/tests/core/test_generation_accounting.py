import json
from decimal import Decimal
from types import SimpleNamespace

from director.core.generation_accounting import (
    CostStatus,
    RateCardLine,
    RateCardSnapshot,
    build_cost_summary,
    infer_model_metadata,
    load_rate_card_from_environment,
)
from director.core.generation_lifecycle import create_operation, record_persisted
from director.core.generation_provenance import (
    create_provenance_manifest,
    sync_provenance_manifest,
)


def make_manifest(*, attempts=1, model="kling-v1", length=4):
    generation_run_id = "genrun:cost-test"
    operation = create_operation(
        generation_run_id,
        kind="scene_video",
        provider="kling",
        index=0,
    )
    operation.attempt_count = attempts
    media = {"id": "video-1", "length": length, "collection_id": "collection-1"}
    record_persisted(operation, media)
    operation.attempt_count = attempts

    checkpoint = SimpleNamespace(
        generation_run_id=generation_run_id,
        final_video=None,
        scenes=[
            SimpleNamespace(
                index=0,
                plan={"story_beat": "opening", "suggested_duration": length},
                prompt="cinematic opening",
                media=media,
                operation=operation,
            )
        ],
        audio_prompt=None,
        audio_media=None,
        audio_operation=None,
    )
    manifest = create_provenance_manifest(
        generation_run_id=generation_run_id,
        checkpoint_id="text_to_movie:cost-test",
        request_fingerprint="fingerprint-cost",
        video_provider="kling",
        audio_provider="videodb",
        video_config={"model": model},
    )
    sync_provenance_manifest(manifest, checkpoint)
    return manifest


def test_accounting_records_usage_without_inventing_price():
    manifest = make_manifest(length=4)
    accounting = manifest.accounting

    assert accounting is not None
    assert accounting.total_amount is None
    assert accounting.currency is None
    assert accounting.unknown_operation_count == 1
    scene = accounting.operations[0]
    assert scene.cost_status == CostStatus.unknown
    assert scene.model == "kling-v1"
    assert {(item.unit, item.quantity) for item in scene.usage} == {
        ("operation", 1.0),
        ("attempt", 1.0),
        ("output_second", 4.0),
    }


def test_operator_rate_card_estimates_only_matching_explicit_unit():
    manifest = make_manifest(length=4)
    rate_card = RateCardSnapshot(
        rate_card_id="rates-2026-08",
        currency="usd",
        effective_at="2026-08-01",
        source_reference="finance-approved-sheet-v3",
        lines=[
            RateCardLine(
                provider="kling",
                operation_kind="scene_video",
                model="kling-v1",
                unit="output_second",
                price_per_unit=Decimal("0.25"),
            )
        ],
    )

    accounting = build_cost_summary(manifest, rate_card=rate_card)

    assert accounting.currency == "USD"
    assert accounting.total_amount == Decimal("1.000000")
    assert accounting.estimated_operation_count == 1
    assert accounting.unknown_operation_count == 0
    scene = accounting.operations[0]
    assert scene.cost_status == CostStatus.estimated
    assert scene.amount == Decimal("1.000000")
    assert scene.pricing_unit == "output_second"
    assert scene.rate_card_id == "rates-2026-08"
    assert scene.retry_waste_amount is None


def test_attempt_pricing_can_attribute_retry_waste_without_guessing_failed_duration():
    manifest = make_manifest(attempts=3, length=4)
    rate_card = RateCardSnapshot(
        rate_card_id="attempt-rates",
        currency="USD",
        lines=[
            RateCardLine(
                provider="kling",
                operation_kind="scene_video",
                unit="attempt",
                price_per_unit=Decimal("2.50"),
            )
        ],
    )

    accounting = build_cost_summary(manifest, rate_card=rate_card)
    scene = accounting.operations[0]

    assert scene.amount == Decimal("7.500000")
    assert scene.retry_waste_amount == Decimal("5.000000")
    assert accounting.retry_waste_amount == Decimal("5.000000")


def test_model_specific_rate_does_not_apply_to_unknown_or_other_model():
    manifest = make_manifest(model="kling-v1")
    rate_card = RateCardSnapshot(
        rate_card_id="model-rates",
        currency="USD",
        lines=[
            RateCardLine(
                provider="kling",
                operation_kind="scene_video",
                model="kling-v2",
                unit="output_second",
                price_per_unit=Decimal("9"),
            )
        ],
    )

    accounting = build_cost_summary(manifest, rate_card=rate_card)
    assert accounting.total_amount is None
    assert accounting.operations[0].cost_status == CostStatus.unknown


def test_rate_card_environment_is_explicit_and_invalid_payload_is_ignored(monkeypatch):
    monkeypatch.setenv(
        "GENERATION_RATE_CARD_JSON",
        json.dumps(
            {
                "rate_card_id": "env-rates",
                "currency": "KRW",
                "lines": [
                    {
                        "provider": "kling",
                        "operation_kind": "scene_video",
                        "unit": "operation",
                        "price_per_unit": "1000",
                    }
                ],
            }
        ),
    )
    loaded = load_rate_card_from_environment()
    assert loaded is not None
    assert loaded.rate_card_id == "env-rates"
    assert loaded.currency == "KRW"

    monkeypatch.setenv("GENERATION_RATE_CARD_JSON", "not-json")
    assert load_rate_card_from_environment() is None


def test_model_metadata_is_evidence_driven_from_config_only():
    assert infer_model_metadata({"model": "kling-v1", "model_version": "2026-08"}) == (
        "kling-v1",
        "2026-08",
    )
    assert infer_model_metadata({"model_id": "eleven_multilingual_v2"}) == (
        "eleven_multilingual_v2",
        None,
    )
    assert infer_model_metadata({}) == (None, None)
