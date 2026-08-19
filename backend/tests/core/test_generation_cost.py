from director.core.generation_cost import (
    GenerationCostRate,
    GenerationCostRateCard,
    PricingBasis,
    build_generation_cost_attribution,
    extract_model_hint,
    safe_cost_summary,
)
from director.core.generation_provenance import (
    AudioArtifactProvenance,
    SceneArtifactProvenance,
    create_provenance_manifest,
)


def make_manifest(*, accounting_complete=True):
    manifest = create_provenance_manifest(
        generation_run_id="genrun:test",
        checkpoint_id="checkpoint-1",
        request_fingerprint="fingerprint-1",
        collection_id="collection-1",
        storyline="A quiet reunion",
        video_provider="kling",
        audio_provider="elevenlabs",
        video_config={"model": "kling-v1", "mode": "std"},
        audio_config={"voice": "music"},
    )
    manifest.scenes = [
        SceneArtifactProvenance(
            index=0,
            operation_id="genop:scene_video:0:test",
            provider="kling",
            operation_state="persisted",
            attempt_count=3,
            submission_count=2,
            resume_count=1,
            accounting_complete=accounting_complete,
            provider_request_id="provider-task-secret",
            plan_digest="plan-digest",
            planned_duration_seconds=5,
            prompt="private prompt",
            prompt_digest="prompt-digest",
            provider_config={"model": "kling-v1", "mode": "std"},
            artifact_id="video-1",
            artifact_collection_id="collection-1",
            artifact_length=5,
        )
    ]
    manifest.audio = AudioArtifactProvenance(
        operation_id="genop:background_audio:test",
        provider="elevenlabs",
        operation_state="persisted",
        attempt_count=1,
        submission_count=1,
        resume_count=0,
        accounting_complete=True,
        prompt="private audio prompt",
        prompt_digest="audio-prompt-digest",
        provider_config={"voice": "music"},
        artifact_id="audio-1",
        artifact_collection_id="collection-1",
        artifact_length=8,
    )
    manifest.refresh_digest()
    return manifest


def test_no_rate_card_preserves_usage_facts_without_inventing_money():
    attribution = build_generation_cost_attribution(make_manifest())

    assert attribution.pricing_status == "unpriced"
    assert attribution.totals_micros_by_currency == {}
    assert attribution.known_submission_count == 3
    assert attribution.known_resume_count == 1
    assert attribution.known_retry_submission_count == 1
    assert attribution.unknown_accounting_operations == 0
    assert attribution.lines[0].retry_submission_count == 1
    assert attribution.lines[0].amount_micros is None


def test_explicit_rate_card_prices_submission_and_persisted_duration():
    rate_card = GenerationCostRateCard(
        rate_card_id="finance-2026q3",
        rate_card_version="2026.08.19",
        rates=[
            GenerationCostRate(
                provider="kling",
                media_type="video",
                model="kling-v1",
                basis=PricingBasis.per_submission,
                unit_price_micros=250_000,
                currency="USD",
            ),
            GenerationCostRate(
                provider="elevenlabs",
                media_type="audio",
                basis=PricingBasis.per_persisted_second,
                unit_price_micros=10_000,
                currency="USD",
            ),
        ],
    )

    attribution = build_generation_cost_attribution(
        make_manifest(),
        rate_card=rate_card,
    )

    assert attribution.pricing_status == "priced"
    assert attribution.totals_micros_by_currency == {"USD": 580_000}
    assert attribution.retry_cost_micros_by_currency == {"USD": 250_000}
    assert attribution.lines[0].amount_micros == 500_000
    assert attribution.lines[0].retry_cost_micros == 250_000
    assert attribution.lines[1].amount_micros == 80_000
    assert attribution.lines[1].retry_cost_micros is None
    assert attribution.rate_card_digest == rate_card.rate_card_digest


def test_planned_second_pricing_is_explicitly_marked_estimated():
    rate_card = GenerationCostRateCard(
        rate_card_id="estimated-video-rate",
        rate_card_version="1",
        rates=[
            GenerationCostRate(
                provider="kling",
                media_type="video",
                model="kling-v1",
                basis=PricingBasis.per_planned_second_submission,
                unit_price_micros=1_000,
                currency="USD",
            )
        ],
    )

    attribution = build_generation_cost_attribution(
        make_manifest(),
        rate_card=rate_card,
    )
    video_line = attribution.lines[0]

    assert video_line.pricing_status == "priced"
    assert video_line.quantity == 10
    assert video_line.amount_micros == 10_000
    assert video_line.retry_cost_micros == 5_000
    assert video_line.estimated is True
    assert attribution.estimated_priced_lines == 1
    assert attribution.pricing_status == "partial"


def test_legacy_accounting_is_not_priced_even_with_a_rate_card():
    rate_card = GenerationCostRateCard(
        rate_card_id="provider-rate",
        rate_card_version="1",
        rates=[
            GenerationCostRate(
                provider="kling",
                media_type="video",
                model="kling-v1",
                basis=PricingBasis.per_submission,
                unit_price_micros=1_000,
                currency="USD",
            )
        ],
    )

    attribution = build_generation_cost_attribution(
        make_manifest(accounting_complete=False),
        rate_card=rate_card,
    )

    assert attribution.lines[0].pricing_status == "legacy_accounting_unknown"
    assert attribution.lines[0].amount_micros is None
    assert attribution.unknown_accounting_operations == 1


def test_model_hint_is_only_derived_from_explicit_config_fields():
    assert extract_model_hint({"model": "kling-v1", "mode": "std"}) == "kling-v1"
    assert extract_model_hint({"model_version": 2}) == "2"
    assert extract_model_hint({"mode": "std", "seed": 7}) is None


def test_safe_cost_summary_exposes_totals_not_internal_line_details():
    attribution = build_generation_cost_attribution(make_manifest())
    summary = safe_cost_summary(attribution)

    assert summary["pricing_status"] == "unpriced"
    assert summary["known_submission_count"] == 3
    assert "lines" not in summary
    assert "provider-task-secret" not in str(summary)
    assert "private prompt" not in str(summary)
