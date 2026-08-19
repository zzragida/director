from director.core.generation_cost import (
    GenerationCostRate,
    GenerationCostRateCard,
    PricingBasis,
)
from director.core.generation_provenance import (
    AudioArtifactProvenance,
    FinalArtifactProvenance,
    SceneArtifactProvenance,
    create_provenance_manifest,
)
from director.core.provenance_export import (
    create_portable_provenance_export,
    safe_portable_export_summary,
)


def make_manifest():
    manifest = create_provenance_manifest(
        generation_run_id="genrun:portable",
        checkpoint_id="checkpoint-portable",
        request_fingerprint="fingerprint-portable",
        collection_id="collection-private",
        storyline="private storyline text",
        video_provider="kling",
        audio_provider="elevenlabs",
        video_config={
            "model": "kling-v1",
            "mode": "pro",
            "api_key": "[REDACTED]",
        },
        audio_config={"voice": "music", "access_token": "[REDACTED]"},
    )
    manifest.scenes = [
        SceneArtifactProvenance(
            index=0,
            operation_id="genop:scene_video:0:portable",
            provider="kling",
            operation_state="persisted",
            attempt_count=3,
            submission_count=2,
            resume_count=1,
            accounting_complete=True,
            provider_request_id="private-provider-task",
            plan_digest="plan-digest",
            planned_duration_seconds=5,
            prompt="private scene prompt",
            prompt_digest="prompt-digest",
            provider_config={"model": "kling-v1", "mode": "pro"},
            artifact_id="internal-video-id",
            artifact_collection_id="collection-private",
            artifact_length=5,
        )
    ]
    manifest.audio = AudioArtifactProvenance(
        operation_id="genop:background_audio:portable",
        provider="elevenlabs",
        operation_state="persisted",
        attempt_count=1,
        submission_count=1,
        resume_count=0,
        accounting_complete=True,
        provider_request_id="private-audio-task",
        prompt="private audio prompt",
        prompt_digest="audio-prompt-digest",
        provider_config={"voice": "music"},
        artifact_id="internal-audio-id",
        artifact_collection_id="collection-private",
        artifact_length=9,
    )
    manifest.final = FinalArtifactProvenance(
        operation_id="genop:combine:portable",
        stream_url="https://stream.example/private-final.m3u8",
        source_scene_artifact_ids=["internal-video-id"],
        source_audio_artifact_id="internal-audio-id",
    )
    manifest.refresh_digest()
    return manifest


def make_rate_card():
    return GenerationCostRateCard(
        rate_card_id="portable-costs",
        rate_card_version="2026-08",
        rates=[
            GenerationCostRate(
                provider="kling",
                media_type="video",
                model="kling-v1",
                basis=PricingBasis.per_submission,
                unit_price_micros=100_000,
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


def test_portable_export_replaces_sensitive_internal_values_with_digests_and_refs():
    export = create_portable_provenance_export(
        make_manifest(),
        rate_card=make_rate_card(),
    )
    serialized = export.model_dump_json()

    assert export.request.video_model == "kling-v1"
    assert export.request.storyline_digest
    assert export.request.collection_reference_digest
    assert export.scenes[0].artifact_ref.startswith("artifact:")
    assert export.audio.artifact_ref.startswith("artifact:")
    assert export.final.source_scene_artifact_refs == [export.scenes[0].artifact_ref]
    assert export.final.source_audio_artifact_ref == export.audio.artifact_ref
    assert export.final.stream_url is None
    assert export.final.stream_url_digest

    for secret in (
        "private storyline text",
        "private scene prompt",
        "private audio prompt",
        "private-provider-task",
        "private-audio-task",
        "internal-video-id",
        "internal-audio-id",
        "collection-private",
        "https://stream.example/private-final.m3u8",
    ):
        assert secret not in serialized


def test_portable_export_cost_snapshot_is_bound_to_rate_card_digest():
    export = create_portable_provenance_export(
        make_manifest(),
        rate_card=make_rate_card(),
    )

    assert export.cost.pricing_status == "priced"
    assert export.cost.totals_micros_by_currency == {"USD": 290_000}
    assert export.cost.retry_cost_micros_by_currency == {"USD": 100_000}
    assert export.cost.rate_card_id == "portable-costs"
    assert export.cost.rate_card_digest
    assert export.export_digest


def test_export_digest_is_deterministic_and_changes_with_authoritative_lineage():
    first_manifest = make_manifest()
    second_manifest = make_manifest()
    rate_card = make_rate_card()

    first = create_portable_provenance_export(first_manifest, rate_card=rate_card)
    second = create_portable_provenance_export(second_manifest, rate_card=rate_card)
    assert first.export_id == second.export_id
    assert first.export_digest == second.export_digest

    second_manifest.scenes[0].artifact_length = 6
    second_manifest.refresh_digest()
    changed = create_portable_provenance_export(second_manifest, rate_card=rate_card)
    assert changed.export_id != first.export_id
    assert changed.export_digest != first.export_digest


def test_stream_url_is_opt_in_for_portable_export():
    manifest = make_manifest()
    safe = create_portable_provenance_export(manifest)
    with_url = create_portable_provenance_export(
        manifest,
        include_stream_url=True,
    )

    assert safe.final.stream_url is None
    assert with_url.final.stream_url == "https://stream.example/private-final.m3u8"
    assert safe.export_digest != with_url.export_digest


def test_canonical_json_and_safe_summary_are_stable_and_compact():
    export = create_portable_provenance_export(make_manifest())
    canonical = export.to_canonical_json()
    summary = safe_portable_export_summary(export)

    assert canonical == export.to_canonical_json()
    assert summary["export_id"] == export.export_id
    assert summary["export_digest"] == export.export_digest
    assert summary["known_submission_count"] == 3
    assert summary["known_retry_submission_count"] == 1
    assert "request" not in summary
    assert "scenes" not in summary
    assert "private" not in str(summary)
