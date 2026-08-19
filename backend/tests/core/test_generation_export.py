from decimal import Decimal
from types import SimpleNamespace

from director.core.generation_accounting import RateCardLine, RateCardSnapshot, build_cost_summary
from director.core.generation_export import EXPORT_SCHEMA, export_portable_manifest
from director.core.generation_lifecycle import (
    create_operation,
    record_persisted,
    record_provider_request,
)
from director.core.generation_provenance import (
    create_provenance_manifest,
    sync_provenance_manifest,
)


def make_exportable_manifest():
    generation_run_id = "genrun:export-test"
    scene_operation = create_operation(
        generation_run_id,
        kind="scene_video",
        provider="kling",
        index=0,
    )
    record_provider_request(scene_operation, "private-provider-task")
    scene_media = {"id": "video-1", "length": 4, "collection_id": "collection-1"}
    record_persisted(scene_operation, scene_media)

    audio_operation = create_operation(
        generation_run_id,
        kind="background_audio",
        provider="elevenlabs",
    )
    audio_media = {"id": "audio-1", "length": 4, "collection_id": "collection-1"}
    record_persisted(audio_operation, audio_media)

    checkpoint = SimpleNamespace(
        generation_run_id=generation_run_id,
        final_video="https://stream.example/final.m3u8",
        scenes=[
            SimpleNamespace(
                index=0,
                plan={"story_beat": "opening", "suggested_duration": 4},
                prompt="private cinematic prompt",
                media=scene_media,
                operation=scene_operation,
            )
        ],
        audio_prompt="private audio prompt",
        audio_media=audio_media,
        audio_operation=audio_operation,
    )
    manifest = create_provenance_manifest(
        generation_run_id=generation_run_id,
        checkpoint_id="text_to_movie:export-test",
        request_fingerprint="fingerprint-export",
        collection_id="collection-1",
        storyline="private storyline",
        video_provider="kling",
        audio_provider="elevenlabs",
        video_config={"model": "kling-v1", "api_key": "private-api-key"},
        audio_config={"model_id": "sound-effect-v1", "access_token": "private-token"},
    )
    sync_provenance_manifest(manifest, checkpoint)
    manifest.accounting = build_cost_summary(
        manifest,
        rate_card=RateCardSnapshot(
            rate_card_id="rates-1",
            currency="USD",
            effective_at="2026-08-01",
            source_reference="approved-rate-card",
            lines=[
                RateCardLine(
                    provider="kling",
                    operation_kind="scene_video",
                    model="kling-v1",
                    unit="output_second",
                    price_per_unit=Decimal("0.25"),
                ),
                RateCardLine(
                    provider="elevenlabs",
                    operation_kind="background_audio",
                    model="sound-effect-v1",
                    unit="output_second",
                    price_per_unit=Decimal("0.10"),
                ),
            ],
        ),
    )
    manifest.refresh_digest()
    return manifest


def test_portable_export_contains_digests_lineage_models_and_cost_summary():
    manifest = make_exportable_manifest()

    exported = export_portable_manifest(manifest)

    assert exported.schema_name == EXPORT_SCHEMA
    assert exported.source_manifest_id == manifest.manifest_id
    assert exported.source_manifest_digest == manifest.manifest_digest
    assert exported.request_fingerprint == "fingerprint-export"
    assert exported.storyline_digest == manifest.request.storyline_digest
    assert exported.scenes[0].provider == "kling"
    assert exported.scenes[0].model == "kling-v1"
    assert exported.scenes[0].prompt_digest == manifest.scenes[0].prompt_digest
    assert exported.scenes[0].artifact_id == "video-1"
    assert exported.audio.model == "sound-effect-v1"
    assert exported.final.source_scene_artifact_ids == ["video-1"]
    assert exported.final.source_audio_artifact_id == "audio-1"
    assert exported.cost.currency == "USD"
    assert exported.cost.total_amount == "1.400000"
    assert exported.cost.rate_card_id == "rates-1"
    assert exported.export_digest


def test_portable_export_does_not_expose_raw_prompts_storyline_configs_or_provider_ids():
    manifest = make_exportable_manifest()
    exported = export_portable_manifest(manifest)
    rendered = exported.model_dump_json()

    assert "private storyline" not in rendered
    assert "private cinematic prompt" not in rendered
    assert "private audio prompt" not in rendered
    assert "private-provider-task" not in rendered
    assert "private-api-key" not in rendered
    assert "private-token" not in rendered
    assert "provider_config" not in rendered


def test_portable_export_is_deterministic_for_same_source_manifest():
    manifest = make_exportable_manifest()
    first = export_portable_manifest(manifest)
    second = export_portable_manifest(manifest)

    assert first.export_id == second.export_id
    assert first.export_digest == second.export_digest


def test_portable_export_digest_changes_when_source_lineage_changes():
    manifest = make_exportable_manifest()
    first = export_portable_manifest(manifest)

    manifest.scenes[0].artifact_id = "video-2"
    manifest.refresh_digest()
    second = export_portable_manifest(manifest)

    assert first.source_manifest_digest != second.source_manifest_digest
    assert first.export_id != second.export_id
    assert first.export_digest != second.export_digest
