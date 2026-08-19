from types import SimpleNamespace

from director.core.generation_lifecycle import (
    create_operation,
    record_persisted,
    record_provider_request,
)
from director.core.generation_provenance import (
    create_provenance_manifest,
    redact_generation_config,
    safe_provenance_summary,
    stable_digest,
    sync_provenance_manifest,
)


def test_generation_config_redacts_credentials_without_redacting_normal_token_settings():
    config = {
        "seed": 7,
        "max_tokens": 512,
        "api_key": "secret-api-key",
        "nested": {
            "client_secret": "secret-client",
            "authorization": "Bearer secret",
            "motion_strength": 0.7,
        },
    }

    sanitized = redact_generation_config(config)

    assert sanitized["seed"] == 7
    assert sanitized["max_tokens"] == 512
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["client_secret"] == "[REDACTED]"
    assert sanitized["nested"]["authorization"] == "[REDACTED]"
    assert sanitized["nested"]["motion_strength"] == 0.7
    assert "secret-api-key" not in str(sanitized)
    assert "secret-client" not in str(sanitized)


def test_manifest_records_request_and_lineage_without_mutating_source_config():
    video_config = {"seed": 7, "api_key": "do-not-store"}
    manifest = create_provenance_manifest(
        generation_run_id="genrun:test",
        checkpoint_id="text_to_movie:test",
        request_fingerprint="fingerprint-1",
        collection_id="collection-1",
        storyline="A quiet reunion",
        video_provider="kling",
        audio_provider="elevenlabs",
        video_config=video_config,
        audio_config={"duration": 10, "access_token": "secret"},
    )

    assert manifest.request.collection_id == "collection-1"
    assert manifest.request.storyline == "A quiet reunion"
    assert manifest.request.storyline_digest == stable_digest("A quiet reunion")
    assert manifest.request.video_config["seed"] == 7
    assert manifest.request.video_config["api_key"] == "[REDACTED]"
    assert manifest.request.audio_config["access_token"] == "[REDACTED]"
    assert video_config["api_key"] == "do-not-store"
    assert manifest.manifest_digest


def test_sync_manifest_tracks_scene_audio_and_final_artifact_sources():
    generation_run_id = "genrun:test"
    scene_operation = create_operation(
        generation_run_id,
        kind="scene_video",
        provider="kling",
        index=0,
    )
    record_provider_request(scene_operation, "provider-task-1")
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
                prompt="cinematic apartment reunion",
                media=scene_media,
                operation=scene_operation,
            )
        ],
        audio_prompt="soft piano reunion",
        audio_media=audio_media,
        audio_operation=audio_operation,
    )
    manifest = create_provenance_manifest(
        generation_run_id=generation_run_id,
        checkpoint_id="text_to_movie:test",
        request_fingerprint="fingerprint-1",
        video_provider="kling",
        audio_provider="elevenlabs",
        video_config={"seed": 9},
        audio_config={"duration": 4},
    )
    initial_digest = manifest.manifest_digest

    sync_provenance_manifest(manifest, checkpoint)

    scene = manifest.scenes[0]
    assert scene.operation_id == scene_operation.operation_id
    assert scene.provider == "kling"
    assert scene.provider_request_id == "provider-task-1"
    assert scene.prompt == "cinematic apartment reunion"
    assert scene.prompt_digest == stable_digest(scene.prompt)
    assert scene.provider_config == {"seed": 9}
    assert scene.artifact_id == "video-1"
    assert scene.artifact_collection_id == "collection-1"
    assert manifest.audio.artifact_id == "audio-1"
    assert manifest.audio.prompt == "soft piano reunion"
    assert manifest.final.source_scene_artifact_ids == ["video-1"]
    assert manifest.final.source_audio_artifact_id == "audio-1"
    assert manifest.final.stream_url == "https://stream.example/final.m3u8"
    assert manifest.manifest_digest != initial_digest


def test_safe_summary_excludes_prompts_storyline_configs_and_provider_request_ids():
    manifest = create_provenance_manifest(
        generation_run_id="genrun:test",
        checkpoint_id="text_to_movie:test",
        request_fingerprint="fingerprint-1",
        collection_id="collection-1",
        storyline="private storyline",
        video_provider="kling",
        video_config={"seed": 1, "api_key": "private-key"},
    )
    checkpoint = SimpleNamespace(
        generation_run_id="genrun:test",
        final_video=None,
        scenes=[
            SimpleNamespace(
                index=0,
                plan={"story_beat": "opening"},
                prompt="private prompt",
                media=None,
                operation=SimpleNamespace(
                    operation_id="genop:scene:1",
                    provider="kling",
                    state="submitted",
                    attempt_count=1,
                    provider_request_id="private-provider-id",
                ),
            )
        ],
        audio_prompt=None,
        audio_media=None,
        audio_operation=None,
    )
    sync_provenance_manifest(manifest, checkpoint)

    summary = safe_provenance_summary(manifest)
    rendered = str(summary)

    assert summary["manifest_id"] == manifest.manifest_id
    assert summary["manifest_digest"] == manifest.manifest_digest
    assert summary["scene_count"] == 1
    assert "private storyline" not in rendered
    assert "private prompt" not in rendered
    assert "private-provider-id" not in rendered
    assert "private-key" not in rendered
