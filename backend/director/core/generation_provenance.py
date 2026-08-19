import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_accounting import (
    GenerationCostSummary,
    build_cost_summary,
    load_rate_card_from_environment,
)
from director.core.generation_lifecycle import make_operation_id


PROVENANCE_VERSION = 2
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_key",
    "secret_key",
    "access_token",
    "refresh_token",
    "client_secret",
    "authorization",
    "password",
    "credential",
    "credentials",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def redact_generation_config(value: Any) -> Any:
    """Return a provenance-safe copy without common credential fields."""

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_KEYS:
                sanitized[key] = _REDACTED
            else:
                sanitized[key] = redact_generation_config(item)
        return sanitized
    if isinstance(value, list):
        return [redact_generation_config(item) for item in value]
    if isinstance(value, tuple):
        return [redact_generation_config(item) for item in value]
    return value


class GenerationRequestProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_fingerprint: str
    collection_id: Optional[str] = None
    storyline: Optional[str] = None
    storyline_digest: Optional[str] = None
    video_provider: Optional[str] = None
    audio_provider: Optional[str] = None
    video_config: Dict[str, Any] = Field(default_factory=dict)
    audio_config: Dict[str, Any] = Field(default_factory=dict)


class SceneArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    operation_id: Optional[str] = None
    provider: Optional[str] = None
    operation_state: Optional[str] = None
    attempt_count: int = Field(default=0, ge=0)
    provider_request_id: Optional[str] = None
    plan_digest: str
    prompt: Optional[str] = None
    prompt_digest: Optional[str] = None
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    artifact_id: Optional[str] = None
    artifact_collection_id: Optional[str] = None
    artifact_length: Optional[float] = None


class AudioArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: Optional[str] = None
    provider: Optional[str] = None
    operation_state: Optional[str] = None
    attempt_count: int = Field(default=0, ge=0)
    provider_request_id: Optional[str] = None
    prompt: Optional[str] = None
    prompt_digest: Optional[str] = None
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    artifact_id: Optional[str] = None
    artifact_collection_id: Optional[str] = None
    artifact_length: Optional[float] = None


class FinalArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: Optional[str] = None
    stream_url: Optional[str] = None
    source_scene_artifact_ids: List[str] = Field(default_factory=list)
    source_audio_artifact_id: Optional[str] = None


class GenerationProvenanceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = PROVENANCE_VERSION
    manifest_id: str
    generation_run_id: str
    checkpoint_id: str
    request: GenerationRequestProvenance
    scenes: List[SceneArtifactProvenance] = Field(default_factory=list)
    audio: Optional[AudioArtifactProvenance] = None
    final: Optional[FinalArtifactProvenance] = None
    accounting: Optional[GenerationCostSummary] = None
    manifest_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"manifest_digest"})
        self.manifest_digest = stable_digest(payload)


def make_manifest_id(generation_run_id: str) -> str:
    digest = hashlib.sha256(generation_run_id.encode("utf-8")).hexdigest()[:24]
    return f"genprov:{digest}"


def create_provenance_manifest(
    *,
    generation_run_id: str,
    checkpoint_id: str,
    request_fingerprint: str,
    collection_id: Optional[str] = None,
    storyline: Optional[str] = None,
    video_provider: Optional[str] = None,
    audio_provider: Optional[str] = None,
    video_config: Optional[Dict[str, Any]] = None,
    audio_config: Optional[Dict[str, Any]] = None,
) -> GenerationProvenanceManifest:
    manifest = GenerationProvenanceManifest(
        manifest_id=make_manifest_id(generation_run_id),
        generation_run_id=generation_run_id,
        checkpoint_id=checkpoint_id,
        request=GenerationRequestProvenance(
            request_fingerprint=request_fingerprint,
            collection_id=collection_id,
            storyline=storyline,
            storyline_digest=(
                stable_digest(storyline) if storyline is not None else None
            ),
            video_provider=video_provider,
            audio_provider=audio_provider,
            video_config=redact_generation_config(video_config or {}),
            audio_config=redact_generation_config(audio_config or {}),
        ),
    )
    manifest.accounting = build_cost_summary(manifest)
    manifest.refresh_digest()
    return manifest


def _operation_fields(operation: Any) -> Dict[str, Any]:
    if operation is None:
        return {
            "operation_id": None,
            "provider": None,
            "operation_state": None,
            "attempt_count": 0,
            "provider_request_id": None,
        }
    return {
        "operation_id": getattr(operation, "operation_id", None),
        "provider": getattr(operation, "provider", None),
        "operation_state": str(getattr(operation, "state", "")) or None,
        "attempt_count": int(getattr(operation, "attempt_count", 0) or 0),
        "provider_request_id": getattr(operation, "provider_request_id", None),
    }


def sync_provenance_manifest(
    manifest: GenerationProvenanceManifest,
    checkpoint: Any,
) -> GenerationProvenanceManifest:
    """Synchronize lineage and accounting from the authoritative checkpoint."""

    video_config = redact_generation_config(manifest.request.video_config)
    scene_entries: List[SceneArtifactProvenance] = []
    for scene in getattr(checkpoint, "scenes", []) or []:
        media = getattr(scene, "media", None) or {}
        prompt = getattr(scene, "prompt", None)
        operation = getattr(scene, "operation", None)
        scene_entries.append(
            SceneArtifactProvenance(
                index=int(getattr(scene, "index", 0)),
                plan_digest=stable_digest(getattr(scene, "plan", {}) or {}),
                prompt=prompt,
                prompt_digest=stable_digest(prompt) if prompt else None,
                provider_config=video_config,
                artifact_id=media.get("id"),
                artifact_collection_id=media.get("collection_id"),
                artifact_length=media.get("length"),
                **_operation_fields(operation),
            )
        )
    manifest.scenes = scene_entries

    audio_operation = getattr(checkpoint, "audio_operation", None)
    audio_media = getattr(checkpoint, "audio_media", None) or {}
    audio_prompt = getattr(checkpoint, "audio_prompt", None)
    manifest.audio = AudioArtifactProvenance(
        prompt=audio_prompt,
        prompt_digest=stable_digest(audio_prompt) if audio_prompt else None,
        provider_config=redact_generation_config(manifest.request.audio_config),
        artifact_id=audio_media.get("id"),
        artifact_collection_id=audio_media.get("collection_id"),
        artifact_length=audio_media.get("length"),
        **_operation_fields(audio_operation),
    )

    scene_ids = [entry.artifact_id for entry in manifest.scenes if entry.artifact_id]
    final_video = getattr(checkpoint, "final_video", None)
    if final_video or scene_ids or manifest.audio.artifact_id:
        manifest.final = FinalArtifactProvenance(
            operation_id=make_operation_id(
                manifest.generation_run_id,
                kind="combine",
            ),
            stream_url=final_video,
            source_scene_artifact_ids=scene_ids,
            source_audio_artifact_id=manifest.audio.artifact_id,
        )

    prior_rate_card = manifest.accounting.rate_card if manifest.accounting else None
    rate_card = prior_rate_card or load_rate_card_from_environment()
    manifest.accounting = build_cost_summary(manifest, rate_card=rate_card)
    manifest.version = PROVENANCE_VERSION
    manifest.refresh_digest()
    return manifest


def safe_provenance_summary(
    manifest: Optional[GenerationProvenanceManifest],
) -> Dict[str, Any]:
    if manifest is None:
        return {}
    accounting = manifest.accounting
    return {
        "manifest_id": manifest.manifest_id,
        "manifest_digest": manifest.manifest_digest,
        "generation_run_id": manifest.generation_run_id,
        "scene_count": len(manifest.scenes),
        "persisted_scene_count": sum(1 for scene in manifest.scenes if scene.artifact_id),
        "audio_persisted": bool(manifest.audio and manifest.audio.artifact_id),
        "finalized": bool(manifest.final and manifest.final.stream_url),
        "cost_known": bool(accounting and accounting.total_amount is not None),
        "cost_currency": accounting.currency if accounting else None,
    }
