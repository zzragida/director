import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_accounting import GenerationCostSummary
from director.core.generation_provenance import GenerationProvenanceManifest


EXPORT_SCHEMA = "director.generation.provenance"
EXPORT_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def export_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class PortableSceneLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    index: int
    operation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    operation_state: Optional[str] = None
    attempt_count: int = 0
    plan_digest: str
    prompt_digest: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_collection_id: Optional[str] = None
    artifact_length: Optional[float] = None


class PortableAudioLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    operation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    operation_state: Optional[str] = None
    attempt_count: int = 0
    prompt_digest: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_collection_id: Optional[str] = None
    artifact_length: Optional[float] = None


class PortableFinalLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: Optional[str] = None
    source_scene_artifact_ids: List[str] = Field(default_factory=list)
    source_audio_artifact_id: Optional[str] = None
    stream_url: Optional[str] = None


class PortableCostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Optional[str] = None
    total_amount: Optional[str] = None
    retry_waste_amount: Optional[str] = None
    estimated_operation_count: int = 0
    actual_operation_count: int = 0
    unknown_operation_count: int = 0
    rate_card_id: Optional[str] = None
    rate_card_effective_at: Optional[str] = None
    rate_card_source_reference: Optional[str] = None


class PortableGenerationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: str = EXPORT_SCHEMA
    version: int = EXPORT_VERSION
    export_id: str
    export_digest: Optional[str] = None
    source_manifest_id: str
    source_manifest_digest: Optional[str] = None
    generation_run_id: str
    request_fingerprint: str
    collection_id: Optional[str] = None
    storyline_digest: Optional[str] = None
    video_provider: Optional[str] = None
    audio_provider: Optional[str] = None
    scenes: List[PortableSceneLineage] = Field(default_factory=list)
    audio: Optional[PortableAudioLineage] = None
    final: Optional[PortableFinalLineage] = None
    cost: PortableCostSummary = Field(default_factory=PortableCostSummary)

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"export_digest"})
        self.export_digest = export_digest(payload)


def _model_metadata(config: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    model = config.get("model") or config.get("model_id")
    version = config.get("model_version") or config.get("version")
    return (
        str(model) if model not in (None, "") else None,
        str(version) if version not in (None, "") else None,
    )


def _portable_cost(accounting: Optional[GenerationCostSummary]) -> PortableCostSummary:
    if accounting is None:
        return PortableCostSummary()
    rate_card = accounting.rate_card
    return PortableCostSummary(
        currency=accounting.currency,
        total_amount=str(accounting.total_amount) if accounting.total_amount is not None else None,
        retry_waste_amount=(
            str(accounting.retry_waste_amount)
            if accounting.retry_waste_amount is not None
            else None
        ),
        estimated_operation_count=accounting.estimated_operation_count,
        actual_operation_count=accounting.actual_operation_count,
        unknown_operation_count=accounting.unknown_operation_count,
        rate_card_id=rate_card.rate_card_id if rate_card else None,
        rate_card_effective_at=rate_card.effective_at if rate_card else None,
        rate_card_source_reference=rate_card.source_reference if rate_card else None,
    )


def make_export_id(manifest: GenerationProvenanceManifest) -> str:
    seed = f"{manifest.manifest_id}|{manifest.manifest_digest or ''}|v{EXPORT_VERSION}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"genexport:{digest}"


def export_portable_manifest(
    manifest: GenerationProvenanceManifest,
) -> PortableGenerationManifest:
    """Create a shareable manifest without raw prompts, configs, or provider request IDs."""

    scenes: List[PortableSceneLineage] = []
    for scene in manifest.scenes:
        model, model_version = _model_metadata(scene.provider_config)
        scenes.append(
            PortableSceneLineage(
                index=scene.index,
                operation_id=scene.operation_id,
                provider=scene.provider,
                model=model,
                model_version=model_version,
                operation_state=scene.operation_state,
                attempt_count=scene.attempt_count,
                plan_digest=scene.plan_digest,
                prompt_digest=scene.prompt_digest,
                artifact_id=scene.artifact_id,
                artifact_collection_id=scene.artifact_collection_id,
                artifact_length=scene.artifact_length,
            )
        )

    audio = None
    if manifest.audio is not None:
        model, model_version = _model_metadata(manifest.audio.provider_config)
        audio = PortableAudioLineage(
            operation_id=manifest.audio.operation_id,
            provider=manifest.audio.provider,
            model=model,
            model_version=model_version,
            operation_state=manifest.audio.operation_state,
            attempt_count=manifest.audio.attempt_count,
            prompt_digest=manifest.audio.prompt_digest,
            artifact_id=manifest.audio.artifact_id,
            artifact_collection_id=manifest.audio.artifact_collection_id,
            artifact_length=manifest.audio.artifact_length,
        )

    final = None
    if manifest.final is not None:
        final = PortableFinalLineage(
            operation_id=manifest.final.operation_id,
            source_scene_artifact_ids=list(manifest.final.source_scene_artifact_ids),
            source_audio_artifact_id=manifest.final.source_audio_artifact_id,
            stream_url=manifest.final.stream_url,
        )

    exported = PortableGenerationManifest(
        export_id=make_export_id(manifest),
        source_manifest_id=manifest.manifest_id,
        source_manifest_digest=manifest.manifest_digest,
        generation_run_id=manifest.generation_run_id,
        request_fingerprint=manifest.request.request_fingerprint,
        collection_id=manifest.request.collection_id,
        storyline_digest=manifest.request.storyline_digest,
        video_provider=manifest.request.video_provider,
        audio_provider=manifest.request.audio_provider,
        scenes=scenes,
        audio=audio,
        final=final,
        cost=_portable_cost(getattr(manifest, "accounting", None)),
    )
    exported.refresh_digest()
    return exported
