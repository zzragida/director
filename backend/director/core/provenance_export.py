import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_cost import (
    GenerationCostAttribution,
    GenerationCostRateCard,
    build_generation_cost_attribution,
    extract_model_hint,
)
from director.core.generation_provenance import (
    GenerationProvenanceManifest,
    stable_digest,
)


PORTABLE_EXPORT_VERSION = 1


def _artifact_ref(artifact_id: Optional[str], collection_id: Optional[str]) -> Optional[str]:
    if not artifact_id:
        return None
    digest = stable_digest(
        {
            "artifact_id": artifact_id,
            "collection_id": collection_id,
        }
    )[:24]
    return f"artifact:{digest}"


def _config_digest(config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not config:
        return None
    return stable_digest(config)


class PortableRequestProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_fingerprint: str
    collection_reference_digest: Optional[str] = None
    storyline_digest: Optional[str] = None
    video_provider: Optional[str] = None
    audio_provider: Optional[str] = None
    video_model: Optional[str] = None
    audio_model: Optional[str] = None
    video_config_digest: Optional[str] = None
    audio_config_digest: Optional[str] = None


class PortableSceneLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    operation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    operation_state: Optional[str] = None
    accounting_complete: bool = False
    submission_count: Optional[int] = Field(default=None, ge=0)
    resume_count: Optional[int] = Field(default=None, ge=0)
    plan_digest: str
    prompt_digest: Optional[str] = None
    provider_config_digest: Optional[str] = None
    planned_duration_seconds: Optional[float] = Field(default=None, ge=0)
    artifact_ref: Optional[str] = None
    artifact_length: Optional[float] = None


class PortableAudioLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    operation_state: Optional[str] = None
    accounting_complete: bool = False
    submission_count: Optional[int] = Field(default=None, ge=0)
    resume_count: Optional[int] = Field(default=None, ge=0)
    prompt_digest: Optional[str] = None
    provider_config_digest: Optional[str] = None
    artifact_ref: Optional[str] = None
    artifact_length: Optional[float] = None


class PortableFinalLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: Optional[str] = None
    source_scene_artifact_refs: List[str] = Field(default_factory=list)
    source_audio_artifact_ref: Optional[str] = None
    stream_url: Optional[str] = None
    stream_url_digest: Optional[str] = None


class PortableProvenanceExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = PORTABLE_EXPORT_VERSION
    export_id: str
    provenance_manifest_id: str
    provenance_manifest_digest: Optional[str] = None
    generation_run_id: str
    request: PortableRequestProvenance
    scenes: List[PortableSceneLineage] = Field(default_factory=list)
    audio: Optional[PortableAudioLineage] = None
    final: Optional[PortableFinalLineage] = None
    cost: GenerationCostAttribution
    export_digest: Optional[str] = None

    def refresh_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"export_digest"})
        self.export_digest = stable_digest(payload)

    def to_canonical_json(self) -> str:
        self.refresh_digest()
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _make_export_id(
    manifest: GenerationProvenanceManifest,
    cost: GenerationCostAttribution,
) -> str:
    digest = stable_digest(
        {
            "manifest_id": manifest.manifest_id,
            "manifest_digest": manifest.manifest_digest,
            "rate_card_digest": cost.rate_card_digest,
        }
    )[:24]
    return f"genexport:{digest}"


def create_portable_provenance_export(
    manifest: GenerationProvenanceManifest,
    *,
    rate_card: Optional[GenerationCostRateCard] = None,
    include_stream_url: bool = False,
) -> PortableProvenanceExport:
    manifest.refresh_digest()
    cost = build_generation_cost_attribution(manifest, rate_card=rate_card)

    request = PortableRequestProvenance(
        request_fingerprint=manifest.request.request_fingerprint,
        collection_reference_digest=(
            stable_digest(manifest.request.collection_id)
            if manifest.request.collection_id
            else None
        ),
        storyline_digest=manifest.request.storyline_digest,
        video_provider=manifest.request.video_provider,
        audio_provider=manifest.request.audio_provider,
        video_model=extract_model_hint(manifest.request.video_config),
        audio_model=extract_model_hint(manifest.request.audio_config),
        video_config_digest=_config_digest(manifest.request.video_config),
        audio_config_digest=_config_digest(manifest.request.audio_config),
    )

    scenes: List[PortableSceneLineage] = []
    artifact_ref_by_id: Dict[str, str] = {}
    for scene in manifest.scenes:
        artifact_ref = _artifact_ref(scene.artifact_id, scene.artifact_collection_id)
        if scene.artifact_id and artifact_ref:
            artifact_ref_by_id[scene.artifact_id] = artifact_ref
        scenes.append(
            PortableSceneLineage(
                index=scene.index,
                operation_id=scene.operation_id,
                provider=scene.provider,
                model=extract_model_hint(scene.provider_config),
                operation_state=scene.operation_state,
                accounting_complete=scene.accounting_complete,
                submission_count=scene.submission_count,
                resume_count=scene.resume_count,
                plan_digest=scene.plan_digest,
                prompt_digest=scene.prompt_digest,
                provider_config_digest=_config_digest(scene.provider_config),
                planned_duration_seconds=scene.planned_duration_seconds,
                artifact_ref=artifact_ref,
                artifact_length=scene.artifact_length,
            )
        )

    audio = None
    audio_ref = None
    if manifest.audio is not None:
        audio_ref = _artifact_ref(
            manifest.audio.artifact_id,
            manifest.audio.artifact_collection_id,
        )
        audio = PortableAudioLineage(
            operation_id=manifest.audio.operation_id,
            provider=manifest.audio.provider,
            model=extract_model_hint(manifest.audio.provider_config),
            operation_state=manifest.audio.operation_state,
            accounting_complete=manifest.audio.accounting_complete,
            submission_count=manifest.audio.submission_count,
            resume_count=manifest.audio.resume_count,
            prompt_digest=manifest.audio.prompt_digest,
            provider_config_digest=_config_digest(manifest.audio.provider_config),
            artifact_ref=audio_ref,
            artifact_length=manifest.audio.artifact_length,
        )

    final = None
    if manifest.final is not None:
        final = PortableFinalLineage(
            operation_id=manifest.final.operation_id,
            source_scene_artifact_refs=[
                artifact_ref_by_id[artifact_id]
                for artifact_id in manifest.final.source_scene_artifact_ids
                if artifact_id in artifact_ref_by_id
            ],
            source_audio_artifact_ref=audio_ref,
            stream_url=(manifest.final.stream_url if include_stream_url else None),
            stream_url_digest=(
                stable_digest(manifest.final.stream_url)
                if manifest.final.stream_url
                else None
            ),
        )

    export = PortableProvenanceExport(
        export_id=_make_export_id(manifest, cost),
        provenance_manifest_id=manifest.manifest_id,
        provenance_manifest_digest=manifest.manifest_digest,
        generation_run_id=manifest.generation_run_id,
        request=request,
        scenes=scenes,
        audio=audio,
        final=final,
        cost=cost,
    )
    export.refresh_digest()
    return export


def safe_portable_export_summary(export: PortableProvenanceExport) -> Dict[str, Any]:
    return {
        "export_id": export.export_id,
        "export_digest": export.export_digest,
        "provenance_manifest_id": export.provenance_manifest_id,
        "provenance_manifest_digest": export.provenance_manifest_digest,
        "generation_run_id": export.generation_run_id,
        "scene_count": len(export.scenes),
        "finalized": bool(export.final and export.final.stream_url_digest),
        "pricing_status": export.cost.pricing_status,
        "totals_micros_by_currency": dict(export.cost.totals_micros_by_currency),
        "known_submission_count": export.cost.known_submission_count,
        "known_retry_submission_count": export.cost.known_retry_submission_count,
        "unknown_accounting_operations": export.cost.unknown_accounting_operations,
    }
