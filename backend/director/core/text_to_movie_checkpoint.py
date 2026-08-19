import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.context_cas import atomic_update_context
from director.core.generation_budget import (
    BudgetConfigurationError,
    BudgetGuardViolation,
    authorize_checkpoint_budget_transition,
    load_budget_configuration_from_env,
)
from director.core.generation_lease import (
    GenerationLease,
    get_active_fencing_lease,
    validate_fencing_lease_in_context,
)
from director.core.generation_lifecycle import (
    GenerationOperation,
    create_operation,
    make_generation_run_id,
)
from director.core.generation_provenance import (
    GenerationProvenanceManifest,
    create_provenance_manifest,
    redact_generation_config,
    stable_digest,
    sync_provenance_manifest,
)
from director.core.session import ContextMessage, RoleTypes


CHECKPOINT_CONTEXT_KEY = "__text_to_movie_checkpoints__"
CHECKPOINT_VERSION = 6
ACTIVE_AGENT_CALL_STATE_KEY = "__active_agent_call__"


class CheckpointConflictError(RuntimeError):
    """Raised when a stale checkpoint attempts to overwrite newer progress."""


class CheckpointFenceError(CheckpointConflictError):
    """Raised when a checkpoint writer no longer owns the current fence."""

    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


class TextToMovieExecutionError(RuntimeError):
    """Safe typed failure for resumable Text-to-Movie execution."""

    def __init__(
        self,
        *,
        stage: str,
        code: str,
        message: str,
        scene_index: Optional[int] = None,
        resumable: bool = True,
        operation_id: Optional[str] = None,
        reconciliation: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.scene_index = scene_index
        self.resumable = resumable
        self.operation_id = operation_id
        self.reconciliation = reconciliation or None


class SceneCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    plan: Dict[str, Any]
    prompt: Optional[str] = None
    media: Optional[Dict[str, Any]] = None
    status: str = "pending"
    operation: Optional[GenerationOperation] = None


class TextToMovieCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = CHECKPOINT_VERSION
    revision: int = Field(default=0, ge=0)
    checkpoint_id: str
    request_fingerprint: str
    generation_run_id: Optional[str] = None
    status: str = "planned"
    visual_style: Dict[str, Any]
    scenes: List[SceneCheckpoint]
    audio_prompt: Optional[str] = None
    audio_media: Optional[Dict[str, Any]] = None
    audio_operation: Optional[GenerationOperation] = None
    final_video: Optional[str] = None
    failure_stage: Optional[str] = None
    failure_code: Optional[str] = None
    failed_scene_index: Optional[int] = None
    provenance: Optional[GenerationProvenanceManifest] = None

    @property
    def completed_scene_count(self) -> int:
        return sum(
            1
            for scene in self.scenes
            if scene.status == "complete" and scene.media
        )

    def ensure_lifecycle(self, *, video_provider: str, audio_provider: str) -> bool:
        changed = False
        if not self.generation_run_id:
            self.generation_run_id = make_generation_run_id(self.request_fingerprint)
            changed = True

        for scene in self.scenes:
            if scene.operation is None:
                scene.operation = create_operation(
                    self.generation_run_id,
                    kind="scene_video",
                    provider=video_provider,
                    index=scene.index,
                )
                if scene.media:
                    scene.operation.artifact = dict(scene.media)
                    scene.operation.state = "persisted"
                    scene.operation.recoverable = False
                changed = True

        if self.audio_operation is None:
            self.audio_operation = create_operation(
                self.generation_run_id,
                kind="background_audio",
                provider=audio_provider,
            )
            if self.audio_media:
                self.audio_operation.artifact = dict(self.audio_media)
                self.audio_operation.state = "persisted"
                self.audio_operation.recoverable = False
            changed = True

        if self.provenance is None:
            self.provenance = create_provenance_manifest(
                generation_run_id=self.generation_run_id,
                checkpoint_id=self.checkpoint_id,
                request_fingerprint=self.request_fingerprint,
                video_provider=video_provider,
                audio_provider=audio_provider,
            )
            changed = True

        if self.version != CHECKPOINT_VERSION:
            self.version = CHECKPOINT_VERSION
            changed = True
        return changed

    def unresolved_provider_operations(self) -> List[GenerationOperation]:
        operations = [
            scene.operation
            for scene in self.scenes
            if scene.operation is not None
            and scene.operation.has_provider_resume_token
        ]
        if (
            self.audio_operation is not None
            and self.audio_operation.has_provider_resume_token
        ):
            operations.append(self.audio_operation)
        return operations


def build_request_fingerprint(
    *,
    collection_id: str,
    engine: str,
    audio_engine: str,
    storyline: str,
    video_config: Optional[dict] = None,
    audio_config: Optional[dict] = None,
) -> str:
    payload = {
        "collection_id": collection_id,
        "engine": engine,
        "audio_engine": audio_engine,
        "storyline": storyline,
        "video_config": video_config or {},
        "audio_config": audio_config or {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_checkpoint_id(request_fingerprint: str) -> str:
    return f"text_to_movie:{request_fingerprint[:24]}"


def create_checkpoint(
    *,
    request_fingerprint: str,
    visual_style: Dict[str, Any],
    scenes: List[Dict[str, Any]],
    video_provider: Optional[str] = None,
    audio_provider: Optional[str] = None,
    collection_id: Optional[str] = None,
    storyline: Optional[str] = None,
    video_config: Optional[Dict[str, Any]] = None,
    audio_config: Optional[Dict[str, Any]] = None,
) -> TextToMovieCheckpoint:
    checkpoint = TextToMovieCheckpoint(
        checkpoint_id=make_checkpoint_id(request_fingerprint),
        request_fingerprint=request_fingerprint,
        generation_run_id=make_generation_run_id(request_fingerprint),
        visual_style=visual_style,
        scenes=[
            SceneCheckpoint(index=index, plan=scene)
            for index, scene in enumerate(scenes)
        ],
    )
    if video_provider and audio_provider:
        checkpoint.ensure_lifecycle(
            video_provider=video_provider,
            audio_provider=audio_provider,
        )
        checkpoint.provenance = create_provenance_manifest(
            generation_run_id=checkpoint.generation_run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            request_fingerprint=request_fingerprint,
            collection_id=collection_id,
            storyline=storyline,
            video_provider=video_provider,
            audio_provider=audio_provider,
            video_config=video_config,
            audio_config=audio_config,
        )
        sync_provenance_manifest(checkpoint.provenance, checkpoint)
    return checkpoint


def compact_media(media: Any) -> Dict[str, Any]:
    if not isinstance(media, dict):
        raise ValueError("media result must be an object")
    media_id = media.get("id")
    if not media_id:
        raise ValueError("media result is missing id")

    compact = {"id": media_id}
    if media.get("length") is not None:
        compact["length"] = media.get("length")
    if media.get("collection_id") is not None:
        compact["collection_id"] = media.get("collection_id")
    return compact


class TextToMovieCheckpointStore:
    """CAS-backed checkpoint store inside the existing session context JSON."""

    def __init__(self, session, *, max_cas_attempts: int = 8):
        self.session = session
        self.max_cas_attempts = max_cas_attempts

    @staticmethod
    def _read_document_from_context(context: Dict) -> Dict[str, dict]:
        messages = context.get(CHECKPOINT_CONTEXT_KEY, [])
        if not messages:
            return {}
        content = (
            messages[-1].get("content")
            if isinstance(messages[-1], dict)
            else None
        )
        if not isinstance(content, str) or not content:
            return {}
        try:
            document = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return {}
        return document if isinstance(document, dict) else {}

    def _read_document(self) -> Dict[str, dict]:
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        return self._read_document_from_context(context)

    def _authoritative_now(self) -> int:
        current_epoch = getattr(self.session.db, "current_epoch", None)
        if callable(current_epoch):
            try:
                return int(current_epoch())
            except (AttributeError, NotImplementedError):
                pass
        return int(time.time())

    def get(self, checkpoint_id: str) -> Optional[TextToMovieCheckpoint]:
        raw = self._read_document().get(checkpoint_id)
        if raw is None:
            return None
        try:
            return TextToMovieCheckpoint.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _infer_provider(checkpoint: TextToMovieCheckpoint, media_type: str) -> Optional[str]:
        if media_type == "video":
            for scene in checkpoint.scenes:
                if scene.operation is not None and scene.operation.provider:
                    return scene.operation.provider
            return None
        if checkpoint.audio_operation is not None:
            return checkpoint.audio_operation.provider
        return None

    @staticmethod
    def _raw_submission_count(operation: Optional[Dict[str, Any]]) -> int:
        if not isinstance(operation, dict):
            return 0
        value = operation.get("submission_count")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

    @classmethod
    def _has_new_submission(
        cls,
        raw_checkpoint: Optional[Dict[str, Any]],
        checkpoint: TextToMovieCheckpoint,
    ) -> bool:
        raw_checkpoint = raw_checkpoint if isinstance(raw_checkpoint, dict) else {}
        old_scenes = {
            scene.get("index"): scene
            for scene in raw_checkpoint.get("scenes", []) or []
            if isinstance(scene, dict)
        }
        for scene in checkpoint.scenes:
            operation = scene.operation
            if operation is None or operation.submission_count is None:
                continue
            old_scene = old_scenes.get(scene.index) or {}
            old_count = cls._raw_submission_count(old_scene.get("operation"))
            if int(operation.submission_count) > old_count:
                return True

        operation = checkpoint.audio_operation
        if operation is not None and operation.submission_count is not None:
            old_count = cls._raw_submission_count(raw_checkpoint.get("audio_operation"))
            if int(operation.submission_count) > old_count:
                return True
        return False

    @staticmethod
    def _restore_checkpoint(
        checkpoint: TextToMovieCheckpoint,
        raw_checkpoint: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(raw_checkpoint, dict):
            return
        try:
            restored = TextToMovieCheckpoint.model_validate(raw_checkpoint)
        except Exception:
            return
        for field_name in TextToMovieCheckpoint.model_fields:
            setattr(checkpoint, field_name, getattr(restored, field_name))

    def _active_text_to_movie_arguments(self) -> Optional[Dict[str, Any]]:
        state = getattr(self.session, "state", None)
        if not isinstance(state, dict):
            return None
        active = state.get(ACTIVE_AGENT_CALL_STATE_KEY)
        if not isinstance(active, dict) or active.get("agent_name") != "text_to_movie":
            return None
        arguments = active.get("arguments")
        return arguments if isinstance(arguments, dict) else None

    def _enrich_request_provenance(self, checkpoint: TextToMovieCheckpoint) -> None:
        if checkpoint.provenance is None:
            return
        arguments = self._active_text_to_movie_arguments()
        if not arguments:
            return

        request = checkpoint.provenance.request
        payload = arguments.get("text_to_movie")
        payload = payload if isinstance(payload, dict) else {}
        engine = arguments.get("engine")
        audio_engine = arguments.get("audio_engine", "videodb")
        storyline = payload.get("storyline")

        if request.collection_id is None and arguments.get("collection_id"):
            request.collection_id = str(arguments.get("collection_id"))
        if request.storyline is None and isinstance(storyline, str) and storyline.strip():
            request.storyline = storyline.strip()
            request.storyline_digest = stable_digest(request.storyline)
        if request.video_provider is None and engine:
            request.video_provider = str(engine)
        if request.audio_provider is None and audio_engine:
            request.audio_provider = str(audio_engine)

        if not request.video_config:
            video_config_key = (
                "video_stabilityai_config"
                if engine == "stabilityai"
                else "video_kling_config"
                if engine == "kling"
                else None
            )
            if video_config_key:
                config = payload.get(video_config_key)
                if isinstance(config, dict):
                    request.video_config = redact_generation_config(config)
        if not request.audio_config and audio_engine == "elevenlabs":
            config = payload.get("audio_elevenlabs_config")
            if isinstance(config, dict):
                request.audio_config = redact_generation_config(config)

    def _sync_provenance(self, checkpoint: TextToMovieCheckpoint) -> None:
        if not checkpoint.generation_run_id:
            checkpoint.generation_run_id = make_generation_run_id(
                checkpoint.request_fingerprint
            )
        if checkpoint.provenance is None:
            checkpoint.provenance = create_provenance_manifest(
                generation_run_id=checkpoint.generation_run_id,
                checkpoint_id=checkpoint.checkpoint_id,
                request_fingerprint=checkpoint.request_fingerprint,
                video_provider=self._infer_provider(checkpoint, "video"),
                audio_provider=self._infer_provider(checkpoint, "audio"),
            )
        self._enrich_request_provenance(checkpoint)
        sync_provenance_manifest(checkpoint.provenance, checkpoint)

    def save(
        self,
        checkpoint: TextToMovieCheckpoint,
        *,
        fencing_lease: Optional[GenerationLease] = None,
    ) -> None:
        self._sync_provenance(checkpoint)
        effective_lease = fencing_lease or get_active_fencing_lease(self.session)
        expected_revision = checkpoint.revision
        next_revision = expected_revision + 1
        stored = checkpoint.model_copy(deep=True)
        stored.revision = next_revision
        stored.version = CHECKPOINT_VERSION
        message_holder = {}
        conflict_reason = None
        fence_reason = None
        budget_error = None
        budget_restore_raw = None

        def mutate(context: Dict) -> Optional[Dict]:
            nonlocal conflict_reason, fence_reason, budget_error, budget_restore_raw
            if effective_lease is not None:
                fence_reason = validate_fencing_lease_in_context(
                    context,
                    effective_lease,
                    now_epoch=self._authoritative_now(),
                )
                if fence_reason is not None:
                    return None

            document = self._read_document_from_context(context)
            raw_current = document.get(checkpoint.checkpoint_id)

            if raw_current is None:
                if expected_revision != 0:
                    conflict_reason = "checkpoint_missing_after_prior_write"
                    return None
            else:
                current_revision = int(raw_current.get("revision", 0))
                if current_revision != expected_revision:
                    conflict_reason = "checkpoint_revision_conflict"
                    return None

            if self._has_new_submission(raw_current, checkpoint):
                try:
                    policy, rate_card = load_budget_configuration_from_env()
                    if policy is not None:
                        authorize_checkpoint_budget_transition(
                            context=context,
                            previous_checkpoint=raw_current,
                            checkpoint=checkpoint,
                            policy=policy,
                            rate_card=rate_card,
                            now_epoch=self._authoritative_now(),
                        )
                except (BudgetConfigurationError, BudgetGuardViolation) as exc:
                    budget_error = exc
                    budget_restore_raw = raw_current
                    return None

            document[checkpoint.checkpoint_id] = stored.model_dump(mode="json")
            content = json.dumps(
                document,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            message = ContextMessage(content=content, role=RoleTypes.system)
            message_holder["message"] = message
            context[CHECKPOINT_CONTEXT_KEY] = [message.to_llm_msg()]
            return context

        atomic_update_context(
            self.session.db,
            self.session.session_id,
            mutate,
            max_attempts=self.max_cas_attempts,
        )

        if fence_reason:
            raise CheckpointFenceError(fence_reason)
        if conflict_reason:
            raise CheckpointConflictError(conflict_reason)
        if budget_error is not None:
            self._restore_checkpoint(checkpoint, budget_restore_raw)
            if isinstance(budget_error, BudgetGuardViolation):
                code = budget_error.code
                operation_id = budget_error.operation_id
                scene_index = budget_error.scene_index
            else:
                code = str(budget_error)
                operation_id = None
                scene_index = None
            raise TextToMovieExecutionError(
                stage="budget",
                code=code,
                message="Generation budget guard blocked a new provider submission.",
                scene_index=scene_index,
                resumable=True,
                operation_id=operation_id,
            )

        checkpoint.revision = next_revision
        checkpoint.version = CHECKPOINT_VERSION
        message = message_holder.get("message")
        if message is not None:
            self.session.agent_context[CHECKPOINT_CONTEXT_KEY] = [message]
