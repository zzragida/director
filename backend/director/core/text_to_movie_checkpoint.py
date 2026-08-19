import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.generation_lifecycle import (
    GenerationOperation,
    create_operation,
    make_generation_run_id,
)
from director.core.session import ContextMessage, RoleTypes


CHECKPOINT_CONTEXT_KEY = "__text_to_movie_checkpoints__"
CHECKPOINT_VERSION = 3


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

    @property
    def completed_scene_count(self) -> int:
        return sum(
            1
            for scene in self.scenes
            if scene.status == "complete" and scene.media
        )

    def ensure_lifecycle(self, *, video_provider: str, audio_provider: str) -> bool:
        """Backfill deterministic operation IDs for new or legacy checkpoints."""

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
    """Return a deterministic fingerprint for resumable Text-to-Movie work."""

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
) -> TextToMovieCheckpoint:
    checkpoint_id = make_checkpoint_id(request_fingerprint)
    generation_run_id = make_generation_run_id(request_fingerprint)
    checkpoint = TextToMovieCheckpoint(
        checkpoint_id=checkpoint_id,
        request_fingerprint=request_fingerprint,
        generation_run_id=generation_run_id,
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
    return checkpoint


def compact_media(media: Any) -> Dict[str, Any]:
    """Persist only stable media fields needed for resume/composition."""

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
    """Persist resumable checkpoints inside the existing session context JSON.

    The database context row is merged in place so a checkpoint written after a
    scene completes does not wait for the final reasoning save. A mirrored
    ContextMessage is also placed in ``session.agent_context`` so the normal
    Session.save_context_messages() path keeps the checkpoint on later saves.
    """

    def __init__(self, session):
        self.session = session

    def _read_document(self) -> Dict[str, dict]:
        context = self.session.db.get_context_messages(self.session.session_id) or {}
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

    def get(self, checkpoint_id: str) -> Optional[TextToMovieCheckpoint]:
        raw = self._read_document().get(checkpoint_id)
        if raw is None:
            return None
        try:
            return TextToMovieCheckpoint.model_validate(raw)
        except Exception:
            return None

    def save(self, checkpoint: TextToMovieCheckpoint) -> None:
        document = self._read_document()
        document[checkpoint.checkpoint_id] = checkpoint.model_dump(mode="json")
        content = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        message = ContextMessage(content=content, role=RoleTypes.system)

        self.session.agent_context[CHECKPOINT_CONTEXT_KEY] = [message]

        context = self.session.db.get_context_messages(self.session.session_id) or {}
        context[CHECKPOINT_CONTEXT_KEY] = [message.to_llm_msg()]
        self.session.db.add_or_update_context_msg(self.session.session_id, context)
