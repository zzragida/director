import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from director.core.session import ContextMessage, RoleTypes


CHECKPOINT_CONTEXT_KEY = "__text_to_movie_checkpoints__"
CHECKPOINT_VERSION = 1


class SceneCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    plan: Dict[str, Any]
    prompt: Optional[str] = None
    media: Optional[Dict[str, Any]] = None
    status: str = "pending"


class TextToMovieCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = CHECKPOINT_VERSION
    checkpoint_id: str
    request_fingerprint: str
    status: str = "planned"
    visual_style: Dict[str, Any]
    scenes: List[SceneCheckpoint]
    audio_prompt: Optional[str] = None
    audio_media: Optional[Dict[str, Any]] = None
    final_video: Optional[str] = None
    failure_stage: Optional[str] = None
    failure_code: Optional[str] = None
    failed_scene_index: Optional[int] = None

    @property
    def completed_scene_count(self) -> int:
        return sum(1 for scene in self.scenes if scene.status == "complete" and scene.media)


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
) -> TextToMovieCheckpoint:
    checkpoint_id = make_checkpoint_id(request_fingerprint)
    return TextToMovieCheckpoint(
        checkpoint_id=checkpoint_id,
        request_fingerprint=request_fingerprint,
        visual_style=visual_style,
        scenes=[
            SceneCheckpoint(index=index, plan=scene)
            for index, scene in enumerate(scenes)
        ],
    )


def compact_media(media: Any) -> Dict[str, Any]:
    """Persist only the stable media fields needed for resume/composition."""

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

        content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
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

        # Keep the normal Session save path aware of the checkpoint.
        self.session.agent_context[CHECKPOINT_CONTEXT_KEY] = [message]

        # Persist immediately so completed external work survives a later crash.
        context = self.session.db.get_context_messages(self.session.session_id) or {}
        context[CHECKPOINT_CONTEXT_KEY] = [message.to_llm_msg()]
        self.session.db.add_or_update_context_msg(self.session.session_id, context)
