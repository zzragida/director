from typing import Any, Dict, List, Optional


class ToolSemanticValidationError(ValueError):
    """Raised when structurally valid tool arguments violate session semantics."""

    code = "invalid_tool_semantics"

    def __init__(self, details: List[dict]):
        super().__init__("Invalid agent tool semantics")
        self.details = details


def _current_collection(session: Any) -> Any:
    state = getattr(session, "state", {}) or {}
    if isinstance(state, dict):
        return state.get("collection")
    return None


def _current_collection_id(session: Any, collection: Any) -> Optional[str]:
    session_collection_id = getattr(session, "collection_id", None)
    if session_collection_id is not None:
        return str(session_collection_id)

    collection_id = getattr(collection, "id", None)
    if collection_id is not None:
        return str(collection_id)
    return None


def _detail(field: str, code: str, message: str) -> dict:
    return {"field": field, "code": code, "message": message}


def _resolve_video_from_session(session: Any, video_id: str) -> Any:
    state = getattr(session, "state", {}) or {}
    if not isinstance(state, dict):
        return None

    session_video_id = getattr(session, "video_id", None)
    if session_video_id is not None and str(session_video_id) == str(video_id):
        existing_video = state.get("video")
        if existing_video is not None:
            return existing_video

    cache = state.setdefault("resolved_videos", {})
    if str(video_id) in cache:
        return cache[str(video_id)]
    return None


def _cache_video(session: Any, video_id: str, video: Any) -> None:
    state = getattr(session, "state", {}) or {}
    if isinstance(state, dict):
        state.setdefault("resolved_videos", {})[str(video_id)] = video


def validate_tool_semantics(
    session: Any,
    agent_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate tool arguments against the authoritative chat media context.

    This layer intentionally runs after structural tool validation. It currently
    enforces only invariants supported by the existing Director session model:

    - an explicit tool `collection_id` must match the session collection;
    - an explicit top-level `video_id` must resolve from that collection;
    - a resolved video that exposes `collection_id` must belong to that collection.

    Error details never echo the rejected identifiers or provider exception text.
    """

    if not isinstance(arguments, dict):
        return arguments

    collection = _current_collection(session)
    current_collection_id = _current_collection_id(session, collection)

    requested_collection_id = arguments.get("collection_id")
    if requested_collection_id is not None:
        if current_collection_id is None:
            raise ToolSemanticValidationError(
                [
                    _detail(
                        "collection_id",
                        "semantic_context_unavailable",
                        "session collection context is unavailable",
                    )
                ]
            )
        if str(requested_collection_id) != current_collection_id:
            raise ToolSemanticValidationError(
                [
                    _detail(
                        "collection_id",
                        "collection_context_mismatch",
                        "tool collection does not match the current session collection",
                    )
                ]
            )

    video_id = arguments.get("video_id")
    if video_id is None:
        return arguments

    if collection is None or current_collection_id is None:
        raise ToolSemanticValidationError(
            [
                _detail(
                    "video_id",
                    "semantic_context_unavailable",
                    "session media context is unavailable",
                )
            ]
        )

    video = _resolve_video_from_session(session, str(video_id))
    if video is None:
        try:
            video = collection.get_video(video_id)
        except Exception as exc:
            raise ToolSemanticValidationError(
                [
                    _detail(
                        "video_id",
                        "video_lookup_failed",
                        "unable to resolve the requested video",
                    )
                ]
            ) from exc

        if video is None:
            raise ToolSemanticValidationError(
                [
                    _detail(
                        "video_id",
                        "video_not_found",
                        "requested video was not found in the current collection",
                    )
                ]
            )

        _cache_video(session, str(video_id), video)

    actual_collection_id = getattr(video, "collection_id", None)
    if (
        actual_collection_id is not None
        and str(actual_collection_id) != current_collection_id
    ):
        raise ToolSemanticValidationError(
            [
                _detail(
                    "video_id",
                    "video_collection_mismatch",
                    "requested video does not belong to the current session collection",
                )
            ]
        )

    return arguments
