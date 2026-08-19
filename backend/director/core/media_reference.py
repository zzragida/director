from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class MediaReferenceError(Exception):
    """Safe domain error raised while resolving VideoDB references."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def resolve_media_reference(
    connect_fn: Callable[..., Any],
    *,
    base_url: str,
    collection_id: Optional[str],
    video_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve collection/video references without guessing provider error semantics.

    `None` results are treated as not-found. Provider exceptions are mapped to
    lookup/service failures because the pinned SDK does not expose a repository-
    level error contract here that safely distinguishes all failure causes.
    """

    if not collection_id:
        raise MediaReferenceError(
            code="collection_required",
            message="A collection_id is required to process this chat request.",
        )

    try:
        conn = connect_fn(base_url=base_url)
    except Exception as exc:
        raise MediaReferenceError(
            code="media_service_unavailable",
            message="Unable to connect to the media service.",
        ) from exc

    try:
        collection = conn.get_collection(collection_id)
    except Exception as exc:
        raise MediaReferenceError(
            code="collection_lookup_failed",
            message="Unable to resolve the requested collection.",
        ) from exc

    if collection is None:
        raise MediaReferenceError(
            code="collection_not_found",
            message="The requested collection was not found.",
        )

    state: Dict[str, Any] = {
        "conn": conn,
        "collection": collection,
    }

    if not video_id:
        return state

    try:
        video = collection.get_video(video_id)
    except Exception as exc:
        raise MediaReferenceError(
            code="video_lookup_failed",
            message="Unable to resolve the requested video.",
        ) from exc

    if video is None:
        raise MediaReferenceError(
            code="video_not_found",
            message="The requested video was not found in the collection.",
        )

    expected_collection_id = getattr(collection, "id", collection_id)
    actual_collection_id = getattr(video, "collection_id", None)
    if (
        actual_collection_id is not None
        and str(actual_collection_id) != str(expected_collection_id)
    ):
        raise MediaReferenceError(
            code="video_collection_mismatch",
            message="The requested video does not belong to the requested collection.",
        )

    state["video"] = video
    return state
