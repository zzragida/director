import pytest

from director.core.tool_semantics import (
    ToolSemanticValidationError,
    validate_tool_semantics,
)


class FakeVideo:
    def __init__(self, video_id="video-1", collection_id="collection-1"):
        self.id = video_id
        self.collection_id = collection_id


class FakeCollection:
    def __init__(self, collection_id="collection-1", videos=None, error=None):
        self.id = collection_id
        self.videos = videos or {}
        self.error = error
        self.get_video_calls = 0

    def get_video(self, video_id):
        self.get_video_calls += 1
        if self.error is not None:
            raise self.error
        return self.videos.get(video_id)


class FakeSession:
    def __init__(
        self,
        *,
        collection_id="collection-1",
        collection=None,
        video_id=None,
        video=None,
    ):
        self.collection_id = collection_id
        self.video_id = video_id
        self.state = {}
        if collection is not None:
            self.state["collection"] = collection
        if video is not None:
            self.state["video"] = video


def assert_semantic_error(error, code, field):
    assert error.value.code == "invalid_tool_semantics"
    assert error.value.details[0]["code"] == code
    assert error.value.details[0]["field"] == field


def test_collection_mismatch_is_rejected_without_provider_lookup():
    collection = FakeCollection()
    session = FakeSession(collection=collection)

    with pytest.raises(ToolSemanticValidationError) as error:
        validate_tool_semantics(
            session,
            "summarize_video",
            {"collection_id": "other-collection"},
        )

    assert_semantic_error(error, "collection_context_mismatch", "collection_id")
    assert collection.get_video_calls == 0
    assert "other-collection" not in str(error.value.details)


def test_matching_collection_without_video_is_allowed():
    collection = FakeCollection()
    session = FakeSession(collection=collection)

    arguments = {"collection_id": "collection-1", "prompt": "summarize"}
    assert validate_tool_semantics(session, "summarize_video", arguments) is arguments
    assert collection.get_video_calls == 0


def test_pre_resolved_session_video_is_reused_without_lookup():
    video = FakeVideo(video_id="video-1")
    collection = FakeCollection(videos={"video-1": video})
    session = FakeSession(
        collection=collection,
        video_id="video-1",
        video=video,
    )

    arguments = {"collection_id": "collection-1", "video_id": "video-1"}
    assert validate_tool_semantics(session, "summarize_video", arguments) is arguments
    assert collection.get_video_calls == 0


def test_new_video_is_resolved_once_and_cached():
    video = FakeVideo(video_id="video-2")
    collection = FakeCollection(videos={"video-2": video})
    session = FakeSession(collection=collection)
    arguments = {"collection_id": "collection-1", "video_id": "video-2"}

    validate_tool_semantics(session, "index", arguments)
    validate_tool_semantics(session, "index", arguments)

    assert collection.get_video_calls == 1
    assert session.state["resolved_videos"]["video-2"] is video


def test_missing_video_is_reported_as_not_found_without_echoing_id():
    collection = FakeCollection()
    session = FakeSession(collection=collection)

    with pytest.raises(ToolSemanticValidationError) as error:
        validate_tool_semantics(
            session,
            "summarize_video",
            {"collection_id": "collection-1", "video_id": "secret-video-id"},
        )

    assert_semantic_error(error, "video_not_found", "video_id")
    assert "secret-video-id" not in str(error.value.details)


def test_provider_exception_is_not_guessed_as_not_found():
    collection = FakeCollection(error=RuntimeError("provider detail must stay private"))
    session = FakeSession(collection=collection)

    with pytest.raises(ToolSemanticValidationError) as error:
        validate_tool_semantics(
            session,
            "summarize_video",
            {"collection_id": "collection-1", "video_id": "video-2"},
        )

    assert_semantic_error(error, "video_lookup_failed", "video_id")
    assert "provider detail" not in str(error.value.details)


def test_video_from_different_collection_is_rejected():
    video = FakeVideo(video_id="video-2", collection_id="collection-2")
    collection = FakeCollection(videos={"video-2": video})
    session = FakeSession(collection=collection)

    with pytest.raises(ToolSemanticValidationError) as error:
        validate_tool_semantics(
            session,
            "index",
            {"collection_id": "collection-1", "video_id": "video-2"},
        )

    assert_semantic_error(error, "video_collection_mismatch", "video_id")


def test_video_reference_requires_session_media_context():
    session = FakeSession(collection_id=None, collection=None)

    with pytest.raises(ToolSemanticValidationError) as error:
        validate_tool_semantics(session, "index", {"video_id": "video-2"})

    assert_semantic_error(error, "semantic_context_unavailable", "video_id")
