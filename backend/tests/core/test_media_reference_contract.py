import pytest

from director.core.media_reference import MediaReferenceError, resolve_media_reference


class FakeVideo:
    def __init__(self, video_id="video-1", collection_id="collection-1"):
        self.id = video_id
        self.collection_id = collection_id


class FakeCollection:
    def __init__(
        self,
        collection_id="collection-1",
        *,
        video=None,
        video_error=None,
    ):
        self.id = collection_id
        self.video = video
        self.video_error = video_error

    def get_video(self, video_id):
        if self.video_error:
            raise self.video_error
        return self.video


class FakeConnection:
    def __init__(self, *, collection=None, collection_error=None):
        self.collection = collection
        self.collection_error = collection_error

    def get_collection(self, collection_id):
        if self.collection_error:
            raise self.collection_error
        return self.collection


def make_connect(connection=None, error=None):
    def connect(*, base_url):
        if error:
            raise error
        return connection

    return connect


def assert_media_error(code, call):
    with pytest.raises(MediaReferenceError) as exc_info:
        call()
    assert exc_info.value.code == code
    return exc_info.value


def test_collection_id_is_required_for_chat_media_context():
    assert_media_error(
        "collection_required",
        lambda: resolve_media_reference(
            make_connect(), base_url="https://media", collection_id=None
        ),
    )


def test_connection_failure_is_not_reported_as_not_found():
    error = assert_media_error(
        "media_service_unavailable",
        lambda: resolve_media_reference(
            make_connect(error=RuntimeError("network down")),
            base_url="https://media",
            collection_id="collection-1",
        ),
    )
    assert "network down" not in error.message


def test_collection_none_is_reported_as_not_found():
    assert_media_error(
        "collection_not_found",
        lambda: resolve_media_reference(
            make_connect(FakeConnection(collection=None)),
            base_url="https://media",
            collection_id="collection-1",
        ),
    )


def test_collection_exception_is_reported_as_lookup_failure():
    assert_media_error(
        "collection_lookup_failed",
        lambda: resolve_media_reference(
            make_connect(
                FakeConnection(collection_error=RuntimeError("provider detail"))
            ),
            base_url="https://media",
            collection_id="collection-1",
        ),
    )


def test_valid_collection_without_video_returns_collection_state():
    collection = FakeCollection()
    connection = FakeConnection(collection=collection)

    state = resolve_media_reference(
        make_connect(connection),
        base_url="https://media",
        collection_id="collection-1",
    )

    assert state["conn"] is connection
    assert state["collection"] is collection
    assert "video" not in state


def test_video_none_is_reported_as_not_found():
    collection = FakeCollection(video=None)
    assert_media_error(
        "video_not_found",
        lambda: resolve_media_reference(
            make_connect(FakeConnection(collection=collection)),
            base_url="https://media",
            collection_id="collection-1",
            video_id="video-1",
        ),
    )


def test_video_exception_is_reported_as_lookup_failure():
    collection = FakeCollection(video_error=RuntimeError("provider detail"))
    assert_media_error(
        "video_lookup_failed",
        lambda: resolve_media_reference(
            make_connect(FakeConnection(collection=collection)),
            base_url="https://media",
            collection_id="collection-1",
            video_id="video-1",
        ),
    )


def test_video_collection_mismatch_is_rejected():
    collection = FakeCollection(
        collection_id="collection-1",
        video=FakeVideo(collection_id="collection-2"),
    )
    assert_media_error(
        "video_collection_mismatch",
        lambda: resolve_media_reference(
            make_connect(FakeConnection(collection=collection)),
            base_url="https://media",
            collection_id="collection-1",
            video_id="video-1",
        ),
    )


def test_valid_video_is_added_to_resolved_state():
    video = FakeVideo()
    collection = FakeCollection(video=video)
    state = resolve_media_reference(
        make_connect(FakeConnection(collection=collection)),
        base_url="https://media",
        collection_id="collection-1",
        video_id="video-1",
    )

    assert state["collection"] is collection
    assert state["video"] is video
