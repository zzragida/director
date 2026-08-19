import importlib.util
import sys
import types
from pathlib import Path


class DummyNamespace:
    def __init__(self, *args, **kwargs):
        pass


class MediaReferenceError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def load_socket_module(monkeypatch, calls, media_error=None):
    flask_module = types.ModuleType("flask")
    flask_module.current_app = types.SimpleNamespace(config={"DB_TYPE": "sqlite"})

    socketio_module = types.ModuleType("flask_socketio")
    socketio_module.Namespace = DummyNamespace

    db_module = types.ModuleType("director.db")

    def load_db(db_type):
        calls["load_db"] += 1
        calls["db_type"] = db_type
        return "fake-db"

    db_module.load_db = load_db

    handler_module = types.ModuleType("director.handler")

    class FakeChatHandler:
        def __init__(self, db):
            calls["handler_init"] += 1
            calls["db"] = db

        def chat(self, message, media_state=None):
            calls["chat"] += 1
            calls["message"] = message
            calls["media_state"] = media_state
            return {"status": "handled"}

    handler_module.ChatHandler = FakeChatHandler

    media_module = types.ModuleType("director.core.media_reference")
    media_module.MediaReferenceError = MediaReferenceError

    def resolve_media_reference(
        connect_fn, *, base_url, collection_id, video_id=None
    ):
        calls["media_resolve"] += 1
        calls["collection_id"] = collection_id
        calls["video_id"] = video_id
        calls["base_url"] = base_url
        if media_error:
            raise media_error
        return {
            "conn": "fake-conn",
            "collection": "fake-collection",
            **({"video": "fake-video"} if video_id else {}),
        }

    media_module.resolve_media_reference = resolve_media_reference

    videodb_module = types.ModuleType("videodb")
    videodb_module.connect = lambda **kwargs: "unused-by-fake-resolver"

    monkeypatch.setitem(sys.modules, "flask", flask_module)
    monkeypatch.setitem(sys.modules, "flask_socketio", socketio_module)
    monkeypatch.setitem(sys.modules, "director.db", db_module)
    monkeypatch.setitem(sys.modules, "director.handler", handler_module)
    monkeypatch.setitem(sys.modules, "director.core.media_reference", media_module)
    monkeypatch.setitem(sys.modules, "videodb", videodb_module)

    socket_path = (
        Path(__file__).resolve().parents[2]
        / "director"
        / "entrypoint"
        / "api"
        / "socket_io.py"
    )
    module_name = "director_chat_socket_contract_test"
    spec = importlib.util.spec_from_file_location(module_name, socket_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_calls():
    return {
        "load_db": 0,
        "handler_init": 0,
        "chat": 0,
        "media_resolve": 0,
        "db_type": None,
        "db": None,
        "message": None,
        "media_state": None,
        "collection_id": None,
        "video_id": None,
        "base_url": None,
    }


def test_invalid_request_is_rejected_before_media_db_and_handler(monkeypatch):
    calls = make_calls()
    socket_module = load_socket_module(monkeypatch, calls)
    namespace = socket_module.ChatNamespace("/chat")

    response = namespace.on_chat(
        {"session_id": " ", "conv_id": "conv-1", "content": "hello"}
    )

    assert response["status"] == "error"
    assert response["error"] == "invalid_chat_request"
    assert calls["media_resolve"] == 0
    assert calls["load_db"] == 0
    assert calls["handler_init"] == 0
    assert calls["chat"] == 0


def test_missing_collection_is_rejected_before_db_and_handler(monkeypatch):
    calls = make_calls()
    media_error = MediaReferenceError(
        "collection_required",
        "A collection_id is required to process this chat request.",
    )
    socket_module = load_socket_module(monkeypatch, calls, media_error=media_error)
    namespace = socket_module.ChatNamespace("/chat")

    response = namespace.on_chat(
        {
            "session_id": "session-1",
            "conv_id": "conv-1",
            "content": "hello",
        }
    )

    assert response == {
        "status": "error",
        "error": "invalid_media_reference",
        "code": "collection_required",
        "message": "A collection_id is required to process this chat request.",
    }
    assert calls["media_resolve"] == 1
    assert calls["load_db"] == 0
    assert calls["handler_init"] == 0
    assert calls["chat"] == 0


def test_media_lookup_error_does_not_echo_provider_details(monkeypatch):
    calls = make_calls()
    media_error = MediaReferenceError(
        "video_lookup_failed",
        "Unable to resolve the requested video.",
    )
    socket_module = load_socket_module(monkeypatch, calls, media_error=media_error)
    namespace = socket_module.ChatNamespace("/chat")

    response = namespace.on_chat(
        {
            "session_id": "session-1",
            "conv_id": "conv-1",
            "content": "hello",
            "collection_id": "collection-1",
            "video_id": "video-secret-reference",
        }
    )

    assert response["error"] == "invalid_media_reference"
    assert response["code"] == "video_lookup_failed"
    assert "video-secret-reference" not in response["message"]
    assert calls["load_db"] == 0
    assert calls["handler_init"] == 0


def test_valid_request_resolves_media_before_handler(monkeypatch):
    calls = make_calls()
    socket_module = load_socket_module(monkeypatch, calls)
    namespace = socket_module.ChatNamespace("/chat")

    response = namespace.on_chat(
        {
            "session_id": " session-1 ",
            "conv_id": " conv-1 ",
            "content": "hello",
            "collection_id": " collection-1 ",
            "video_id": " video-1 ",
            "agents": ["summarize_video"],
            "client_trace_id": "trace-1",
        }
    )

    assert response == {"status": "handled"}
    assert calls["media_resolve"] == 1
    assert calls["collection_id"] == "collection-1"
    assert calls["video_id"] == "video-1"
    assert calls["load_db"] == 1
    assert calls["handler_init"] == 1
    assert calls["chat"] == 1
    assert calls["db_type"] == "sqlite"
    assert calls["message"]["session_id"] == "session-1"
    assert calls["message"]["conv_id"] == "conv-1"
    assert calls["message"]["collection_id"] == "collection-1"
    assert calls["message"]["video_id"] == "video-1"
    assert calls["message"]["client_trace_id"] == "trace-1"
    assert calls["media_state"]["collection"] == "fake-collection"
    assert calls["media_state"]["video"] == "fake-video"
