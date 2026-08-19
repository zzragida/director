import importlib.util
import sys
import types
from pathlib import Path


class DummyNamespace:
    def __init__(self, *args, **kwargs):
        pass


def load_socket_module(monkeypatch, calls):
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

        def chat(self, message):
            calls["chat"] += 1
            calls["message"] = message
            return {"status": "handled"}

    handler_module.ChatHandler = FakeChatHandler

    monkeypatch.setitem(sys.modules, "flask", flask_module)
    monkeypatch.setitem(sys.modules, "flask_socketio", socketio_module)
    monkeypatch.setitem(sys.modules, "director.db", db_module)
    monkeypatch.setitem(sys.modules, "director.handler", handler_module)

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
        "db_type": None,
        "db": None,
        "message": None,
    }


def test_invalid_request_is_rejected_before_db_and_handler(monkeypatch):
    calls = make_calls()
    socket_module = load_socket_module(monkeypatch, calls)
    namespace = socket_module.ChatNamespace("/chat")

    response = namespace.on_chat(
        {"session_id": " ", "conv_id": "conv-1", "content": "hello"}
    )

    assert response["status"] == "error"
    assert response["error"] == "invalid_chat_request"
    assert calls["load_db"] == 0
    assert calls["handler_init"] == 0
    assert calls["chat"] == 0


def test_valid_request_is_normalized_before_handler(monkeypatch):
    calls = make_calls()
    socket_module = load_socket_module(monkeypatch, calls)
    namespace = socket_module.ChatNamespace("/chat")

    response = namespace.on_chat(
        {
            "session_id": " session-1 ",
            "conv_id": " conv-1 ",
            "content": "hello",
            "collection_id": "   ",
            "video_id": "",
            "agents": ["summarize_video"],
            "client_trace_id": "trace-1",
        }
    )

    assert response == {"status": "handled"}
    assert calls["load_db"] == 1
    assert calls["handler_init"] == 1
    assert calls["chat"] == 1
    assert calls["db_type"] == "sqlite"
    assert calls["message"]["session_id"] == "session-1"
    assert calls["message"]["conv_id"] == "conv-1"
    assert calls["message"]["collection_id"] is None
    assert calls["message"]["video_id"] is None
    assert calls["message"]["client_trace_id"] == "trace-1"
