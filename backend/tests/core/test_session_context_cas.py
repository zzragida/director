import copy
import importlib.util
import sys
import types
from pathlib import Path


class FakeDB:
    def __init__(self):
        self.context = {
            "session-1": {
                "reasoning": [
                    {"role": "user", "content": "persisted reasoning"}
                ],
                "__generation_operation_leases__": [
                    {"role": "system", "content": "{\"lease\":1}"}
                ],
                "__text_to_movie_checkpoints__": [
                    {"role": "system", "content": "{\"checkpoint\":1}"}
                ],
            }
        }
        self.fail_next_cas = False

    def get_context_messages(self, session_id):
        return copy.deepcopy(self.context.get(session_id, {}))

    def compare_and_swap_context_msg(self, session_id, expected_context, context_messages):
        if self.fail_next_cas:
            self.fail_next_cas = False
            peer = copy.deepcopy(self.context.get(session_id, {}))
            peer["__generation_operation_leases__"] = [
                {"role": "system", "content": "{\"lease\":2}"}
            ]
            self.context[session_id] = peer
            return False
        current = self.context.get(session_id, {})
        if current != expected_context:
            return False
        self.context[session_id] = copy.deepcopy(context_messages)
        return True

    def add_or_update_msg_to_conv(self, **kwargs):
        pass

    def get_conversations(self, session_id):
        return []

    def get_session(self, session_id):
        return {"session_id": session_id}

    def get_sessions(self):
        return []

    def create_session(self, **kwargs):
        pass

    def delete_session(self, session_id):
        return True, []


def load_session_module(monkeypatch):
    socket_module = types.ModuleType("flask_socketio")
    socket_module.emit = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flask_socketio", socket_module)

    session_path = (
        Path(__file__).resolve().parents[2]
        / "director"
        / "core"
        / "session.py"
    )
    spec = importlib.util.spec_from_file_location(
        "director_session_context_cas_test",
        session_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_save_preserves_reserved_durable_context(monkeypatch):
    session_module = load_session_module(monkeypatch)
    db = FakeDB()
    session = session_module.Session(
        db=db,
        session_id="session-1",
        conv_id="conv-1",
    )

    session.reasoning_context.append(
        session_module.ContextMessage(
            role=session_module.RoleTypes.assistant,
            content="new reasoning",
        )
    )
    session.agent_context["summarize_video"] = [
        session_module.ContextMessage(
            role=session_module.RoleTypes.system,
            content="agent context",
        )
    ]
    # A stale reserved entry in local agent context must not override DB state.
    session.agent_context["__generation_operation_leases__"] = [
        session_module.ContextMessage(
            role=session_module.RoleTypes.system,
            content="{\"lease\":0}",
        )
    ]

    session.save_context_messages()

    persisted = db.context["session-1"]
    assert persisted["__generation_operation_leases__"][0]["content"] == "{\"lease\":1}"
    assert persisted["__text_to_movie_checkpoints__"][0]["content"] == "{\"checkpoint\":1}"
    assert persisted["summarize_video"][0]["content"] == "agent context"
    assert persisted["reasoning"][-1]["content"] == "new reasoning"


def test_session_save_retries_cas_and_preserves_peer_lease_update(monkeypatch):
    session_module = load_session_module(monkeypatch)
    db = FakeDB()
    db.fail_next_cas = True
    session = session_module.Session(
        db=db,
        session_id="session-1",
        conv_id="conv-1",
    )
    session.reasoning_context.append(
        session_module.ContextMessage(
            role=session_module.RoleTypes.assistant,
            content="local update",
        )
    )

    session.save_context_messages()

    persisted = db.context["session-1"]
    assert persisted["__generation_operation_leases__"][0]["content"] == "{\"lease\":2}"
    assert persisted["__text_to_movie_checkpoints__"][0]["content"] == "{\"checkpoint\":1}"
    assert persisted["reasoning"][-1]["content"] == "local update"
