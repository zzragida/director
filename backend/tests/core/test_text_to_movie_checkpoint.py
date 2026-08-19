import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class RoleTypes:
    system = "system"


class ContextMessage:
    def __init__(self, content=None, role=None, **kwargs):
        self.content = content
        self.role = role

    def to_llm_msg(self):
        return {"role": self.role, "content": self.content}


class FakeDB:
    def __init__(self):
        self.context = {}
        self.write_count = 0

    def get_context_messages(self, session_id):
        return copy.deepcopy(self.context.get(session_id, {}))

    def add_or_update_context_msg(self, session_id, context):
        self.write_count += 1
        self.context[session_id] = copy.deepcopy(context)

    def compare_and_swap_context_msg(self, session_id, expected_context, context_messages):
        current = self.context.get(session_id, {})
        if current != expected_context:
            return False
        self.context[session_id] = copy.deepcopy(context_messages)
        self.write_count += 1
        return True


class FakeSession:
    def __init__(self, db, session_id="session-1"):
        self.db = db
        self.session_id = session_id
        self.agent_context = {}


def load_checkpoint_module(monkeypatch):
    session_module = types.ModuleType("director.core.session")
    session_module.ContextMessage = ContextMessage
    session_module.RoleTypes = RoleTypes
    monkeypatch.setitem(sys.modules, "director.core.session", session_module)

    checkpoint_path = (
        Path(__file__).resolve().parents[2]
        / "director"
        / "core"
        / "text_to_movie_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "director_text_to_movie_checkpoint_contract_test", checkpoint_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_scenes():
    return [
        {
            "story_beat": "opening",
            "scene_description": "wide apartment shot",
            "suggested_duration": 4,
        },
        {
            "story_beat": "reunion",
            "scene_description": "close emotional shot",
            "suggested_duration": 5,
        },
    ]


def test_request_fingerprint_is_deterministic_and_configuration_sensitive(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)

    first = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
        video_config={"seed": 7},
        audio_config={},
    )
    same = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
        video_config={"seed": 7},
        audio_config={},
    )
    changed = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
        video_config={"seed": 8},
        audio_config={},
    )

    assert first == same
    assert first != changed
    assert checkpoint.make_checkpoint_id(first).startswith("text_to_movie:")


def test_checkpoint_store_persists_and_preserves_existing_context(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)
    db = FakeDB()
    db.context["session-1"] = {
        "reasoning": [{"role": "user", "content": "hello"}],
        "other_agent": [{"role": "system", "content": "keep-me"}],
    }
    session = FakeSession(db)

    fingerprint = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
    )
    value = checkpoint.create_checkpoint(
        request_fingerprint=fingerprint,
        visual_style={"camera_setup": "35mm"},
        scenes=sample_scenes(),
    )
    value.scenes[0].status = "complete"
    value.scenes[0].media = {"id": "video-1", "length": 4}

    store = checkpoint.TextToMovieCheckpointStore(session)
    store.save(value)

    persisted = db.context["session-1"]
    assert persisted["reasoning"][0]["content"] == "hello"
    assert persisted["other_agent"][0]["content"] == "keep-me"
    assert checkpoint.CHECKPOINT_CONTEXT_KEY in persisted
    assert checkpoint.CHECKPOINT_CONTEXT_KEY in session.agent_context
    assert db.write_count == 1
    assert value.revision == 1

    resumed_session = FakeSession(db)
    resumed = checkpoint.TextToMovieCheckpointStore(resumed_session).get(
        value.checkpoint_id
    )

    assert resumed is not None
    assert resumed.request_fingerprint == fingerprint
    assert resumed.completed_scene_count == 1
    assert resumed.scenes[0].media["id"] == "video-1"
    assert resumed.revision == 1


def test_checkpoint_store_keeps_multiple_request_fingerprints(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)
    db = FakeDB()
    session = FakeSession(db)
    store = checkpoint.TextToMovieCheckpointStore(session)

    first_fingerprint = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="First story",
    )
    second_fingerprint = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="Second story",
    )

    first = checkpoint.create_checkpoint(
        request_fingerprint=first_fingerprint,
        visual_style={"camera_setup": "35mm"},
        scenes=sample_scenes(),
    )
    second = checkpoint.create_checkpoint(
        request_fingerprint=second_fingerprint,
        visual_style={"camera_setup": "50mm"},
        scenes=sample_scenes(),
    )
    store.save(first)
    store.save(second)

    assert store.get(first.checkpoint_id).request_fingerprint == first_fingerprint
    assert store.get(second.checkpoint_id).request_fingerprint == second_fingerprint


def test_stale_checkpoint_revision_cannot_overwrite_newer_progress(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)
    db = FakeDB()
    store = checkpoint.TextToMovieCheckpointStore(FakeSession(db))

    fingerprint = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        storyline="A quiet reunion",
    )
    value = checkpoint.create_checkpoint(
        request_fingerprint=fingerprint,
        visual_style={"camera_setup": "35mm"},
        scenes=sample_scenes(),
    )
    store.save(value)

    winner = store.get(value.checkpoint_id)
    stale = store.get(value.checkpoint_id)
    winner.status = "generating"
    winner.scenes[0].status = "complete"
    winner.scenes[0].media = {"id": "scene-1", "length": 4}
    store.save(winner)

    stale.status = "failed"
    with pytest.raises(checkpoint.CheckpointConflictError):
        store.save(stale)

    persisted = store.get(value.checkpoint_id)
    assert persisted.revision == 2
    assert persisted.status == "generating"
    assert persisted.scenes[0].media["id"] == "scene-1"


def test_compact_media_requires_stable_id(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)

    assert checkpoint.compact_media(
        {"id": "video-1", "length": 5, "ignored": "large-provider-payload"}
    ) == {"id": "video-1", "length": 5}

    with pytest.raises(ValueError):
        checkpoint.compact_media({"length": 5})


def test_corrupt_checkpoint_document_is_ignored(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)
    db = FakeDB()
    db.context["session-1"] = {
        checkpoint.CHECKPOINT_CONTEXT_KEY: [
            {"role": "system", "content": "not-json"}
        ]
    }

    store = checkpoint.TextToMovieCheckpointStore(FakeSession(db))
    assert store.get("text_to_movie:anything") is None
