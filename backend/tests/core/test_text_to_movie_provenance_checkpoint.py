import copy
import importlib.util
import sys
import types
from pathlib import Path


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

    def get_context_messages(self, session_id):
        return copy.deepcopy(self.context.get(session_id, {}))

    def compare_and_swap_context_msg(self, session_id, expected_context, context_messages):
        current = self.context.get(session_id, {})
        if current != expected_context:
            return False
        self.context[session_id] = copy.deepcopy(context_messages)
        return True


class FakeSession:
    def __init__(self, db, session_id="session-1"):
        self.db = db
        self.session_id = session_id
        self.agent_context = {}
        self.state = {}


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
        "director_text_to_movie_provenance_checkpoint_test",
        checkpoint_path,
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
        }
    ]


def test_checkpoint_enriches_request_provenance_from_validated_agent_context(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)
    db = FakeDB()
    session = FakeSession(db)
    session.state[checkpoint.ACTIVE_AGENT_CALL_STATE_KEY] = {
        "agent_name": "text_to_movie",
        "arguments": {
            "collection_id": "collection-1",
            "engine": "kling",
            "audio_engine": "elevenlabs",
            "job_type": "text_to_movie",
            "text_to_movie": {
                "storyline": "A quiet reunion",
                "video_kling_config": {
                    "seed": 7,
                    "api_key": "must-not-persist",
                },
                "audio_elevenlabs_config": {
                    "duration": 4,
                    "access_token": "must-not-persist-either",
                },
            },
        },
    }

    fingerprint = checkpoint.build_request_fingerprint(
        collection_id="collection-1",
        engine="kling",
        audio_engine="elevenlabs",
        storyline="A quiet reunion",
        video_config={"seed": 7, "api_key": "must-not-persist"},
        audio_config={"duration": 4, "access_token": "must-not-persist-either"},
    )
    value = checkpoint.create_checkpoint(
        request_fingerprint=fingerprint,
        visual_style={"camera_setup": "35mm"},
        scenes=sample_scenes(),
        video_provider="kling",
        audio_provider="elevenlabs",
    )

    store = checkpoint.TextToMovieCheckpointStore(session)
    store.save(value)

    resumed = store.get(value.checkpoint_id)
    assert resumed.provenance is not None
    request = resumed.provenance.request
    assert request.collection_id == "collection-1"
    assert request.storyline == "A quiet reunion"
    assert request.video_provider == "kling"
    assert request.audio_provider == "elevenlabs"
    assert request.video_config["seed"] == 7
    assert request.video_config["api_key"] == "[REDACTED]"
    assert request.audio_config["access_token"] == "[REDACTED]"
    assert "must-not-persist" not in str(db.context)
    assert "must-not-persist-either" not in str(db.context)


def test_checkpoint_save_synchronizes_scene_audio_and_final_lineage(monkeypatch):
    checkpoint = load_checkpoint_module(monkeypatch)
    db = FakeDB()
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
        video_provider="videodb",
        audio_provider="videodb",
        collection_id="collection-1",
        storyline="A quiet reunion",
    )
    store = checkpoint.TextToMovieCheckpointStore(session)
    store.save(value)
    first_digest = value.provenance.manifest_digest

    scene = value.scenes[0]
    scene.prompt = "cinematic reunion prompt"
    scene.media = {"id": "scene-1", "length": 4, "collection_id": "collection-1"}
    scene.status = "complete"
    scene.operation.state = "persisted"
    scene.operation.artifact = dict(scene.media)
    scene.operation.provider_request_id = "provider-scene-1"
    scene.operation.attempt_count = 1

    value.audio_prompt = "soft piano"
    value.audio_media = {"id": "audio-1", "length": 4, "collection_id": "collection-1"}
    value.audio_operation.state = "persisted"
    value.audio_operation.artifact = dict(value.audio_media)
    value.audio_operation.attempt_count = 1
    value.final_video = "https://stream.example/final.m3u8"
    value.status = "complete"
    store.save(value)

    resumed = store.get(value.checkpoint_id)
    manifest = resumed.provenance
    assert manifest.manifest_digest != first_digest
    assert manifest.scenes[0].prompt == "cinematic reunion prompt"
    assert manifest.scenes[0].artifact_id == "scene-1"
    assert manifest.scenes[0].provider_request_id == "provider-scene-1"
    assert manifest.audio.prompt == "soft piano"
    assert manifest.audio.artifact_id == "audio-1"
    assert manifest.final.stream_url == "https://stream.example/final.m3u8"
    assert manifest.final.source_scene_artifact_ids == ["scene-1"]
    assert manifest.final.source_audio_artifact_id == "audio-1"
