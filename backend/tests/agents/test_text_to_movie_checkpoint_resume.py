import copy
import importlib.util
import json
import sys
import types
from pathlib import Path


class AgentStatus:
    SUCCESS = "success"
    ERROR = "error"


class AgentResponse:
    def __init__(self, status=AgentStatus.SUCCESS, message="", data=None):
        self.status = status
        self.message = message
        self.data = data or {}


class MsgStatus:
    progress = "progress"
    success = "success"
    error = "error"


class RoleTypes:
    user = "user"
    system = "system"


class ContextMessage:
    def __init__(self, content=None, role=None, **kwargs):
        self.content = content
        self.role = role

    def to_llm_msg(self):
        return {"role": self.role, "content": self.content}


class VideoContent:
    def __init__(self, agent_name=None, status=None, status_message=None, **kwargs):
        self.agent_name = agent_name
        self.status = status
        self.status_message = status_message
        self.video = None


class VideoData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeOutputMessage:
    def __init__(self):
        self.actions = []
        self.content = []
        self.push_count = 0
        self.publish_count = 0

    def push_update(self):
        self.push_count += 1

    def publish(self):
        self.publish_count += 1


class FakeDB:
    def __init__(self):
        self.context = {}
        self.cas_calls = 0

    def get_context_messages(self, session_id):
        return copy.deepcopy(self.context.get(session_id, {}))

    def add_or_update_context_msg(self, session_id, context):
        self.context[session_id] = copy.deepcopy(context)

    def compare_and_swap_context_msg(self, session_id, expected_context, context_messages):
        self.cas_calls += 1
        current = self.context.get(session_id, {})
        if current != expected_context:
            return False
        self.context[session_id] = copy.deepcopy(context_messages)
        return True


class FakeSession:
    def __init__(self, db, session_id="session-1"):
        self.output_message = FakeOutputMessage()
        self.db = db
        self.session_id = session_id
        self.agent_context = {}
        self.state = {}


class BaseAgent:
    def __init__(self, session=None, **kwargs):
        self.session = session
        self.output_message = session.output_message


class FakeLLMResponse:
    def __init__(self, *, content="", status=True):
        self.content = content
        self.status = status


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_completions(self, *args, **kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class FakeVideoGenerationTool:
    script = []
    total_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def text_to_video(self, *args, **kwargs):
        FakeVideoGenerationTool.total_calls += 1
        if not FakeVideoGenerationTool.script:
            raise AssertionError("unexpected video generation call")
        result = FakeVideoGenerationTool.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeAudioGenerationTool:
    script = []
    total_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def generate_sound_effect(self, *args, **kwargs):
        FakeAudioGenerationTool.total_calls += 1
        if not FakeAudioGenerationTool.script:
            raise AssertionError("unexpected audio generation call")
        result = FakeAudioGenerationTool.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class DummyAsset:
    def __init__(self, asset_id=None, *args, **kwargs):
        self.asset_id = asset_id


class FakeTimeline:
    fail_next = False
    generation_calls = 0

    def __init__(self):
        self.inline_ids = []
        self.overlay_ids = []

    def add_inline(self, asset):
        self.inline_ids.append(asset.asset_id)

    def add_overlay(self, start, asset):
        self.overlay_ids.append(asset.asset_id)

    def generate_stream(self):
        FakeTimeline.generation_calls += 1
        if FakeTimeline.fail_next:
            FakeTimeline.fail_next = False
            raise RuntimeError("combine provider failed")
        return "https://stream.example/final.m3u8"


class FakeVideoDBTool:
    upload_script = []
    upload_calls = 0

    def __init__(self, collection_id=None):
        self.collection_id = collection_id

    def upload(self, *args, **kwargs):
        FakeVideoDBTool.upload_calls += 1
        if not FakeVideoDBTool.upload_script:
            raise AssertionError("unexpected upload call")
        result = FakeVideoDBTool.upload_script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get_videos(self):
        return []

    def get_audios(self):
        return []

    def get_and_set_timeline(self):
        return FakeTimeline()


def valid_visual_style():
    return json.dumps(
        {
            "camera_setup": "35mm cinema camera",
            "color_grading": "warm contrast",
            "lighting_style": "soft directional",
            "movement_style": "slow dolly",
            "film_mood": "hopeful",
            "director_reference": "restrained cinematic realism",
            "character_constants": {
                "physical_description": "short dark hair",
                "costume_details": "navy jacket",
            },
            "setting_constants": {
                "time_period": "present day",
                "environment": "urban apartment",
            },
        }
    )


def valid_scene_sequence():
    return json.dumps(
        {
            "scenes": [
                {
                    "story_beat": "arrival",
                    "scene_description": "wide apartment entrance",
                    "suggested_duration": 4,
                },
                {
                    "story_beat": "recognition",
                    "scene_description": "medium reaction shot",
                    "suggested_duration": 5,
                },
                {
                    "story_beat": "reunion",
                    "scene_description": "close embrace shot",
                    "suggested_duration": 5,
                },
            ]
        }
    )


def load_text_to_movie_module(monkeypatch):
    videodb_module = types.ModuleType("videodb")
    asset_module = types.ModuleType("videodb.asset")
    asset_module.VideoAsset = DummyAsset
    asset_module.AudioAsset = DummyAsset
    videodb_module.asset = asset_module

    base_module = types.ModuleType("director.agents.base")
    base_module.BaseAgent = BaseAgent
    base_module.AgentResponse = AgentResponse
    base_module.AgentStatus = AgentStatus

    session_module = types.ModuleType("director.core.session")
    session_module.Session = FakeSession
    session_module.ContextMessage = ContextMessage
    session_module.MsgStatus = MsgStatus
    session_module.VideoContent = VideoContent
    session_module.RoleTypes = RoleTypes
    session_module.VideoData = VideoData

    llm_module = types.ModuleType("director.llm")
    llm_module.get_default_llm = lambda: None

    kling_module = types.ModuleType("director.tools.kling")
    kling_module.KlingAITool = FakeVideoGenerationTool
    kling_module.PARAMS_CONFIG = {"text_to_video": {}}

    stability_module = types.ModuleType("director.tools.stabilityai")
    stability_module.StabilityAITool = FakeVideoGenerationTool
    stability_module.PARAMS_CONFIG = {"text_to_video": {}}

    elevenlabs_module = types.ModuleType("director.tools.elevenlabs")
    elevenlabs_module.ElevenLabsTool = FakeAudioGenerationTool
    elevenlabs_module.PARAMS_CONFIG = {"sound_effect": {}}

    videodb_tool_module = types.ModuleType("director.tools.videodb_tool")
    videodb_tool_module.VDBAudioGenerationTool = FakeAudioGenerationTool
    videodb_tool_module.VDBVideoGenerationTool = FakeVideoGenerationTool
    videodb_tool_module.VideoDBTool = FakeVideoDBTool

    constants_module = types.ModuleType("director.constants")
    constants_module.DOWNLOADS_PATH = "/tmp/director-checkpoint-tests"

    monkeypatch.setitem(sys.modules, "videodb", videodb_module)
    monkeypatch.setitem(sys.modules, "videodb.asset", asset_module)
    monkeypatch.setitem(sys.modules, "director.agents.base", base_module)
    monkeypatch.setitem(sys.modules, "director.core.session", session_module)
    monkeypatch.setitem(sys.modules, "director.llm", llm_module)
    monkeypatch.setitem(sys.modules, "director.tools.kling", kling_module)
    monkeypatch.setitem(sys.modules, "director.tools.stabilityai", stability_module)
    monkeypatch.setitem(sys.modules, "director.tools.elevenlabs", elevenlabs_module)
    monkeypatch.setitem(sys.modules, "director.tools.videodb_tool", videodb_tool_module)
    monkeypatch.setitem(sys.modules, "director.constants", constants_module)

    sys.modules.pop("director.core.text_to_movie_checkpoint", None)

    agent_path = Path(__file__).resolve().parents[2] / "director" / "agents" / "text_to_movie.py"
    spec = importlib.util.spec_from_file_location(
        "director_text_to_movie_checkpoint_resume_test", agent_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_scripts():
    FakeVideoGenerationTool.script = []
    FakeVideoGenerationTool.total_calls = 0
    FakeAudioGenerationTool.script = []
    FakeAudioGenerationTool.total_calls = 0
    FakeVideoDBTool.upload_script = []
    FakeVideoDBTool.upload_calls = 0
    FakeTimeline.fail_next = False
    FakeTimeline.generation_calls = 0


def make_agent(module, db, llm):
    session = FakeSession(db)
    agent = module.TextToMovieAgent(session=session)
    agent.llm = llm
    return agent, session


def run_request(agent):
    return agent.run(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )


def first_run_llm():
    return FakeLLM(
        [
            FakeLLMResponse(content=valid_visual_style()),
            FakeLLMResponse(content=valid_scene_sequence()),
            FakeLLMResponse(content="prompt scene one"),
            FakeLLMResponse(content="prompt scene two"),
            FakeLLMResponse(content="prompt scene three"),
        ]
    )


def test_scene_failure_persists_completed_scenes_and_retry_resumes_only_failed_scene(monkeypatch):
    reset_scripts()
    module = load_text_to_movie_module(monkeypatch)
    db = FakeDB()
    FakeVideoGenerationTool.script = [
        {"id": "scene-1", "length": 4},
        {"id": "scene-2", "length": 5},
        RuntimeError("scene three provider failure"),
    ]
    first_agent, _ = make_agent(module, db, first_run_llm())
    first_response = run_request(first_agent)

    assert first_response.status == AgentStatus.ERROR
    assert first_response.data["error"] == "text_to_movie_partial_failure"
    assert first_response.data["stage"] == "scene_generation"
    assert first_response.data["code"] == "scene_generation_failed"
    assert first_response.data["completed_scenes"] == 2
    assert first_response.data["failed_scene_index"] == 2
    assert first_response.data["resumable"] is True
    assert FakeVideoGenerationTool.total_calls == 3
    assert FakeAudioGenerationTool.total_calls == 0

    checkpoint_id = first_response.data["checkpoint_id"]
    store = module.TextToMovieCheckpointStore(FakeSession(db))
    persisted = store.get(checkpoint_id)
    assert persisted.completed_scene_count == 2
    assert persisted.scenes[0].media["id"] == "scene-1"
    assert persisted.scenes[1].media["id"] == "scene-2"
    assert persisted.scenes[2].prompt == "prompt scene three"
    assert persisted.scenes[2].media is None

    FakeVideoGenerationTool.script = [{"id": "scene-3", "length": 5}]
    FakeAudioGenerationTool.script = [{"id": "audio-1", "length": 14}]
    retry_llm = FakeLLM([FakeLLMResponse(content="soft piano rising to warmth")])
    retry_agent, _ = make_agent(module, db, retry_llm)
    retry_response = run_request(retry_agent)

    assert retry_response.status == AgentStatus.SUCCESS
    assert retry_response.data["video_url"] == "https://stream.example/final.m3u8"
    assert retry_response.data["checkpoint_id"] == checkpoint_id
    assert FakeVideoGenerationTool.total_calls == 4
    assert retry_llm.calls == 1
    assert FakeAudioGenerationTool.total_calls == 1
    assert FakeTimeline.generation_calls == 1

    completed = store.get(checkpoint_id)
    assert completed.status == "complete"
    assert completed.completed_scene_count == 3
    assert completed.audio_media["id"] == "audio-1"
    assert completed.final_video == "https://stream.example/final.m3u8"


def test_combine_failure_retries_without_regenerating_scene_or_audio_assets(monkeypatch):
    reset_scripts()
    module = load_text_to_movie_module(monkeypatch)
    db = FakeDB()
    FakeVideoGenerationTool.script = [
        {"id": "scene-1", "length": 4},
        {"id": "scene-2", "length": 5},
        {"id": "scene-3", "length": 5},
    ]
    FakeAudioGenerationTool.script = [{"id": "audio-1", "length": 14}]
    FakeTimeline.fail_next = True
    first_llm = FakeLLM(
        [
            FakeLLMResponse(content=valid_visual_style()),
            FakeLLMResponse(content=valid_scene_sequence()),
            FakeLLMResponse(content="prompt scene one"),
            FakeLLMResponse(content="prompt scene two"),
            FakeLLMResponse(content="prompt scene three"),
            FakeLLMResponse(content="soft piano rising to warmth"),
        ]
    )
    first_agent, _ = make_agent(module, db, first_llm)
    first_response = run_request(first_agent)

    assert first_response.status == AgentStatus.ERROR
    assert first_response.data["stage"] == "combine"
    assert first_response.data["code"] == "combine_failed"
    assert first_response.data["completed_scenes"] == 3
    assert FakeVideoGenerationTool.total_calls == 3
    assert FakeAudioGenerationTool.total_calls == 1
    assert FakeTimeline.generation_calls == 1

    retry_agent, _ = make_agent(module, db, FakeLLM([]))
    retry_response = run_request(retry_agent)
    assert retry_response.status == AgentStatus.SUCCESS
    assert FakeVideoGenerationTool.total_calls == 3
    assert FakeAudioGenerationTool.total_calls == 1
    assert retry_response.data["video_url"] == "https://stream.example/final.m3u8"
    assert FakeTimeline.generation_calls == 2


def test_completed_checkpoint_short_circuits_all_providers(monkeypatch):
    reset_scripts()
    module = load_text_to_movie_module(monkeypatch)
    db = FakeDB()
    FakeVideoGenerationTool.script = [
        {"id": "scene-1", "length": 4},
        {"id": "scene-2", "length": 5},
        {"id": "scene-3", "length": 5},
    ]
    FakeAudioGenerationTool.script = [{"id": "audio-1", "length": 14}]
    first_llm = FakeLLM(
        [
            FakeLLMResponse(content=valid_visual_style()),
            FakeLLMResponse(content=valid_scene_sequence()),
            FakeLLMResponse(content="prompt scene one"),
            FakeLLMResponse(content="prompt scene two"),
            FakeLLMResponse(content="prompt scene three"),
            FakeLLMResponse(content="soft piano rising to warmth"),
        ]
    )
    first_agent, _ = make_agent(module, db, first_llm)
    assert run_request(first_agent).status == AgentStatus.SUCCESS

    video_calls = FakeVideoGenerationTool.total_calls
    audio_calls = FakeAudioGenerationTool.total_calls
    combine_calls = FakeTimeline.generation_calls

    cached_agent, _ = make_agent(module, db, FakeLLM([]))
    cached_response = run_request(cached_agent)
    assert cached_response.status == AgentStatus.SUCCESS
    assert cached_response.data["resumed"] is True
    assert FakeVideoGenerationTool.total_calls == video_calls
    assert FakeAudioGenerationTool.total_calls == audio_calls
    assert FakeTimeline.generation_calls == combine_calls
