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


class FakeSession:
    def __init__(self):
        self.output_message = FakeOutputMessage()


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

    def chat_completions(self, *args, **kwargs):
        return self.responses.pop(0)


class FakeVideoDBTool:
    upload_calls = 0

    def __init__(self, collection_id=None):
        self.collection_id = collection_id

    def upload(self, *args, **kwargs):
        FakeVideoDBTool.upload_calls += 1
        return {"id": "uploaded", "length": 1}


class FakeVideoGenerationTool:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        FakeVideoGenerationTool.instances.append(self)

    def text_to_video(self, *args, **kwargs):
        self.calls += 1
        return {"id": "generated", "length": 1}


class FakeAudioGenerationTool:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        FakeAudioGenerationTool.instances.append(self)

    def generate_sound_effect(self, *args, **kwargs):
        self.calls += 1
        return {"id": "audio"}


class DummyAsset:
    def __init__(self, *args, **kwargs):
        pass


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
    constants_module.DOWNLOADS_PATH = "/tmp/director-contract-tests"

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

    agent_path = (
        Path(__file__).resolve().parents[2]
        / "director"
        / "agents"
        / "text_to_movie.py"
    )
    spec = importlib.util.spec_from_file_location(
        "director_text_to_movie_structured_contract_test", agent_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_side_effect_counters():
    FakeVideoDBTool.upload_calls = 0
    FakeVideoGenerationTool.instances = []
    FakeAudioGenerationTool.instances = []


def count_video_calls():
    return sum(tool.calls for tool in FakeVideoGenerationTool.instances)


def count_audio_calls():
    return sum(tool.calls for tool in FakeAudioGenerationTool.instances)


def make_agent(module, llm):
    session = FakeSession()
    agent = module.TextToMovieAgent(session=session)
    agent.llm = llm
    return agent, session


def test_invalid_visual_style_stops_before_media_generation(monkeypatch):
    reset_side_effect_counters()
    module = load_text_to_movie_module(monkeypatch)
    agent, session = make_agent(
        module,
        FakeLLM([FakeLLMResponse(content="not-json")]),
    )

    response = agent.run(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )

    assert response.status == AgentStatus.ERROR
    assert response.data["error"] == "structured_generation_failed"
    assert response.data["stage"] == "visual_style"
    assert response.data["code"] == "invalid_json"
    assert count_video_calls() == 0
    assert count_audio_calls() == 0
    assert FakeVideoDBTool.upload_calls == 0
    assert session.output_message.content[0].status == MsgStatus.error


def test_invalid_scene_sequence_stops_before_media_generation(monkeypatch):
    reset_side_effect_counters()
    module = load_text_to_movie_module(monkeypatch)
    agent, _ = make_agent(
        module,
        FakeLLM(
            [
                FakeLLMResponse(content=valid_visual_style()),
                FakeLLMResponse(content='{\"scenes\": []}'),
            ]
        ),
    )

    response = agent.run(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "A quiet reunion"},
    )

    assert response.status == AgentStatus.ERROR
    assert response.data["stage"] == "scene_sequence"
    assert response.data["code"] == "invalid_structure"
    assert count_video_calls() == 0
    assert count_audio_calls() == 0
    assert FakeVideoDBTool.upload_calls == 0


def test_storyline_validation_stops_before_tool_initialization(monkeypatch):
    reset_side_effect_counters()
    module = load_text_to_movie_module(monkeypatch)
    agent, _ = make_agent(module, FakeLLM([]))

    response = agent.run(
        collection_id="collection-1",
        engine="videodb",
        audio_engine="videodb",
        job_type="text_to_movie",
        text_to_movie={"storyline": "   "},
    )

    assert response.status == AgentStatus.ERROR
    assert response.data["stage"] == "input"
    assert response.data["code"] == "invalid_storyline"
    assert FakeVideoGenerationTool.instances == []
    assert FakeAudioGenerationTool.instances == []
    assert FakeVideoDBTool.upload_calls == 0
