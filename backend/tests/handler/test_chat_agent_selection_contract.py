import importlib.util
import sys
import types
from pathlib import Path


class MsgStatus:
    progress = "progress"
    success = "success"
    error = "error"


class FakeOutputMessage:
    def __init__(self):
        self.actions = []
        self.status = MsgStatus.progress
        self.update_count = 0

    def update_status(self, status):
        self.status = status
        self.update_count += 1


class FakeSession:
    last_instance = None

    def __init__(self, db=None, **kwargs):
        self.db = db
        self.output_message = FakeOutputMessage()
        self.collection_id = kwargs.get("collection_id")
        self.video_id = kwargs.get("video_id")
        self.state = {}
        FakeSession.last_instance = self

    def create(self):
        return None


class FakeInputMessage:
    def __init__(self, db=None, **kwargs):
        self.db = db
        self.agents = kwargs.get("agents", [])
        self.content = kwargs.get("content", "")

    def publish(self):
        return None


class FakeReasoningEngine:
    instances = 0

    def __init__(self, *args, **kwargs):
        FakeReasoningEngine.instances += 1
        self.registered = []
        self.ran = False

    def register_agents(self, agents):
        self.registered.extend(agents)

    def run(self):
        self.ran = True


class GenericAgent:
    agent_name = "generic"

    def __init__(self, session=None, **kwargs):
        self.session = session
        self.name = self.agent_name
        self.agent_description = self.agent_name


class KnownAgent(GenericAgent):
    agent_name = "known"


class FakeVideoDBTool:
    def __init__(self, collection_id="default"):
        self.collection_id = collection_id


AGENT_IMPORTS = {
    "director.agents.frame": "FrameAgent",
    "director.agents.summarize_video": "SummarizeVideoAgent",
    "director.agents.download": "DownloadAgent",
    "director.agents.pricing": "PricingAgent",
    "director.agents.upload": "UploadAgent",
    "director.agents.search": "SearchAgent",
    "director.agents.prompt_clip": "PromptClipAgent",
    "director.agents.index": "IndexAgent",
    "director.agents.censor": "CensorAgent",
    "director.agents.image_generation": "ImageGenerationAgent",
    "director.agents.audio_generation": "AudioGenerationAgent",
    "director.agents.video_generation": "VideoGenerationAgent",
    "director.agents.stream_video": "StreamVideoAgent",
    "director.agents.subtitle": "SubtitleAgent",
    "director.agents.slack_agent": "SlackAgent",
    "director.agents.editing": "EditingAgent",
    "director.agents.dubbing": "DubbingAgent",
    "director.agents.text_to_movie": "TextToMovieAgent",
    "director.agents.composio": "ComposioAgent",
    "director.agents.transcription": "TranscriptionAgent",
    "director.agents.comparison": "ComparisonAgent",
    "director.agents.code_assistant": "CodeAssistantAgent",
    "director.agents.web_search_agent": "WebSearchAgent",
    "director.agents.clone_voice": "CloneVoiceAgent",
    "director.agents.voice_replacement": "VoiceReplacementAgent",
}


def load_handler_module(monkeypatch):
    for module_name, class_name in AGENT_IMPORTS.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, GenericAgent)
        monkeypatch.setitem(sys.modules, module_name, module)

    session_module = types.ModuleType("director.core.session")
    session_module.Session = FakeSession
    session_module.InputMessage = FakeInputMessage
    session_module.MsgStatus = MsgStatus
    monkeypatch.setitem(sys.modules, "director.core.session", session_module)

    reasoning_module = types.ModuleType("director.core.reasoning")
    reasoning_module.ReasoningEngine = FakeReasoningEngine
    monkeypatch.setitem(sys.modules, "director.core.reasoning", reasoning_module)

    db_base = types.ModuleType("director.db.base")
    db_base.BaseDB = object
    monkeypatch.setitem(sys.modules, "director.db.base", db_base)

    db_module = types.ModuleType("director.db")
    db_module.load_db = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "director.db", db_module)

    videodb_tool = types.ModuleType("director.tools.videodb_tool")
    videodb_tool.VideoDBTool = FakeVideoDBTool
    monkeypatch.setitem(sys.modules, "director.tools.videodb_tool", videodb_tool)

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    handler_path = Path(__file__).resolve().parents[2] / "director" / "handler.py"
    spec = importlib.util.spec_from_file_location(
        "director_handler_agent_selection_contract_test", handler_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_requested_agents_returns_unknown_names_without_keyerror(monkeypatch):
    handler_module = load_handler_module(monkeypatch)
    agents = [KnownAgent(), GenericAgent()]

    selected, unknown = handler_module.ChatHandler.select_requested_agents(
        agents, ["known", "missing"]
    )

    assert [agent.name for agent in selected] == ["known"]
    assert unknown == ["missing"]


def test_unknown_client_agent_stops_before_reasoning(monkeypatch):
    handler_module = load_handler_module(monkeypatch)
    FakeReasoningEngine.instances = 0
    FakeSession.last_instance = None

    handler = handler_module.ChatHandler(db=object())
    handler.agents = [KnownAgent]
    handler.add_videodb_state = lambda session: None

    handler.chat(
        {
            "session_id": "session",
            "conv_id": "conversation",
            "content": "hello",
            "agents": ["missing"],
        }
    )

    session = FakeSession.last_instance
    assert session.output_message.status == MsgStatus.error
    assert session.output_message.update_count == 1
    assert any("Unknown agent requested: missing" in action for action in session.output_message.actions)
    assert FakeReasoningEngine.instances == 0


def test_known_client_agent_reaches_reasoning(monkeypatch):
    handler_module = load_handler_module(monkeypatch)
    FakeReasoningEngine.instances = 0

    handler = handler_module.ChatHandler(db=object())
    handler.agents = [KnownAgent]
    handler.add_videodb_state = lambda session: None

    handler.chat(
        {
            "session_id": "session",
            "conv_id": "conversation",
            "content": "hello",
            "agents": ["known"],
        }
    )

    assert FakeReasoningEngine.instances == 1
