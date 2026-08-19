import importlib.util
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


class BaseContent:
    def __init__(self, agent_name=None, status=MsgStatus.progress, status_message=None, **kwargs):
        self.agent_name = agent_name
        self.status = status
        self.status_message = status_message


class TextContent(BaseContent):
    def __init__(self, text="", **kwargs):
        super().__init__(**kwargs)
        self.text = text


class VideoContent(BaseContent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.video = None


class VideoData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class VideosUpdateEvent:
    def __init__(self, collection_id=None):
        self.collection_id = collection_id


class OutputMessage:
    def __init__(self):
        self.actions = []
        self.content = []
        self.publish_count = 0
        self.push_count = 0

    def publish(self):
        self.publish_count += 1

    def push_update(self):
        self.push_count += 1


class Session:
    def __init__(self):
        self.output_message = OutputMessage()
        self.events = []

    def emit_event(self, event):
        self.events.append(event)


class BaseAgent:
    def __init__(self, session=None, **kwargs):
        self.session = session
        self.output_message = session.output_message

    def get_parameters(self):
        return {}


class FakeLLMResponse:
    def __init__(self, status, content=""):
        self.status = status
        self.content = content


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def chat_completions(self, *args, **kwargs):
        return self.response


class FakeVideoDBTool:
    def __init__(self, collection_id=None):
        self.collection_id = collection_id

    def get_transcript(self, video_id):
        return "transcript"

    def index_spoken_words(self, video_id):
        return None


class DummyYoutubeDL:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, *args, **kwargs):
        return {}


def install_common_stubs(monkeypatch):
    agents_base = types.ModuleType("director.agents.base")
    agents_base.BaseAgent = BaseAgent
    agents_base.AgentStatus = AgentStatus
    agents_base.AgentResponse = AgentResponse

    session_module = types.ModuleType("director.core.session")
    session_module.Session = Session
    session_module.MsgStatus = MsgStatus
    session_module.ContextMessage = ContextMessage
    session_module.RoleTypes = RoleTypes
    session_module.TextContent = TextContent
    session_module.VideoContent = VideoContent
    session_module.VideoData = VideoData
    session_module.VideosUpdateEvent = VideosUpdateEvent

    videodb_tool = types.ModuleType("director.tools.videodb_tool")
    videodb_tool.VideoDBTool = FakeVideoDBTool

    monkeypatch.setitem(sys.modules, "director.agents.base", agents_base)
    monkeypatch.setitem(sys.modules, "director.core.session", session_module)
    monkeypatch.setitem(sys.modules, "director.tools.videodb_tool", videodb_tool)


def load_module(monkeypatch, relative_path, module_name):
    install_common_stubs(monkeypatch)

    llm_module = types.ModuleType("director.llm")
    llm_module.get_default_llm = lambda: FakeLLM(FakeLLMResponse(True, "summary"))
    monkeypatch.setitem(sys.modules, "director.llm", llm_module)

    yt_dlp_module = types.ModuleType("yt_dlp")
    yt_dlp_module.YoutubeDL = DummyYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", yt_dlp_module)

    path = Path(__file__).resolve().parents[2] / "director" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_llm_failure_publishes_error_instead_of_invalid_status(monkeypatch):
    module = load_module(
        monkeypatch,
        "agents/summarize_video.py",
        "director_summarize_failure_contract_test",
    )
    session = Session()
    agent = module.SummarizeVideoAgent(session=session)
    agent.llm = FakeLLM(FakeLLMResponse(False, "provider failed"))

    response = agent.run(collection_id="collection", video_id="video", prompt="short")

    assert response.status == AgentStatus.ERROR
    assert session.output_message.content[-1].status == MsgStatus.error
    assert session.output_message.publish_count == 1
    assert "LLM error" in response.message


def test_playlist_partial_failure_returns_error_with_item_counts(monkeypatch):
    module = load_module(
        monkeypatch,
        "agents/upload.py",
        "director_upload_failure_contract_test",
    )
    session = Session()
    agent = module.UploadAgent(session=session)

    def fake_upload(source, source_type, media_type, name=None):
        if source.endswith("bad"):
            return AgentResponse(status=AgentStatus.ERROR, message="upload failed")
        return AgentResponse(status=AgentStatus.SUCCESS, message="ok")

    agent._upload = fake_upload
    playlist = [
        {"title": "one", "url": "https://example/one"},
        {"title": "bad", "url": "https://example/bad"},
        {"title": "three", "url": "https://example/three"},
    ]

    response = agent._upload_yt_playlist(playlist, "video")

    assert response.status == AgentStatus.ERROR
    assert response.data["total"] == 3
    assert response.data["success_count"] == 2
    assert response.data["failure_count"] == 1
    assert response.data["failed_items"][0]["title"] == "bad"
    assert any("Upload failed for bad" in action for action in session.output_message.actions)


def test_playlist_all_success_preserves_success(monkeypatch):
    module = load_module(
        monkeypatch,
        "agents/upload.py",
        "director_upload_success_contract_test",
    )
    session = Session()
    agent = module.UploadAgent(session=session)
    agent._upload = lambda *args, **kwargs: AgentResponse(
        status=AgentStatus.SUCCESS, message="ok"
    )

    playlist = [
        {"title": "one", "url": "https://example/one"},
        {"title": "two", "url": "https://example/two"},
    ]

    response = agent._upload_yt_playlist(playlist, "video")

    assert response.status == AgentStatus.SUCCESS
    assert response.data["success_count"] == 2
    assert response.data["failure_count"] == 0
