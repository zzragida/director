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

    def __str__(self):
        return f"AgentResponse(status={self.status}, message={self.message})"


class MsgStatus:
    progress = "progress"
    success = "success"
    error = "error"


class RoleTypes:
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class TextContent:
    def __init__(
        self,
        text="",
        status=MsgStatus.progress,
        status_message=None,
        agent_name=None,
        **kwargs,
    ):
        self.text = text
        self.status = status
        self.status_message = status_message
        self.agent_name = agent_name


class ContextMessage:
    def __init__(
        self,
        content=None,
        role=RoleTypes.system,
        tool_calls=None,
        tool_call_id=None,
        **kwargs,
    ):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

    def to_llm_msg(self):
        message = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


class OutputMessage:
    def __init__(self):
        self.actions = []
        self.agents = []
        self.content = []
        self.status = MsgStatus.progress
        self.push_count = 0
        self.publish_count = 0

    def push_update(self):
        self.push_count += 1

    def publish(self):
        self.publish_count += 1


class Session:
    def __init__(self):
        self.output_message = OutputMessage()
        self.reasoning_context = [ContextMessage(content="hello", role=RoleTypes.user)]
        self.video_id = None
        self.state = {}
        self.saved = False

    def save_context_messages(self):
        self.saved = True


class InputMessage:
    def __init__(self, content="hello"):
        self.content = content


class FakeAgent:
    def __init__(self, agent_name, response_status):
        self.agent_name = agent_name
        self.response_status = response_status

    def safe_call(self, *args, **kwargs):
        return AgentResponse(status=self.response_status, message=self.agent_name)

    def to_llm_format(self):
        return {"name": self.agent_name, "description": self.agent_name, "parameters": {}}


class FakeLLMResponse:
    def __init__(
        self,
        *,
        status=True,
        content="",
        tool_calls=None,
        finish_reason="stop",
    ):
        self.status = status
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completions(self, *args, **kwargs):
        return self.responses.pop(0)


def load_reasoning_module(monkeypatch):
    agents_base = types.ModuleType("director.agents.base")
    agents_base.BaseAgent = FakeAgent
    agents_base.AgentStatus = AgentStatus
    agents_base.AgentResponse = AgentResponse

    session_module = types.ModuleType("director.core.session")
    session_module.Session = Session
    session_module.OutputMessage = OutputMessage
    session_module.InputMessage = InputMessage
    session_module.ContextMessage = ContextMessage
    session_module.RoleTypes = RoleTypes
    session_module.TextContent = TextContent
    session_module.MsgStatus = MsgStatus

    llm_base = types.ModuleType("director.llm.base")
    llm_base.LLMResponse = FakeLLMResponse

    llm_module = types.ModuleType("director.llm")
    llm_module.get_default_llm = lambda: None

    monkeypatch.setitem(sys.modules, "director.agents.base", agents_base)
    monkeypatch.setitem(sys.modules, "director.core.session", session_module)
    monkeypatch.setitem(sys.modules, "director.llm.base", llm_base)
    monkeypatch.setitem(sys.modules, "director.llm", llm_module)

    reasoning_path = (
        Path(__file__).resolve().parents[2]
        / "director"
        / "core"
        / "reasoning.py"
    )
    module_name = "director_reasoning_contract_test"
    spec = importlib.util.spec_from_file_location(module_name, reasoning_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_engine(reasoning):
    session = Session()
    engine = reasoning.ReasoningEngine(InputMessage(), session)
    return engine, session


def test_outcome_is_success_without_failed_agents(monkeypatch):
    reasoning = load_reasoning_module(monkeypatch)
    engine, _ = make_engine(reasoning)

    assert engine.get_outcome_status() == MsgStatus.success


def test_outcome_is_error_when_any_agent_failed(monkeypatch):
    reasoning = load_reasoning_module(monkeypatch)
    engine, _ = make_engine(reasoning)
    engine.failed_agents.append("bad_agent")

    assert engine.get_outcome_status() == MsgStatus.error


def test_unknown_tool_returns_typed_error_instead_of_crashing(monkeypatch):
    reasoning = load_reasoning_module(monkeypatch)
    engine, session = make_engine(reasoning)

    response = engine.run_agent("missing_agent")

    assert response.status == AgentStatus.ERROR
    assert "Unknown agent requested" in response.message
    assert session.output_message.status == MsgStatus.progress
    assert session.output_message.push_count == 1


def test_multi_agent_failure_marks_summary_and_output_as_error(monkeypatch):
    reasoning = load_reasoning_module(monkeypatch)
    engine, session = make_engine(reasoning)
    engine.register_agents(
        [
            FakeAgent("good_agent", AgentStatus.SUCCESS),
            FakeAgent("bad_agent", AgentStatus.ERROR),
        ]
    )
    engine.iterations = engine.max_iterations - 1
    engine.llm = FakeLLM(
        [
            FakeLLMResponse(
                content="done",
                finish_reason="stop",
                tool_calls=[
                    {
                        "id": "call-good",
                        "tool": {"name": "good_agent", "arguments": {}},
                    },
                    {
                        "id": "call-bad",
                        "tool": {"name": "bad_agent", "arguments": {}},
                    },
                ],
            )
        ]
    )

    engine.step()

    assert engine.failed_agents == ["bad_agent"]
    assert session.output_message.status == MsgStatus.error
    assert engine.summary_content.status == MsgStatus.error
    assert session.output_message.publish_count == 1


def test_all_successful_agents_keep_output_success(monkeypatch):
    reasoning = load_reasoning_module(monkeypatch)
    engine, session = make_engine(reasoning)
    engine.register_agents(
        [
            FakeAgent("agent_a", AgentStatus.SUCCESS),
            FakeAgent("agent_b", AgentStatus.SUCCESS),
        ]
    )
    engine.iterations = engine.max_iterations - 1
    engine.llm = FakeLLM(
        [
            FakeLLMResponse(
                content="done",
                finish_reason="stop",
                tool_calls=[
                    {
                        "id": "call-a",
                        "tool": {"name": "agent_a", "arguments": {}},
                    },
                    {
                        "id": "call-b",
                        "tool": {"name": "agent_b", "arguments": {}},
                    },
                ],
            )
        ]
    )

    engine.step()

    assert engine.failed_agents == []
    assert session.output_message.status == MsgStatus.success
    assert engine.summary_content.status == MsgStatus.success
