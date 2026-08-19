import importlib.util
import sys
import types
from pathlib import Path


class FakeSession:
    def __init__(self):
        self.output_message = object()


def load_base_agent_module(monkeypatch):
    function_calling = types.ModuleType("openai_function_calling")

    class FunctionInferrer:
        @staticmethod
        def infer_from_function_reference(*args, **kwargs):
            raise AssertionError("inference should not be used in this contract test")

    function_calling.FunctionInferrer = FunctionInferrer

    session_module = types.ModuleType("director.core.session")
    session_module.Session = FakeSession
    session_module.OutputMessage = object

    monkeypatch.setitem(sys.modules, "openai_function_calling", function_calling)
    monkeypatch.setitem(sys.modules, "director.core.session", session_module)

    base_path = Path(__file__).resolve().parents[2] / "director" / "agents" / "base.py"
    spec = importlib.util.spec_from_file_location("director_base_tool_contract_test", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_agent(base_module, name="contract_agent", parameters=None):
    parameters = parameters or {
        "type": "object",
        "properties": {
            "collection_id": {"type": "string"},
            "duration": {"type": "integer"},
        },
        "required": ["collection_id", "duration"],
    }

    class ContractAgent(base_module.BaseAgent):
        def __init__(self):
            self.agent_name = name
            self.description = "contract test agent"
            self.parameters = parameters
            self.executions = 0
            super().__init__(session=FakeSession())

        def run(self, *args, **kwargs):
            self.executions += 1
            return base_module.AgentResponse(
                status=base_module.AgentStatus.SUCCESS,
                message="executed",
            )

    return ContractAgent()


def test_invalid_keyword_arguments_return_typed_error_without_execution(monkeypatch):
    base_module = load_base_agent_module(monkeypatch)
    agent = make_agent(base_module)

    response = agent.safe_call(collection_id="collection-1", duration="5")

    assert response.status == base_module.AgentStatus.ERROR
    assert response.data["error"] == "invalid_tool_arguments"
    assert response.data["details"][0]["field"] == "duration"
    assert agent.executions == 0
    assert "5" not in str(response.data["details"])


def test_valid_keyword_arguments_execute_agent(monkeypatch):
    base_module = load_base_agent_module(monkeypatch)
    agent = make_agent(base_module)

    response = agent.safe_call(collection_id="collection-1", duration=5)

    assert response.status == base_module.AgentStatus.SUCCESS
    assert agent.executions == 1


def test_positional_internal_calls_keep_existing_compatibility(monkeypatch):
    base_module = load_base_agent_module(monkeypatch)
    agent = make_agent(base_module)

    response = agent.safe_call("collection-1", 5)

    assert response.status == base_module.AgentStatus.SUCCESS
    assert agent.executions == 1


def test_text_to_movie_advertises_runtime_required_payload(monkeypatch):
    base_module = load_base_agent_module(monkeypatch)
    parameters = {
        "type": "object",
        "properties": {
            "collection_id": {"type": "string"},
            "engine": {"type": "string"},
            "job_type": {"type": "string", "enum": ["text_to_movie"]},
            "text_to_movie": {
                "type": "object",
                "properties": {"storyline": {"type": "string"}},
                "required": ["storyline"],
            },
        },
        "required": ["collection_id", "engine", "job_type"],
    }
    agent = make_agent(base_module, name="text_to_movie", parameters=parameters)

    llm_schema = agent.to_llm_format()["parameters"]
    assert "text_to_movie" in llm_schema["required"]

    response = agent.safe_call(
        collection_id="collection-1",
        engine="videodb",
        job_type="text_to_movie",
    )

    assert response.status == base_module.AgentStatus.ERROR
    assert response.data["details"][0]["field"] == "text_to_movie"
    assert agent.executions == 0
