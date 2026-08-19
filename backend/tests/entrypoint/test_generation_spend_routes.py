import importlib.util
import sys
import types
from pathlib import Path


class FakeBlueprint:
    def __init__(self, *args, **kwargs):
        self.routes = []

    def route(self, rule, **options):
        def decorator(func):
            self.routes.append((rule, options, func.__name__))
            return func

        return decorator


class FakeStatus:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, mode=None):
        return dict(self.payload)


def load_spend_routes(monkeypatch, calls):
    flask_module = types.ModuleType("flask")
    flask_module.Blueprint = FakeBlueprint
    flask_module.current_app = types.SimpleNamespace(config={"DB_TYPE": "sqlite"})

    db_module = types.ModuleType("director.db")

    def load_db(db_type):
        calls["load_db"] += 1
        calls["db_type"] = db_type
        return "fake-db"

    db_module.load_db = load_db

    spend_module = types.ModuleType("director.core.generation_spend")

    class FakeService:
        def __init__(self, db):
            calls["service_init"] += 1
            calls["db"] = db

        def session_status(self, session_id):
            calls["session_status"] += 1
            calls["session_id"] = session_id
            return FakeStatus(
                {
                    "session_id": session_id,
                    "run_count": 1,
                    "runs": [],
                    "status_digest": "safe-digest",
                }
            )

        def run_status(self, session_id, generation_run_id):
            calls["run_status"] += 1
            calls["session_id"] = session_id
            calls["generation_run_id"] = generation_run_id
            if generation_run_id == "genrun:missing":
                return None
            return FakeStatus(
                {
                    "generation_run_id": generation_run_id,
                    "effective_exposure_micros": 120,
                    "remaining_total_micros": 80,
                    "circuit_breaker": {"state": "closed", "reason_codes": []},
                }
            )

    spend_module.GenerationSpendStatusService = FakeService

    monkeypatch.setitem(sys.modules, "flask", flask_module)
    monkeypatch.setitem(sys.modules, "director.db", db_module)
    monkeypatch.setitem(sys.modules, "director.core.generation_spend", spend_module)

    route_path = (
        Path(__file__).resolve().parents[2]
        / "director"
        / "entrypoint"
        / "api"
        / "spend_routes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "director_generation_spend_routes_test",
        route_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_calls():
    return {
        "load_db": 0,
        "service_init": 0,
        "session_status": 0,
        "run_status": 0,
        "db_type": None,
        "db": None,
        "session_id": None,
        "generation_run_id": None,
    }


def test_session_spend_route_is_read_only_status_response(monkeypatch):
    calls = make_calls()
    module = load_spend_routes(monkeypatch, calls)

    response, status_code = module.get_session_spend_status("session-1")

    assert status_code == 200
    assert response["data"]["session_id"] == "session-1"
    assert response["data"]["status_digest"] == "safe-digest"
    assert calls["load_db"] == 1
    assert calls["service_init"] == 1
    assert calls["session_status"] == 1
    assert calls["run_status"] == 0
    assert calls["db_type"] == "sqlite"


def test_run_spend_route_returns_safe_read_model(monkeypatch):
    calls = make_calls()
    module = load_spend_routes(monkeypatch, calls)

    response, status_code = module.get_generation_run_spend_status(
        "session-1",
        "genrun:test",
    )

    assert status_code == 200
    assert response["data"]["generation_run_id"] == "genrun:test"
    assert response["data"]["effective_exposure_micros"] == 120
    assert response["data"]["remaining_total_micros"] == 80
    assert response["data"]["circuit_breaker"]["state"] == "closed"
    assert "invoice_id" not in str(response)
    assert "provider_request_id" not in str(response)
    assert calls["run_status"] == 1


def test_missing_run_returns_stable_404_contract(monkeypatch):
    calls = make_calls()
    module = load_spend_routes(monkeypatch, calls)

    response, status_code = module.get_generation_run_spend_status(
        "session-1",
        "genrun:missing",
    )

    assert status_code == 404
    assert response == {
        "error": "generation_run_not_found",
        "message": "Generation spend status was not found.",
    }
    assert calls["run_status"] == 1
