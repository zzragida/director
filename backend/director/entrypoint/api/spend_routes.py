import os

from flask import Blueprint, current_app as app

from director.core.generation_spend import GenerationSpendStatusService
from director.db import load_db


spend_bp = Blueprint("generation_spend", __name__, url_prefix="/generation/spend")


def _service() -> GenerationSpendStatusService:
    db = load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
    return GenerationSpendStatusService(db)


@spend_bp.route("/session/<session_id>", methods=["GET"])
def get_session_spend_status(session_id):
    status = _service().session_status(session_id)
    return {"data": status.model_dump(mode="json")}, 200


@spend_bp.route(
    "/session/<session_id>/run/<generation_run_id>",
    methods=["GET"],
)
def get_generation_run_spend_status(session_id, generation_run_id):
    status = _service().run_status(session_id, generation_run_id)
    if status is None:
        return {
            "error": "generation_run_not_found",
            "message": "Generation spend status was not found.",
        }, 404
    return {"data": status.model_dump(mode="json")}, 200
