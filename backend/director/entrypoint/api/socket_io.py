import os

from flask import current_app as app
from flask_socketio import Namespace
from pydantic import ValidationError

from director.core.chat_request import ChatRequest, format_chat_validation_error
from director.db import load_db
from director.handler import ChatHandler


class ChatNamespace(Namespace):
    """Chat namespace for socket.io."""

    def on_chat(self, message):
        """Validate and handle chat messages at the Socket.IO boundary."""
        try:
            request = ChatRequest.model_validate(message)
        except ValidationError as error:
            return {
                "status": "error",
                "error": "invalid_chat_request",
                "details": format_chat_validation_error(error),
            }

        chat_handler = ChatHandler(
            db=load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
        )
        return chat_handler.chat(request.model_dump())
