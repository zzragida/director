import os

from flask import current_app as app
from flask_socketio import Namespace
from pydantic import ValidationError

from director.core.chat_request import ChatRequest, format_chat_validation_error
from director.core.media_reference import MediaReferenceError, resolve_media_reference
from director.db import load_db
from director.handler import ChatHandler


class ChatNamespace(Namespace):
    """Chat namespace for socket.io."""

    def on_chat(self, message):
        """Validate chat input and media references before side effects."""
        try:
            request = ChatRequest.model_validate(message)
        except ValidationError as error:
            return {
                "status": "error",
                "error": "invalid_chat_request",
                "details": format_chat_validation_error(error),
            }

        from videodb import connect

        try:
            media_state = resolve_media_reference(
                connect,
                base_url=os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io"),
                collection_id=request.collection_id,
                video_id=request.video_id,
            )
        except MediaReferenceError as error:
            return {
                "status": "error",
                "error": "invalid_media_reference",
                "code": error.code,
                "message": error.message,
            }

        chat_handler = ChatHandler(
            db=load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
        )
        return chat_handler.chat(request.model_dump(), media_state=media_state)
