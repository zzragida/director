from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


ChatContent = Union[str, List[dict]]


class ChatRequest(BaseModel):
    """Validated boundary model for Socket.IO chat requests.

    The contract intentionally keeps optional media and context fields compatible
    with the existing client while requiring the identifiers and content needed
    to create a conversation message safely.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str = Field(min_length=1)
    conv_id: str = Field(min_length=1)
    content: ChatContent
    collection_id: Optional[str] = None
    video_id: Optional[str] = None
    agents: List[str] = Field(default_factory=list)
    edited_context: Optional[dict] = None

    @field_validator("session_id", "conv_id")
    @classmethod
    def validate_required_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("collection_id", "video_id")
    @classmethod
    def normalize_optional_identifier(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: ChatContent) -> ChatContent:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("content must not be blank")
            return value
        if not value:
            raise ValueError("content must not be empty")
        return value

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, value: List[str]) -> List[str]:
        normalized = []
        for agent_name in value:
            name = agent_name.strip()
            if not name:
                raise ValueError("agent names must not be blank")
            normalized.append(name)
        return normalized


def format_chat_validation_error(error: ValidationError) -> List[dict]:
    """Return validation details without echoing request payload values."""

    details = []
    for item in error.errors(include_input=False):
        details.append(
            {
                "field": ".".join(str(part) for part in item.get("loc", ())),
                "message": item.get("msg", "invalid value"),
                "type": item.get("type", "value_error"),
            }
        )
    return details
