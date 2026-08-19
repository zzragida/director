import pytest
from pydantic import ValidationError

from director.core.chat_request import ChatRequest, format_chat_validation_error


def valid_payload(**overrides):
    payload = {
        "session_id": "session-1",
        "conv_id": "conv-1",
        "content": "summarize this video",
        "collection_id": "collection-1",
        "video_id": "video-1",
        "agents": ["summarize_video"],
    }
    payload.update(overrides)
    return payload


def test_valid_request_preserves_existing_chat_fields():
    request = ChatRequest.model_validate(valid_payload(edited_context={"reasoning": []}))

    assert request.session_id == "session-1"
    assert request.conv_id == "conv-1"
    assert request.content == "summarize this video"
    assert request.collection_id == "collection-1"
    assert request.video_id == "video-1"
    assert request.agents == ["summarize_video"]
    assert request.edited_context == {"reasoning": []}


def test_extra_fields_remain_available_for_backward_compatible_extensions():
    request = ChatRequest.model_validate(valid_payload(client_trace_id="trace-1"))

    assert request.model_dump()["client_trace_id"] == "trace-1"


@pytest.mark.parametrize("field", ["session_id", "conv_id"])
def test_required_identifiers_reject_blank_values(field):
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(valid_payload(**{field: "   "}))


@pytest.mark.parametrize("content", ["", "   ", []])
def test_content_rejects_empty_values(content):
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(valid_payload(content=content))


def test_optional_media_identifiers_normalize_blank_to_none():
    request = ChatRequest.model_validate(
        valid_payload(collection_id="   ", video_id="")
    )

    assert request.collection_id is None
    assert request.video_id is None


def test_agents_reject_blank_names():
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(valid_payload(agents=["summarize_video", " "]))


def test_validation_error_details_do_not_echo_input_values():
    secret_value = "do-not-echo-this-value"

    with pytest.raises(ValidationError) as exc_info:
        ChatRequest.model_validate(valid_payload(session_id=" ", content=secret_value))

    details = format_chat_validation_error(exc_info.value)
    rendered = repr(details)

    assert "session_id" in rendered
    assert secret_value not in rendered
