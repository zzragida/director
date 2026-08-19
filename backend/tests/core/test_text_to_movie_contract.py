import json

import pytest

from director.core.text_to_movie_contract import (
    StructuredGenerationError,
    parse_scene_sequence_response,
    parse_visual_style_response,
)


class Response:
    def __init__(self, content="", status=True):
        self.content = content
        self.status = status


def valid_visual_style_payload():
    return {
        "camera_setup": "35mm cinema camera",
        "color_grading": "warm contrast",
        "lighting_style": "soft directional",
        "movement_style": "slow dolly",
        "film_mood": "hopeful",
        "director_reference": "restrained cinematic realism",
        "character_constants": {
            "physical_description": "short dark hair",
            "costume_details": "navy jacket",
        },
        "setting_constants": {
            "time_period": "present day",
            "environment": "urban apartment",
        },
    }


def test_valid_visual_style_is_parsed_to_typed_model():
    style = parse_visual_style_response(
        Response(json.dumps(valid_visual_style_payload()))
    )

    assert style.camera_setup == "35mm cinema camera"
    assert style.character_constants.physical_description == "short dark hair"
    assert style.setting_constants.environment == "urban apartment"


def test_visual_style_llm_failure_is_distinct_from_invalid_json():
    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_visual_style_response(Response("provider error", status=False))

    assert exc_info.value.stage == "visual_style"
    assert exc_info.value.code == "llm_failed"
    assert exc_info.value.details == []


def test_visual_style_invalid_json_is_typed_without_echoing_content():
    secret_value = "not-json-sensitive-value"

    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_visual_style_response(Response(secret_value))

    error = exc_info.value
    assert error.stage == "visual_style"
    assert error.code == "invalid_json"
    assert secret_value not in str(error.details)
    assert secret_value not in error.message


def test_visual_style_missing_nested_field_is_rejected():
    payload = valid_visual_style_payload()
    del payload["character_constants"]["costume_details"]

    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_visual_style_response(Response(json.dumps(payload)))

    error = exc_info.value
    assert error.code == "invalid_structure"
    assert any(
        detail["field"] == "character_constants.costume_details"
        for detail in error.details
    )


def test_visual_style_blank_required_text_is_rejected():
    payload = valid_visual_style_payload()
    payload["film_mood"] = "   "

    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_visual_style_response(Response(json.dumps(payload)))

    assert exc_info.value.code == "invalid_structure"
    assert any(detail["field"] == "film_mood" for detail in exc_info.value.details)


def test_valid_scene_sequence_preserves_positive_integer_duration():
    response = Response(
        json.dumps(
            {
                "scenes": [
                    {
                        "story_beat": "The character enters the room",
                        "scene_description": "A wide shot of the quiet room",
                        "suggested_duration": 4,
                    }
                ]
            }
        )
    )

    sequence = parse_scene_sequence_response(response)

    assert len(sequence.scenes) == 1
    assert sequence.scenes[0].suggested_duration == 4


def test_empty_scene_list_is_rejected():
    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_scene_sequence_response(Response('{"scenes": []}'))

    assert exc_info.value.stage == "scene_sequence"
    assert exc_info.value.code == "invalid_structure"
    assert any(detail["field"] == "scenes" for detail in exc_info.value.details)


def test_scene_duration_string_is_not_silently_coerced():
    response = Response(
        json.dumps(
            {
                "scenes": [
                    {
                        "story_beat": "Beat",
                        "scene_description": "Description",
                        "suggested_duration": "5",
                    }
                ]
            }
        )
    )

    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_scene_sequence_response(response)

    assert exc_info.value.code == "invalid_structure"
    assert any(
        detail["field"] == "scenes.0.suggested_duration"
        for detail in exc_info.value.details
    )


def test_non_positive_scene_duration_is_rejected():
    response = Response(
        json.dumps(
            {
                "scenes": [
                    {
                        "story_beat": "Beat",
                        "scene_description": "Description",
                        "suggested_duration": 0,
                    }
                ]
            }
        )
    )

    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_scene_sequence_response(response)

    assert exc_info.value.code == "invalid_structure"


def test_scene_response_must_be_json_object():
    with pytest.raises(StructuredGenerationError) as exc_info:
        parse_scene_sequence_response(Response("[]"))

    assert exc_info.value.code == "invalid_structure"
    assert exc_info.value.details[0]["field"] == "$"
