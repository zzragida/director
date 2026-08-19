import pytest

from director.core.tool_contract import (
    ToolArgumentValidationError,
    get_effective_tool_schema,
    validate_tool_arguments,
)


def test_required_nested_enum_and_types_are_validated_without_coercion():
    schema = {
        "type": "object",
        "properties": {
            "collection_id": {"type": "string"},
            "engine": {"type": "string", "enum": ["videodb", "kling"]},
            "config": {
                "type": "object",
                "properties": {
                    "duration": {"type": "integer"},
                },
                "required": ["duration"],
            },
        },
        "required": ["collection_id", "engine", "config"],
    }

    with pytest.raises(ToolArgumentValidationError) as exc_info:
        validate_tool_arguments(
            schema,
            {
                "collection_id": "collection-1",
                "engine": "unknown",
                "config": {"duration": "5"},
            },
        )

    details = exc_info.value.details
    assert {item["field"] for item in details} == {"engine", "config.duration"}
    assert {item["code"] for item in details} == {"invalid_enum", "type_mismatch"}


def test_validation_details_do_not_echo_argument_values():
    secret_value = "private-storyline-value"
    schema = {
        "type": "object",
        "properties": {"storyline": {"type": "integer"}},
        "required": ["storyline"],
    }

    with pytest.raises(ToolArgumentValidationError) as exc_info:
        validate_tool_arguments(schema, {"storyline": secret_value})

    assert secret_value not in str(exc_info.value.details)


def test_boolean_is_not_accepted_as_integer():
    schema = {
        "type": "object",
        "properties": {"duration": {"type": "integer"}},
        "required": ["duration"],
    }

    with pytest.raises(ToolArgumentValidationError) as exc_info:
        validate_tool_arguments(schema, {"duration": True})

    assert exc_info.value.details[0]["code"] == "type_mismatch"


def test_text_to_movie_effective_schema_repairs_runtime_required_contract():
    declared_schema = {
        "type": "object",
        "properties": {
            "job_type": {"type": "string", "enum": ["text_to_movie"]},
            "collection_id": {"type": "string"},
            "engine": {"type": "string"},
            "text_to_movie": {
                "type": "object",
                "properties": {"storyline": {"type": "string"}},
                "required": ["storyline"],
            },
        },
        "required": ["job_type", "collection_id", "engine"],
    }

    effective = get_effective_tool_schema("text_to_movie", declared_schema)

    assert "text_to_movie" in effective["required"]
    assert (
        effective["properties"]["text_to_movie"]["properties"]["storyline"]["minLength"]
        == 1
    )
    assert "text_to_movie" not in declared_schema["required"]

    with pytest.raises(ToolArgumentValidationError) as exc_info:
        validate_tool_arguments(
            effective,
            {
                "job_type": "text_to_movie",
                "collection_id": "collection-1",
                "engine": "videodb",
            },
        )

    assert exc_info.value.details[0]["field"] == "text_to_movie"


def test_valid_nested_arguments_pass_unchanged():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "minLength": 1}},
                    "required": ["name"],
                },
            }
        },
        "required": ["items"],
    }
    arguments = {"items": [{"name": "clip-a"}]}

    assert validate_tool_arguments(schema, arguments) is arguments
