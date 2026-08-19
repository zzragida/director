from copy import deepcopy
from typing import Any, Dict, List


class ToolArgumentValidationError(ValueError):
    """Raised when LLM tool arguments do not satisfy an agent parameter schema."""

    code = "invalid_tool_arguments"

    def __init__(self, details: List[dict]):
        super().__init__("Invalid agent tool arguments")
        self.details = details


# Known schema/runtime mismatches are patched here so the same effective contract
# is advertised to the LLM and enforced immediately before execution.
RUNTIME_SCHEMA_PATCHES = {
    "text_to_movie": {
        "required": ["text_to_movie"],
        "min_length_paths": [("text_to_movie", "storyline")],
    }
}


def get_effective_tool_schema(agent_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    effective = deepcopy(schema or {"type": "object"})
    patch = RUNTIME_SCHEMA_PATCHES.get(agent_name)
    if not patch:
        return effective

    required = effective.setdefault("required", [])
    for field_name in patch.get("required", []):
        if field_name not in required:
            required.append(field_name)

    for path in patch.get("min_length_paths", []):
        current = effective
        for field_name in path:
            current = current.setdefault("properties", {}).setdefault(field_name, {})
        current.setdefault("minLength", 1)

    return effective


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _path(parts: List[str]) -> str:
    return ".".join(parts) if parts else "$"


def _validate(schema: Dict[str, Any], value: Any, path: List[str], details: List[dict]):
    expected_type = schema.get("type")
    if expected_type and not _matches_type(value, expected_type):
        details.append(
            {
                "field": _path(path),
                "code": "type_mismatch",
                "message": f"expected {expected_type}",
            }
        )
        return

    allowed_values = schema.get("enum")
    if allowed_values is not None and value not in allowed_values:
        details.append(
            {
                "field": _path(path),
                "code": "invalid_enum",
                "message": "value is not one of the allowed options",
            }
        )

    if expected_type == "string" and isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            details.append(
                {
                    "field": _path(path),
                    "code": "string_too_short",
                    "message": f"minimum length is {min_length}",
                }
            )

    if expected_type == "array" and isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            details.append(
                {
                    "field": _path(path),
                    "code": "array_too_short",
                    "message": f"minimum item count is {min_items}",
                }
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item_schema, item, path + [str(index)], details)

    if expected_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        for required_name in schema.get("required", []):
            if required_name not in value:
                details.append(
                    {
                        "field": _path(path + [required_name]),
                        "code": "required",
                        "message": "field is required",
                    }
                )

        for name, child_schema in properties.items():
            if name in value and isinstance(child_schema, dict):
                _validate(child_schema, value[name], path + [name], details)


def validate_tool_arguments(schema: Dict[str, Any], arguments: Any) -> dict:
    """Validate tool arguments against the JSON-schema subset used by agents.

    Validation is intentionally strict and non-coercing. It checks structural
    correctness before an agent executes and never includes input values in
    validation details.
    """

    details: List[dict] = []
    root_schema = schema or {"type": "object"}
    _validate(root_schema, arguments, [], details)
    if details:
        raise ToolArgumentValidationError(details)
    return arguments
