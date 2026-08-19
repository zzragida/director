import json
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator


class StructuredGenerationError(ValueError):
    """Safe typed failure for structured LLM generation stages."""

    def __init__(self, *, stage: str, code: str, message: str, details: List[dict] = None):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.details = details or []


class CharacterConstants(BaseModel):
    model_config = ConfigDict(extra="ignore")

    physical_description: str
    costume_details: str

    @field_validator("physical_description", "costume_details")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class SettingConstants(BaseModel):
    model_config = ConfigDict(extra="ignore")

    time_period: str
    environment: str

    @field_validator("time_period", "environment")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class VisualStyle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    camera_setup: str
    color_grading: str
    lighting_style: str
    movement_style: str
    film_mood: str
    director_reference: str
    character_constants: CharacterConstants
    setting_constants: SettingConstants

    @field_validator(
        "camera_setup",
        "color_grading",
        "lighting_style",
        "movement_style",
        "film_mood",
        "director_reference",
    )
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Scene(BaseModel):
    model_config = ConfigDict(extra="ignore")

    story_beat: str
    scene_description: str
    suggested_duration: StrictInt = Field(gt=0)

    @field_validator("story_beat", "scene_description")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class SceneSequence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenes: List[Scene] = Field(min_length=1)


def _validation_details(error: ValidationError) -> List[dict]:
    details = []
    for item in error.errors(include_input=False):
        details.append(
            {
                "field": ".".join(str(part) for part in item.get("loc", ())),
                "code": item.get("type", "validation_error"),
                "message": item.get("msg", "invalid value"),
            }
        )
    return details


def _parse_response_json(response: Any, *, stage: str) -> Dict[str, Any]:
    if not getattr(response, "status", False):
        raise StructuredGenerationError(
            stage=stage,
            code="llm_failed",
            message=f"The {stage} generation request failed.",
        )

    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise StructuredGenerationError(
            stage=stage,
            code="empty_response",
            message=f"The {stage} generation response was empty.",
        )

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredGenerationError(
            stage=stage,
            code="invalid_json",
            message=f"The {stage} generation response was not valid JSON.",
        ) from exc

    if not isinstance(payload, dict):
        raise StructuredGenerationError(
            stage=stage,
            code="invalid_structure",
            message=f"The {stage} generation response must be a JSON object.",
            details=[
                {
                    "field": "$",
                    "code": "object_required",
                    "message": "expected object",
                }
            ],
        )
    return payload


def parse_visual_style_response(response: Any) -> VisualStyle:
    stage = "visual_style"
    payload = _parse_response_json(response, stage=stage)
    try:
        return VisualStyle.model_validate(payload)
    except ValidationError as exc:
        raise StructuredGenerationError(
            stage=stage,
            code="invalid_structure",
            message="The visual style response did not match the required structure.",
            details=_validation_details(exc),
        ) from exc


def parse_scene_sequence_response(response: Any) -> SceneSequence:
    stage = "scene_sequence"
    payload = _parse_response_json(response, stage=stage)
    try:
        return SceneSequence.model_validate(payload)
    except ValidationError as exc:
        raise StructuredGenerationError(
            stage=stage,
            code="invalid_structure",
            message="The scene sequence response did not match the required structure.",
            details=_validation_details(exc),
        ) from exc
