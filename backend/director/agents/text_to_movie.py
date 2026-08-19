import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from videodb.asset import AudioAsset, VideoAsset

from director.agents.base import AgentResponse, AgentStatus, BaseAgent
from director.constants import DOWNLOADS_PATH
from director.core.generation_lifecycle import (
    GenerationOperationState,
    begin_resume,
    begin_submission,
    make_artifact_name,
    record_failure,
    record_materialized,
    record_persisted,
    record_provider_request,
)
from director.core.generation_reconciliation import (
    ReconciliationAction,
    classify_generation_operation,
    safe_reconciliation_summary,
)
from director.core.session import (
    ContextMessage,
    MsgStatus,
    RoleTypes,
    Session,
    VideoContent,
    VideoData,
)
from director.core.text_to_movie_checkpoint import (
    TextToMovieCheckpoint,
    TextToMovieCheckpointStore,
    TextToMovieExecutionError,
    build_request_fingerprint,
    compact_media,
    create_checkpoint,
    make_checkpoint_id,
)
from director.core.text_to_movie_contract import (
    StructuredGenerationError,
    VisualStyle,
    parse_scene_sequence_response,
    parse_visual_style_response,
)
from director.llm import get_default_llm
from director.tools.elevenlabs import (
    ElevenLabsTool,
    PARAMS_CONFIG as ELEVENLABS_PARAMS_CONFIG,
)
from director.tools.kling import KlingAITool, PARAMS_CONFIG as KLING_PARAMS_CONFIG
from director.tools.stabilityai import (
    StabilityAITool,
    PARAMS_CONFIG as STABILITYAI_PARAMS_CONFIG,
)
from director.tools.videodb_tool import (
    VDBAudioGenerationTool,
    VDBVideoGenerationTool,
    VideoDBTool,
)


logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ["stabilityai", "kling", "videodb"]
SUPPORTED_AUDIO_ENGINES = ["elevenlabs", "videodb"]
TEXT_TO_MOVIE_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "Collection ID to store the video",
        },
        "engine": {
            "type": "string",
            "description": "The video generation engine to use",
            "enum": SUPPORTED_ENGINES,
            "default": "videodb",
        },
        "audio_engine": {
            "type": "string",
            "description": "The audio generation engine to use",
            "enum": SUPPORTED_AUDIO_ENGINES,
            "default": "videodb",
        },
        "job_type": {
            "type": "string",
            "enum": ["text_to_movie"],
            "description": "The type of video generation to perform",
        },
        "text_to_movie": {
            "type": "object",
            "properties": {
                "storyline": {
                    "type": "string",
                    "description": "The storyline to generate the video",
                },
                "sound_effects_description": {
                    "type": "string",
                    "description": "Optional description for background music generation",
                    "default": None,
                },
                "video_stabilityai_config": {
                    "type": "object",
                    "description": "Optional configuration for StabilityAI engine",
                    "properties": STABILITYAI_PARAMS_CONFIG["text_to_video"],
                },
                "video_kling_config": {
                    "type": "object",
                    "description": "Optional configuration for Kling engine",
                    "properties": KLING_PARAMS_CONFIG["text_to_video"],
                },
                "audio_elevenlabs_config": {
                    "type": "object",
                    "description": "Optional configuration for ElevenLabs engine",
                    "properties": ELEVENLABS_PARAMS_CONFIG["sound_effect"],
                },
            },
            "required": ["storyline"],
        },
    },
    "required": ["job_type", "collection_id", "engine"],
}


@dataclass
class EngineConfig:
    name: str
    max_duration: int
    preferred_style: str
    prompt_format: str


class TextToMovieAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "text_to_movie"
        self.description = "Agent for generating movies from storylines using Gen AI models"
        self.parameters = TEXT_TO_MOVIE_AGENT_PARAMETERS
        self.llm = get_default_llm()
        self.engine_configs = {
            "kling": EngineConfig("kling", 10, "cinematic", "detailed"),
            "stabilityai": EngineConfig(
                "stabilityai", 4, "photorealistic", "concise"
            ),
            "videodb": EngineConfig("videodb", 6, "cinematic", "detailed"),
        }
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        engine: str = "stabilityai",
        audio_engine: str = "videodb",
        job_type: str = "text_to_movie",
        text_to_movie: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        video_content = None
        checkpoint = None
        checkpoint_store = TextToMovieCheckpointStore(self.session)

        try:
            raw_storyline = self._validate_input(job_type, text_to_movie)
            self._validate_engine_names(engine, audio_engine)

            self.video_gen_config_key = (
                "video_stabilityai_config"
                if engine == "stabilityai"
                else "video_kling_config"
            )
            self.audio_gen_config_key = "audio_elevenlabs_config"
            video_gen_config = text_to_movie.get(self.video_gen_config_key, {})
            audio_gen_config = (
                {}
                if engine == "videodb"
                else text_to_movie.get(self.audio_gen_config_key, {})
            )

            request_fingerprint = build_request_fingerprint(
                collection_id=collection_id,
                engine=engine,
                audio_engine=audio_engine,
                storyline=raw_storyline,
                video_config=video_gen_config,
                audio_config=audio_gen_config,
            )
            checkpoint = checkpoint_store.get(
                make_checkpoint_id(request_fingerprint)
            )

            self.output_message.actions.append("Processing input...")
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Generating movie...",
            )
            self.output_message.content.append(video_content)
            self.output_message.push_update()

            if (
                checkpoint is not None
                and checkpoint.status == "complete"
                and checkpoint.final_video
            ):
                return self._completed_checkpoint_response(
                    checkpoint,
                    video_content,
                )

            self._validate_provider_configuration(engine, audio_engine)

            if checkpoint is None:
                visual_style = self.generate_visual_style(raw_storyline)
                scenes = self.generate_scene_sequence(
                    raw_storyline,
                    visual_style,
                    engine,
                )
                checkpoint = create_checkpoint(
                    request_fingerprint=request_fingerprint,
                    visual_style=visual_style.model_dump(mode="json"),
                    scenes=scenes,
                    video_provider=engine,
                    audio_provider=audio_engine,
                )
                checkpoint_store.save(checkpoint)
            else:
                visual_style = VisualStyle.model_validate(checkpoint.visual_style)
                scenes = [dict(scene.plan) for scene in checkpoint.scenes]
                if checkpoint.ensure_lifecycle(
                    video_provider=engine,
                    audio_provider=audio_engine,
                ):
                    checkpoint_store.save(checkpoint)

            self._initialize_media_tools(collection_id, engine, audio_engine)
            total_duration = self._resume_or_generate_scenes(
                checkpoint=checkpoint,
                checkpoint_store=checkpoint_store,
                scenes=scenes,
                visual_style=visual_style,
                engine=engine,
                video_gen_config=video_gen_config,
            )
            sound_effects_media = self._resume_or_generate_audio(
                checkpoint=checkpoint,
                checkpoint_store=checkpoint_store,
                storyline=raw_storyline,
                total_duration=total_duration,
                audio_gen_config=audio_gen_config,
            )

            self.output_message.actions.append("Combining assets into final video...")
            self.output_message.push_update()
            try:
                final_video = self.combine_assets(scenes, sound_effects_media)
            except Exception as exc:
                raise TextToMovieExecutionError(
                    stage="combine",
                    code="combine_failed",
                    message="Unable to combine generated assets.",
                ) from exc

            checkpoint.status = "complete"
            checkpoint.final_video = final_video
            checkpoint.failure_stage = None
            checkpoint.failure_code = None
            checkpoint.failed_scene_index = None
            checkpoint_store.save(checkpoint)

            video_content.video = VideoData(stream_url=final_video)
            video_content.status = MsgStatus.success
            video_content.status_message = "Movie generation complete"
            self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message="Movie generated successfully",
                data={
                    "video_url": final_video,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "generation_run_id": checkpoint.generation_run_id,
                    "resumed": any(
                        scene.operation is not None
                        and scene.operation.attempt_count > 1
                        for scene in checkpoint.scenes
                    ),
                },
            )

        except StructuredGenerationError as error:
            logger.warning(
                "Structured text-to-movie generation failed at %s: %s",
                error.stage,
                error.code,
            )
            if video_content is not None:
                video_content.status = MsgStatus.error
                video_content.status_message = "Unable to create a valid movie plan"
                self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Movie planning failed validation.",
                data={
                    "error": "structured_generation_failed",
                    "stage": error.stage,
                    "code": error.code,
                    "details": error.details,
                },
            )
        except TextToMovieExecutionError as error:
            logger.warning(
                "Text-to-movie execution paused at %s: %s",
                error.stage,
                error.code,
            )
            if checkpoint is not None:
                checkpoint.status = "failed"
                checkpoint.failure_stage = error.stage
                checkpoint.failure_code = error.code
                checkpoint.failed_scene_index = error.scene_index
                checkpoint_store.save(checkpoint)

            if video_content is not None:
                video_content.status = MsgStatus.error
                video_content.status_message = (
                    "Movie generation requires reconciliation"
                    if error.stage == "reconciliation"
                    else "Movie generation paused and can be resumed"
                )
                self.output_message.publish()

            data = {
                "error": "text_to_movie_partial_failure",
                "stage": error.stage,
                "code": error.code,
                "resumable": bool(error.resumable and checkpoint is not None),
                "completed_scenes": (
                    checkpoint.completed_scene_count if checkpoint else 0
                ),
                "total_scenes": len(checkpoint.scenes) if checkpoint else 0,
            }
            if checkpoint is not None:
                data["checkpoint_id"] = checkpoint.checkpoint_id
                data["generation_run_id"] = checkpoint.generation_run_id
            if error.scene_index is not None:
                data["failed_scene_index"] = error.scene_index
            if error.operation_id is not None:
                data["operation_id"] = error.operation_id
            if error.reconciliation is not None:
                data["reconciliation"] = error.reconciliation

            return AgentResponse(
                status=AgentStatus.ERROR,
                message=(
                    "Movie generation is paused for reconciliation."
                    if error.stage == "reconciliation"
                    else "Movie generation paused after a recoverable pipeline failure."
                ),
                data=data,
            )
        except Exception:
            logger.exception("Unexpected error in %s agent", self.agent_name)
            if video_content is not None:
                video_content.status = MsgStatus.error
                video_content.status_message = "Error generating movie"
                self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Movie generation failed unexpectedly.",
                data={"error": "text_to_movie_failed"},
            )

    def _validate_input(self, job_type: str, text_to_movie: Optional[dict]) -> str:
        if job_type != "text_to_movie":
            raise ValueError(f"Unsupported job type: {job_type}")
        if not isinstance(text_to_movie, dict):
            raise StructuredGenerationError(
                stage="input",
                code="invalid_storyline",
                message="A text_to_movie payload with a storyline is required.",
                details=[
                    {
                        "field": "text_to_movie",
                        "code": "object_required",
                        "message": "expected object",
                    }
                ],
            )
        storyline = text_to_movie.get("storyline")
        if not isinstance(storyline, str) or not storyline.strip():
            raise StructuredGenerationError(
                stage="input",
                code="invalid_storyline",
                message="A non-empty storyline is required.",
                details=[
                    {
                        "field": "text_to_movie.storyline",
                        "code": "non_blank_required",
                        "message": "must not be blank",
                    }
                ],
            )
        return storyline.strip()

    def _validate_engine_names(self, engine: str, audio_engine: str) -> None:
        if engine not in self.engine_configs:
            raise ValueError(f"Unsupported engine: {engine}")
        if audio_engine not in SUPPORTED_AUDIO_ENGINES:
            raise ValueError(f"Unsupported audio engine: {audio_engine}")

    def _completed_checkpoint_response(
        self,
        checkpoint: TextToMovieCheckpoint,
        video_content: VideoContent,
    ) -> AgentResponse:
        video_content.video = VideoData(stream_url=checkpoint.final_video)
        video_content.status = MsgStatus.success
        video_content.status_message = "Movie generation complete"
        self.output_message.publish()
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Movie already generated; resumed from completed checkpoint.",
            data={
                "video_url": checkpoint.final_video,
                "checkpoint_id": checkpoint.checkpoint_id,
                "generation_run_id": checkpoint.generation_run_id,
                "resumed": True,
            },
        )

    def _validate_provider_configuration(self, engine: str, audio_engine: str) -> None:
        if engine == "stabilityai" and not os.getenv("STABILITYAI_API_KEY"):
            raise TextToMovieExecutionError(
                stage="initialization",
                code="video_provider_not_configured",
                message="The selected video provider is not configured.",
                resumable=False,
            )
        if engine == "kling" and (
            not os.getenv("KLING_AI_ACCESS_API_KEY")
            or not os.getenv("KLING_AI_SECRET_API_KEY")
        ):
            raise TextToMovieExecutionError(
                stage="initialization",
                code="video_provider_not_configured",
                message="The selected video provider is not configured.",
                resumable=False,
            )
        if audio_engine == "elevenlabs" and not os.getenv("ELEVENLABS_API_KEY"):
            raise TextToMovieExecutionError(
                stage="initialization",
                code="audio_provider_not_configured",
                message="The selected audio provider is not configured.",
                resumable=False,
            )

    def _initialize_media_tools(
        self,
        collection_id: str,
        engine: str,
        audio_engine: str,
    ) -> None:
        self.videodb_tool = VideoDBTool(collection_id=collection_id)
        if engine == "stabilityai":
            self.video_gen_tool = StabilityAITool(
                api_key=os.getenv("STABILITYAI_API_KEY")
            )
        elif engine == "kling":
            self.video_gen_tool = KlingAITool(
                access_key=os.getenv("KLING_AI_ACCESS_API_KEY"),
                secret_key=os.getenv("KLING_AI_SECRET_API_KEY"),
            )
        else:
            self.video_gen_tool = VDBVideoGenerationTool(
                collection_id=collection_id
            )

        if audio_engine == "elevenlabs":
            self.audio_gen_tool = ElevenLabsTool(
                api_key=os.getenv("ELEVENLABS_API_KEY")
            )
        else:
            self.audio_gen_tool = VDBAudioGenerationTool(
                collection_id=collection_id
            )

    def _find_durable_artifact(self, operation, media_type: str):
        if operation.state not in (
            GenerationOperationState.materialized,
            GenerationOperationState.failed,
        ):
            return None
        if (
            operation.state == GenerationOperationState.failed
            and operation.last_error_code not in {"scene_upload_failed", "audio_upload_failed"}
        ):
            return None

        artifact_name = make_artifact_name(operation.operation_id, media_type)
        try:
            candidates = (
                self.videodb_tool.get_videos()
                if media_type == "video"
                else self.videodb_tool.get_audios()
            )
        except Exception as exc:
            raise TextToMovieExecutionError(
                stage="reconciliation",
                code="artifact_lookup_failed",
                message="Unable to reconcile durable media state.",
                resumable=False,
                operation_id=operation.operation_id,
            ) from exc

        for candidate in candidates:
            if candidate.get("name") == artifact_name:
                return candidate
        return None

    def _reconcile_operation(
        self,
        *,
        checkpoint: TextToMovieCheckpoint,
        checkpoint_store: TextToMovieCheckpointStore,
        operation,
        media_type: str,
        provider_supports_resume: bool,
        scene_index: Optional[int] = None,
    ):
        durable = self._find_durable_artifact(operation, media_type)
        decision = classify_generation_operation(
            operation,
            provider_supports_resume=provider_supports_resume,
            durable_artifact_found=durable is not None,
        )

        if durable is not None:
            try:
                media = compact_media(durable)
            except ValueError as exc:
                raise TextToMovieExecutionError(
                    stage="reconciliation",
                    code="invalid_reconciled_artifact",
                    message="A discovered durable artifact is invalid.",
                    resumable=False,
                    scene_index=scene_index,
                    operation_id=operation.operation_id,
                    reconciliation=safe_reconciliation_summary(decision),
                ) from exc
            record_persisted(operation, media)
            checkpoint_store.save(checkpoint)
            return media

        if decision.action in {
            ReconciliationAction.retry_submission,
            ReconciliationAction.resume_provider,
        }:
            return None

        raise TextToMovieExecutionError(
            stage="reconciliation",
            code=decision.reason_code,
            message="Generation operation requires reconciliation before retry.",
            resumable=False,
            scene_index=scene_index,
            operation_id=operation.operation_id,
            reconciliation=safe_reconciliation_summary(decision),
        )

    def _resume_or_generate_scenes(
        self,
        *,
        checkpoint: TextToMovieCheckpoint,
        checkpoint_store: TextToMovieCheckpointStore,
        scenes: List[dict],
        visual_style: VisualStyle,
        engine: str,
        video_gen_config: dict,
    ) -> float:
        self.output_message.actions.append(
            f"Preparing {len(checkpoint.scenes)} scene videos..."
        )
        self.output_message.push_update()
        total_duration = 0.0
        engine_config = self.engine_configs[engine]

        for scene_checkpoint in checkpoint.scenes:
            index = scene_checkpoint.index
            scene = scenes[index]
            operation = scene_checkpoint.operation
            if operation is None:
                raise TextToMovieExecutionError(
                    stage="scene_generation",
                    code="operation_state_missing",
                    message="Scene operation state is unavailable.",
                    scene_index=index,
                )

            if scene_checkpoint.status == "complete" and scene_checkpoint.media:
                scene["video"] = scene_checkpoint.media
                total_duration += float(scene_checkpoint.media.get("length", 0) or 0)
                continue

            supports_resume = callable(
                getattr(self.video_gen_tool, "resume_text_to_video", None)
            )
            reconciled_media = self._reconcile_operation(
                checkpoint=checkpoint,
                checkpoint_store=checkpoint_store,
                operation=operation,
                media_type="video",
                provider_supports_resume=supports_resume,
                scene_index=index,
            )
            if reconciled_media is not None:
                scene_checkpoint.media = reconciled_media
                scene_checkpoint.status = "complete"
                scene["video"] = reconciled_media
                checkpoint_store.save(checkpoint)
                total_duration += float(reconciled_media.get("length", 0) or 0)
                continue

            self.output_message.actions.append(
                f"Generating video for scene {index + 1}..."
            )
            self.output_message.push_update()

            if not scene_checkpoint.prompt:
                try:
                    scene_checkpoint.prompt = self.generate_engine_prompt(
                        scene,
                        visual_style,
                        engine,
                    )
                    checkpoint_store.save(checkpoint)
                except Exception as exc:
                    raise TextToMovieExecutionError(
                        stage="scene_prompt",
                        code="scene_prompt_failed",
                        message="Unable to prepare the scene generation prompt.",
                        scene_index=index,
                        operation_id=operation.operation_id,
                    ) from exc

            suggested_duration = min(
                scene["suggested_duration"],
                engine_config.max_duration,
            )
            video_path = f"{DOWNLOADS_PATH}/{uuid.uuid4()}.mp4"
            os.makedirs(DOWNLOADS_PATH, exist_ok=True)

            try:
                try:
                    video = self._execute_scene_generation(
                        checkpoint=checkpoint,
                        checkpoint_store=checkpoint_store,
                        operation=operation,
                        prompt=scene_checkpoint.prompt,
                        video_path=video_path,
                        duration=suggested_duration,
                        video_gen_config=video_gen_config,
                    )
                except TextToMovieExecutionError:
                    raise
                except Exception as exc:
                    record_failure(
                        operation,
                        code="scene_generation_failed",
                    )
                    checkpoint_store.save(checkpoint)
                    raise TextToMovieExecutionError(
                        stage="scene_generation",
                        code="scene_generation_failed",
                        message="Unable to generate a scene video.",
                        scene_index=index,
                        operation_id=operation.operation_id,
                    ) from exc

                if video is None:
                    record_materialized(operation)
                    checkpoint_store.save(checkpoint)
                    self.output_message.actions.append(
                        f"Uploading video for scene {index + 1}..."
                    )
                    self.output_message.push_update()
                    try:
                        video = self.videodb_tool.upload(
                            video_path,
                            source_type="file_path",
                            media_type="video",
                            name=make_artifact_name(operation.operation_id, "video"),
                        )
                    except Exception as exc:
                        record_failure(
                            operation,
                            code="scene_upload_failed",
                        )
                        checkpoint_store.save(checkpoint)
                        raise TextToMovieExecutionError(
                            stage="scene_upload",
                            code="scene_upload_failed",
                            message="Unable to persist a generated scene video.",
                            scene_index=index,
                            operation_id=operation.operation_id,
                        ) from exc

                try:
                    media = compact_media(video)
                except ValueError as exc:
                    record_failure(
                        operation,
                        code="invalid_scene_media",
                    )
                    checkpoint_store.save(checkpoint)
                    raise TextToMovieExecutionError(
                        stage="scene_generation",
                        code="invalid_scene_media",
                        message="The scene provider returned an invalid media result.",
                        scene_index=index,
                        operation_id=operation.operation_id,
                    ) from exc

                record_persisted(operation, media)
                scene_checkpoint.media = media
                scene_checkpoint.status = "complete"
                checkpoint.status = "generating"
                checkpoint.failure_stage = None
                checkpoint.failure_code = None
                checkpoint.failed_scene_index = None
                checkpoint_store.save(checkpoint)
                scene["video"] = media
                total_duration += float(media.get("length", 0) or 0)
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

        return total_duration

    def _execute_scene_generation(
        self,
        *,
        checkpoint: TextToMovieCheckpoint,
        checkpoint_store: TextToMovieCheckpointStore,
        operation,
        prompt: str,
        video_path: str,
        duration: float,
        video_gen_config: dict,
    ):
        supports_resume = callable(
            getattr(self.video_gen_tool, "resume_text_to_video", None)
        )
        if operation.provider_request_id and supports_resume:
            begin_resume(operation)
            checkpoint_store.save(checkpoint)
            return self.video_gen_tool.resume_text_to_video(
                operation.provider_request_id,
                video_path,
            )

        begin_submission(operation)
        checkpoint_store.save(checkpoint)

        if supports_resume:
            def persist_request_id(request_id):
                record_provider_request(operation, request_id)
                checkpoint_store.save(checkpoint)

            return self.video_gen_tool.text_to_video(
                prompt=prompt,
                save_at=video_path,
                duration=duration,
                config=video_gen_config,
                on_request_id=persist_request_id,
            )

        return self.video_gen_tool.text_to_video(
            prompt=prompt,
            save_at=video_path,
            duration=duration,
            config=video_gen_config,
        )

    def _resume_or_generate_audio(
        self,
        *,
        checkpoint: TextToMovieCheckpoint,
        checkpoint_store: TextToMovieCheckpointStore,
        storyline: str,
        total_duration: float,
        audio_gen_config: dict,
    ) -> dict:
        if checkpoint.audio_media:
            return checkpoint.audio_media

        operation = checkpoint.audio_operation
        if operation is None:
            raise TextToMovieExecutionError(
                stage="audio_generation",
                code="operation_state_missing",
                message="Audio operation state is unavailable.",
            )

        reconciled_media = self._reconcile_operation(
            checkpoint=checkpoint,
            checkpoint_store=checkpoint_store,
            operation=operation,
            media_type="audio",
            provider_supports_resume=False,
        )
        if reconciled_media is not None:
            checkpoint.audio_media = reconciled_media
            checkpoint.status = "audio_complete"
            checkpoint_store.save(checkpoint)
            return reconciled_media

        if not checkpoint.audio_prompt:
            try:
                checkpoint.audio_prompt = self.generate_audio_prompt(storyline)
                checkpoint_store.save(checkpoint)
            except Exception as exc:
                raise TextToMovieExecutionError(
                    stage="audio_prompt",
                    code="audio_prompt_failed",
                    message="Unable to prepare the background audio prompt.",
                    operation_id=operation.operation_id,
                ) from exc

        self.output_message.actions.append("Generating background music...")
        self.output_message.push_update()
        os.makedirs(DOWNLOADS_PATH, exist_ok=True)
        sound_effects_path = f"{DOWNLOADS_PATH}/{uuid.uuid4()}.mp3"

        try:
            begin_submission(operation)
            checkpoint_store.save(checkpoint)
            try:
                sound_effects_media = self.audio_gen_tool.generate_sound_effect(
                    prompt=checkpoint.audio_prompt,
                    save_at=sound_effects_path,
                    duration=total_duration,
                    config=audio_gen_config,
                )
            except Exception as exc:
                record_failure(operation, code="audio_generation_failed")
                checkpoint_store.save(checkpoint)
                raise TextToMovieExecutionError(
                    stage="audio_generation",
                    code="audio_generation_failed",
                    message="Unable to generate background audio.",
                    operation_id=operation.operation_id,
                ) from exc

            if sound_effects_media is None:
                record_materialized(operation)
                checkpoint_store.save(checkpoint)
                self.output_message.actions.append(
                    "Uploading background music to VideoDB..."
                )
                self.output_message.push_update()
                try:
                    sound_effects_media = self.videodb_tool.upload(
                        sound_effects_path,
                        source_type="file_path",
                        media_type="audio",
                        name=make_artifact_name(operation.operation_id, "audio"),
                    )
                except Exception as exc:
                    record_failure(operation, code="audio_upload_failed")
                    checkpoint_store.save(checkpoint)
                    raise TextToMovieExecutionError(
                        stage="audio_upload",
                        code="audio_upload_failed",
                        message="Unable to persist generated background audio.",
                        operation_id=operation.operation_id,
                    ) from exc

            try:
                media = compact_media(sound_effects_media)
            except ValueError as exc:
                record_failure(operation, code="invalid_audio_media")
                checkpoint_store.save(checkpoint)
                raise TextToMovieExecutionError(
                    stage="audio_generation",
                    code="invalid_audio_media",
                    message="The audio provider returned an invalid media result.",
                    operation_id=operation.operation_id,
                ) from exc

            record_persisted(operation, media)
            checkpoint.audio_media = media
            checkpoint.status = "audio_complete"
            checkpoint.failure_stage = None
            checkpoint.failure_code = None
            checkpoint.failed_scene_index = None
            checkpoint_store.save(checkpoint)
            return media
        finally:
            if os.path.exists(sound_effects_path):
                os.remove(sound_effects_path)

    def generate_visual_style(self, storyline: str) -> VisualStyle:
        style_prompt = f"""
        As a cinematographer, define a consistent visual style for this short film:
        Storyline: {storyline}

        Return a JSON response with visual style parameters:
        {{
            "camera_setup": "Camera and lens combination",
            "color_grading": "Color grading style and palette",
            "lighting_style": "Core lighting approach",
            "movement_style": "Camera movement philosophy",
            "film_mood": "Overall atmospheric mood",
            "director_reference": "Key director's style to reference",
            "character_constants": {{
                "physical_description": "Consistent character details",
                "costume_details": "Consistent costume elements"
            }},
            "setting_constants": {{
                "time_period": "When this takes place",
                "environment": "Core setting elements that stay consistent"
            }}
        }}
        """
        style_message = ContextMessage(content=style_prompt, role=RoleTypes.user)
        llm_response = self.llm.chat_completions(
            [style_message.to_llm_msg()],
            response_format={"type": "json_object"},
        )
        return parse_visual_style_response(llm_response)

    def generate_scene_sequence(
        self,
        storyline: str,
        style: VisualStyle,
        engine: str,
    ) -> List[dict]:
        engine_config = self.engine_configs[engine]
        sequence_prompt = f"""
        Break this storyline into 3 distinct scenes maintaining visual consistency.
        Generate scene descriptions optimized for {engine} {engine_config.preferred_style} style.

        Visual Style:
        - Camera/Lens: {style.camera_setup}
        - Color Grade: {style.color_grading}
        - Lighting: {style.lighting_style}
        - Movement: {style.movement_style}
        - Mood: {style.film_mood}
        - Director Style: {style.director_reference}

        Character Constants:
        {json.dumps(style.character_constants.model_dump(), indent=2)}

        Setting Constants:
        {json.dumps(style.setting_constants.model_dump(), indent=2)}

        Maximum duration per scene: {engine_config.max_duration} seconds
        Storyline: {storyline}

        Return a JSON object with a non-empty `scenes` array. Every scene must contain:
        {{
            "story_beat": "What happens in this scene",
            "scene_description": "Visual description optimized for {engine}",
            "suggested_duration": "Positive integer duration in seconds"
        }}
        Make sure suggested_duration is a number, not a string.
        """
        sequence_message = ContextMessage(
            content=sequence_prompt,
            role=RoleTypes.user,
        )
        llm_response = self.llm.chat_completions(
            [sequence_message.to_llm_msg()],
            response_format={"type": "json_object"},
        )
        sequence = parse_scene_sequence_response(llm_response)
        return [scene.model_dump() for scene in sequence.scenes]

    def generate_engine_prompt(
        self,
        scene: dict,
        style: VisualStyle,
        engine: str,
    ) -> str:
        if engine == "stabilityai":
            return f"""
            {style.director_reference} style.
            {scene['scene_description']}.
            {style.character_constants.physical_description}.
            {style.lighting_style}, {style.color_grading}.
            Photorealistic, detailed, high quality, masterful composition.
            """.strip()

        initial_prompt = f"""
        {style.director_reference} style shot.
        Filmed on {style.camera_setup}.

        {scene['scene_description']}

        Character Details:
        {json.dumps(style.character_constants.model_dump(), indent=2)}

        Setting Elements:
        {json.dumps(style.setting_constants.model_dump(), indent=2)}

        {style.lighting_style} lighting.
        {style.color_grading} color palette.
        {style.movement_style} camera movement.
        Mood: {style.film_mood}
        """
        compression_prompt = f"""
        Compress the following prompt to under 2450 characters while maintaining its structure and key information:
        {initial_prompt}
        """
        compression_message = ContextMessage(
            content=compression_prompt,
            role=RoleTypes.user,
        )
        llm_response = self.llm.chat_completions(
            [compression_message.to_llm_msg()],
            response_format={"type": "text"},
        )
        if not getattr(llm_response, "status", False) or not getattr(
            llm_response,
            "content",
            "",
        ):
            raise ValueError("scene prompt generation failed")
        return llm_response.content

    def generate_audio_prompt(self, storyline: str) -> str:
        audio_prompt = f"""
        As a composer, create a simple musical description focusing ONLY on:
        - Main instrument/sound
        - One key mood change
        - Basic progression

        Keep it under 100 characters. No visual references or scene descriptions.
        Focus on the music.
        Story context: {storyline}
        """
        prompt_message = ContextMessage(
            content=audio_prompt,
            role=RoleTypes.user,
        )
        llm_response = self.llm.chat_completions(
            [prompt_message.to_llm_msg()],
            response_format={"type": "text"},
        )
        if not getattr(llm_response, "status", False) or not getattr(
            llm_response,
            "content",
            "",
        ):
            raise ValueError("audio prompt generation failed")
        return llm_response.content[:100]

    def combine_assets(
        self,
        scenes: List[dict],
        audio_media: Optional[dict],
    ) -> str:
        timeline = self.videodb_tool.get_and_set_timeline()
        for scene in scenes:
            timeline.add_inline(VideoAsset(asset_id=scene["video"]["id"]))
        if audio_media:
            timeline.add_overlay(
                0,
                AudioAsset(
                    asset_id=audio_media["id"],
                    start=0,
                    disable_other_tracks=True,
                ),
            )
        return timeline.generate_stream()
