"""Minimal single-GPU HTTP service for LingBot-VLA v2 action inference."""

from __future__ import annotations

import base64
import binascii
import io
import math
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data import LingBotVlaV2Observation
from .pipeline import LingBotVlaV2CanonicalActionChunk
from .robot_profile import ROBOTWIN_CAMERA_KEYS
from .runtime import get_lingbot_vla_v2_pipeline


@dataclass(frozen=True)
class LingBotVlaV2ServiceConfig:
    """Configuration for one process-local LingBot-VLA v2 replica."""

    model_root: str
    qwen3vl_root: str
    device: str = "cuda:0"
    max_image_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")


class LingBotVlaV2ActionRequest(BaseModel):
    """One RobotWin observation encoded for the HTTP boundary."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1)
    state: list[float] = Field(min_length=14, max_length=14)
    camera_high: str = Field(min_length=1)
    camera_left_wrist: str = Field(min_length=1)
    camera_right_wrist: str = Field(min_length=1)
    seed: int | None = None

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        """Reject whitespace-only instructions."""
        value = value.strip()
        if not value:
            raise ValueError("task must be a non-empty string")
        return value

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: list[float]) -> list[float]:
        """Reject non-finite robot state values."""
        if not all(math.isfinite(item) for item in value):
            raise ValueError("state must contain only finite values")
        return value


class LingBotVlaV2ActionResponse(BaseModel):
    """Normalized canonical action chunk returned by the base checkpoint."""

    canonical_normalized_actions: list[list[float]]
    horizon: int
    action_dim: int
    checkpoint_variant: str
    policy_verified: bool
    verification_status: str


class LingBotVlaV2HealthResponse(BaseModel):
    """Readiness state for the process-local model replica."""

    status: str
    model: str
    device: str
    policy_verified: bool


class _Pipeline(Protocol):
    def __call__(
        self,
        observation: LingBotVlaV2Observation,
        seed: int | None = None,
    ) -> LingBotVlaV2CanonicalActionChunk: ...

    def close(self) -> None: ...


PipelineFactory = Callable[[LingBotVlaV2ServiceConfig], _Pipeline]


def _default_pipeline_factory(config: LingBotVlaV2ServiceConfig) -> _Pipeline:
    return get_lingbot_vla_v2_pipeline(config.model_root, config.qwen3vl_root, device=config.device, warmup=True)


def _decode_image(value: str, *, max_image_bytes: int) -> Image.Image:
    payload = value.strip()
    if payload.startswith("data:"):
        header, separator, payload = payload.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("image data URLs must use base64 encoding")
    max_encoded_length = 4 * ((max_image_bytes + 2) // 3)
    if len(payload) > max_encoded_length:
        raise ValueError(f"decoded image must not exceed {max_image_bytes} bytes")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image must be valid base64") from error
    if not decoded or len(decoded) > max_image_bytes:
        raise ValueError(f"decoded image must contain 1 to {max_image_bytes} bytes")
    try:
        with Image.open(io.BytesIO(decoded)) as image:
            return image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("decoded payload must be a supported image") from error


def predict_lingbot_vla_v2_action(
    pipeline: _Pipeline,
    request: LingBotVlaV2ActionRequest,
    *,
    max_image_bytes: int,
) -> LingBotVlaV2ActionResponse:
    """Decode one request and return the canonical normalized action chunk."""
    encoded_images = (request.camera_high, request.camera_left_wrist, request.camera_right_wrist)
    images = {
        key: _decode_image(value, max_image_bytes=max_image_bytes)
        for key, value in zip(ROBOTWIN_CAMERA_KEYS, encoded_images, strict=True)
    }
    observation = LingBotVlaV2Observation(task=request.task, state=request.state, images=images)
    chunk = pipeline(observation, seed=request.seed)
    return LingBotVlaV2ActionResponse(
        canonical_normalized_actions=chunk.canonical_normalized_actions.tolist(),
        horizon=chunk.horizon,
        action_dim=chunk.action_dim,
        checkpoint_variant=chunk.checkpoint_variant,
        policy_verified=chunk.policy_verified,
        verification_status=chunk.verification_status,
    )


class LingBotVlaV2Service:
    """Serialize requests through one loaded policy replica."""

    def __init__(self, pipeline: _Pipeline, config: LingBotVlaV2ServiceConfig) -> None:
        self.pipeline = pipeline
        self.config = config
        self._inference_lock = threading.Lock()

    def predict(self, request: LingBotVlaV2ActionRequest) -> LingBotVlaV2ActionResponse:
        """Decode one request and run it on the process-local replica."""
        with self._inference_lock:
            return predict_lingbot_vla_v2_action(self.pipeline, request, max_image_bytes=self.config.max_image_bytes)

    def close(self) -> None:
        """Release model resources during application shutdown."""
        self.pipeline.close()


def create_lingbot_vla_v2_app(
    config: LingBotVlaV2ServiceConfig,
    *,
    pipeline_factory: PipelineFactory = _default_pipeline_factory,
) -> FastAPI:
    """Create a FastAPI application backed by exactly one policy replica."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = LingBotVlaV2Service(pipeline_factory(config), config)
        app.state.lingbot_vla_v2_service = service
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="LingBot VLA v2", version="1", lifespan=lifespan)

    @app.get("/health", response_model=LingBotVlaV2HealthResponse)
    async def health() -> LingBotVlaV2HealthResponse:
        return LingBotVlaV2HealthResponse(
            status="ready",
            model="lingbot-vla-v2-6b-base",
            device=config.device,
            policy_verified=False,
        )

    @app.post("/v1/vla/actions", response_model=LingBotVlaV2ActionResponse)
    async def predict(request: LingBotVlaV2ActionRequest) -> LingBotVlaV2ActionResponse:
        service: LingBotVlaV2Service = app.state.lingbot_vla_v2_service
        try:
            return await run_in_threadpool(service.predict, request)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
