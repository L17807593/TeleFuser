"""LiveKit stream service for the ABot-World-0-5B-LF interactive pipeline."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
)
from telefuser.utils.logging import logger

_CONTROL_ALIASES = {
    "ArrowUp": "W",
    "ArrowDown": "S",
    "ArrowLeft": "A",
    "ArrowRight": "D",
    "KeyW": "W",
    "KeyA": "A",
    "KeyS": "S",
    "KeyD": "D",
    "KeyI": "I",
    "KeyJ": "J",
    "KeyK": "K",
    "KeyL": "L",
    "up": "W",
    "down": "S",
    "left": "A",
    "right": "D",
    "forward": "W",
    "backward": "S",
    "w": "W",
    "a": "A",
    "s": "S",
    "d": "D",
    "i": "I",
    "j": "J",
    "k": "K",
    "l": "L",
}
_VALID_CONTROLS = frozenset("WASDIJKL")
_MAX_INPUT_IMAGE_BYTES = 10 * 1024 * 1024
_DEFAULT_OUTPUT_QUEUE_SIZE = 4


@dataclass
class _ABotWorldLiveKitSession:
    session_id: str
    pipeline_session: ABotWorldInteractiveSession
    output_queue: queue.Queue[dict[str, Any]]
    control_event: threading.Event
    config: dict[str, int]
    control_idle_timeout: float = 10.0
    controls: set[str] = field(default_factory=set)
    last_control_at: float = field(default_factory=time.monotonic)
    next_chunk_index: int = 0
    active: bool = True
    worker: threading.Thread | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class ABotWorldLiveKitService:
    """Expose one ABot causal session through TeleFuser's LiveKit contract.

    The service intentionally advertises capacity one because the current
    ABot interactive pipeline owns one retained session at a time. The shared
    TeleFuser LiveKit worker still owns room admission, tokens, pacing, and
    media publication.
    """

    def __init__(
        self,
        pipeline: ABotWorldInteractivePipeline,
        *,
        default_fps: int = 12,
        default_session_config: Mapping[str, object] | None = None,
        output_queue_size: int = _DEFAULT_OUTPUT_QUEUE_SIZE,
        control_idle_timeout: float = 10.0,
        close_timeout: float = 300.0,
    ) -> None:
        if default_fps < 1:
            raise ValueError(f"default_fps must be positive, got {default_fps}")
        if output_queue_size < 1:
            raise ValueError(f"output_queue_size must be positive, got {output_queue_size}")
        if control_idle_timeout <= 0:
            raise ValueError(f"control_idle_timeout must be positive, got {control_idle_timeout}")
        if close_timeout <= 0:
            raise ValueError(f"close_timeout must be positive, got {close_timeout}")
        self.pipeline = pipeline
        self.default_fps = int(default_fps)
        self.default_session_config = dict(default_session_config or {})
        self.output_queue_size = int(output_queue_size)
        self.control_idle_timeout = float(control_idle_timeout)
        self.close_timeout = float(close_timeout)
        self._sessions: dict[str, _ABotWorldLiveKitSession] = {}
        self._sessions_lock = threading.RLock()
        self._capacity_profile: dict[str, object] | None = None

    def start(self) -> None:
        """Preload ABot weights before the LiveKit worker accepts sessions."""
        self.pipeline.preload_models()
        logger.info("ABotWorldLiveKitService started")

    def stop(self) -> None:
        """Close all retained sessions and release the loaded pipeline."""
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.close_session(session_id)
        self.pipeline.close()

    def configure_session_capacity(self, max_sessions: int | None = None) -> dict[str, object]:
        """Report the single retained-session capacity required by ABot."""
        if max_sessions is not None and max_sessions < 1:
            raise ValueError(f"max_sessions must be positive when provided, got {max_sessions}")
        if max_sessions is not None and max_sessions != 1:
            raise ValueError("ABot-World supports one retained LiveKit session per worker")
        profile = {
            "configured_limit": max_sessions,
            "effective_capacity": 1,
            "computed_capacity": 1,
            "model": "ABot-World-0-5B-LF",
        }
        self._capacity_profile = profile
        return dict(profile)

    def session_capacity_profile(self) -> dict[str, object] | None:
        """Return the startup capacity facts for service metadata."""
        return dict(self._capacity_profile) if self._capacity_profile is not None else None

    def has_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            return session_id in self._sessions

    def create_session(self, config: dict) -> str:
        """Create a preview-only session; non-empty controls start generation."""
        with self._sessions_lock:
            if self._sessions:
                raise RuntimeError("ABot-World supports one retained session per worker")

        session_id = str(config.get("session_id") or uuid.uuid4())
        image = self._load_image(config)
        prompt = str(config.get("prompt", self.default_session_config.get("prompt", ""))).strip()
        if not prompt:
            raise ValueError("ABot-World requires a non-empty prompt")
        fps = int(config.get("fps", self.default_session_config.get("fps", self.default_fps)))
        if fps < 1:
            raise ValueError(f"fps must be positive, got {fps}")
        session_idle_timeout = float(config.get("control_idle_timeout", self.control_idle_timeout))
        if session_idle_timeout <= 0:
            raise ValueError(f"control_idle_timeout must be positive, got {session_idle_timeout}")
        control_latent_frames = int(
            config.get(
                "control_latent_frames",
                self.default_session_config.get("control_latent_frames", 3),
            )
        )
        if control_latent_frames not in {1, 3}:
            raise ValueError("control_latent_frames must be 1 or 3")
        seed = int(config.get("seed", self.default_session_config.get("seed", 42)))
        pipeline_session = self.pipeline.create_interactive_session(image, prompt, seed=seed)
        state = _ABotWorldLiveKitSession(
            session_id=session_id,
            pipeline_session=pipeline_session,
            output_queue=queue.Queue(maxsize=self.output_queue_size),
            control_event=threading.Event(),
            config={"fps": fps, "control_latent_frames": control_latent_frames},
            control_idle_timeout=session_idle_timeout,
        )
        with self._sessions_lock:
            if self._sessions:
                self.pipeline.close_interactive_session(pipeline_session)
                raise RuntimeError("ABot-World supports one retained session per worker")
            self._sessions[session_id] = state

        preview = image.convert("RGB").resize(
            (self.pipeline.config.width, self.pipeline.config.height),
            Image.Resampling.BICUBIC,
        )
        self._put_output(
            state,
            {
                "type": "preview",
                "index": -1,
                "fps": fps,
                "timestamp": time.time(),
                "frames": [preview],
            },
        )
        state.worker = threading.Thread(
            target=self._generation_loop,
            args=(state,),
            daemon=True,
            name=f"abot-world-livekit-{session_id[:8]}",
        )
        state.worker.start()
        return session_id

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        """Apply a normalized TeleFuser control message to one ABot session."""
        state = self._session(session_id)
        if state is None:
            return
        message_type = str(chunk.get("type", ""))
        with state.lock:
            if not state.active:
                return
            if message_type == "stop":
                state.active = False
                state.controls.clear()
                state.control_event.set()
                return
            if message_type == "control_state":
                raw_controls = chunk.get("controls", [])
                if not isinstance(raw_controls, list):
                    raise ValueError("control_state controls must be a list")
                state.controls = self._canonical_controls(raw_controls)
            elif message_type == "control":
                control = self._canonical_control(chunk.get("control", chunk.get("key")))
                event = str(chunk.get("event") or chunk.get("action") or "press").lower()
                if event in {"reset", "reset_pose"}:
                    state.controls.clear()
                elif event == "press":
                    state.controls.add(control)
                else:
                    state.controls.discard(control)
            elif message_type in {"reset", "prompt"}:
                # ABot prompt/image state is fixed for a causal session.
                state.controls.clear()
            else:
                raise ValueError(f"Unsupported ABot control message type: {message_type}")
            state.last_control_at = time.monotonic()
            state.control_event.set()

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        """Yield preview and generated frames in order until the session closes."""
        state = self._session(session_id)
        if state is None:
            return
        while True:
            try:
                payload = await asyncio.to_thread(state.output_queue.get, True, 0.25)
            except queue.Empty:
                with state.lock:
                    if not state.active:
                        return
                continue
            yield payload

    def close_session(self, session_id: str, timeout: float | None = None) -> None:
        """Stop generation, wait for the producer, and release the ABot session state."""
        effective_timeout = self.close_timeout if timeout is None else timeout
        with self._sessions_lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return
        with state.lock:
            state.active = False
            state.controls.clear()
            state.control_event.set()
        worker = state.worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=effective_timeout)
        if worker is not None and worker.is_alive():
            logger.warning("ABot session producer did not stop before timeout: session=%s", session_id)
            return
        self.pipeline.close_interactive_session(state.pipeline_session)

    def _generation_loop(self, state: _ABotWorldLiveKitSession) -> None:
        fps = int(state.config["fps"])
        control_latent_frames = int(state.config["control_latent_frames"])
        while True:
            state.control_event.wait(timeout=0.25)
            state.control_event.clear()
            with state.lock:
                if not state.active:
                    return
                controls = set(state.controls)
                idle_seconds = time.monotonic() - state.last_control_at
            if not controls:
                continue
            if idle_seconds >= state.control_idle_timeout:
                with state.lock:
                    state.controls.clear()
                continue
            try:
                frames = self.pipeline.generate_next_block(
                    state.pipeline_session,
                    {key: True for key in controls},
                    control_latent_frames=control_latent_frames,
                )
            except Exception as exc:
                logger.exception("ABot LiveKit generation failed: session=%s", state.session_id)
                self._put_output(
                    state,
                    {"type": "error", "error": str(exc), "timestamp": time.time()},
                )
                with state.lock:
                    state.active = False
                return
            if not frames:
                continue
            if not self._put_output(
                state,
                {
                    "type": "chunk",
                    "index": state.next_chunk_index,
                    "fps": fps,
                    "timestamp": time.time(),
                    "controls": sorted(controls),
                    "frames": frames,
                },
            ):
                return
            state.next_chunk_index += 1

    def _put_output(self, state: _ABotWorldLiveKitSession, payload: dict[str, Any]) -> bool:
        """Queue an ordered payload, blocking the producer when playback is behind."""
        while True:
            with state.lock:
                if not state.active:
                    return False
            try:
                state.output_queue.put(payload, timeout=0.25)
                return True
            except queue.Full:
                continue

    def _session(self, session_id: str) -> _ABotWorldLiveKitSession | None:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def _load_image(self, config: Mapping[str, object]) -> Image.Image:
        image_value = config.get("image", config.get("image_path", self.default_session_config.get("image_path")))
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("ABot-World requires an input image, image_path, or data URL")
        if image_value.startswith("data:"):
            try:
                encoded = image_value.split(",", 1)[1]
            except IndexError as exc:
                raise ValueError("Input image data URL is missing its payload") from exc
            if len(encoded) > (_MAX_INPUT_IMAGE_BYTES * 4 // 3) + 4:
                raise ValueError("Input image exceeds the 10 MiB decoded size limit")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Input image data URL is not valid base64") from exc
            if len(raw) > _MAX_INPUT_IMAGE_BYTES:
                raise ValueError("Input image exceeds the 10 MiB decoded size limit")
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        path = Path(image_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ABot input image does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _canonical_control(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Unsupported ABot control: {value!r}")
        control = _CONTROL_ALIASES.get(value, _CONTROL_ALIASES.get(value.lower()))
        if control is None or control not in _VALID_CONTROLS:
            raise ValueError(f"Unsupported ABot control: {value!r}")
        return control

    @classmethod
    def _canonical_controls(cls, values: list[object]) -> set[str]:
        return {cls._canonical_control(value) for value in values}
