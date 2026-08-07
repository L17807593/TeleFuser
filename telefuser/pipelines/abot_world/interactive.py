"""Persistent single-session interaction for ABot-World on one GPU.

The runtime mirrors LingBot's important session invariant: text embeddings,
causal DiT KV caches, scheduler state, RNG state, and VAE temporal decode
cache all remain resident between control blocks.  It intentionally supports
one local session; LiveKit admission and multi-session scheduling are a later
transport/service layer rather than a prerequisite for browser testing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from PIL import Image

from telefuser.core.config import WeightOffloadType

from .pipeline import ABotWorldPipeline


@dataclass
class ABotWorldInteractiveSession:
    """State retained across causally generated ABot action blocks."""

    prompt_emb: torch.Tensor
    first_frame_latent: torch.Tensor
    self_cache: list[dict[str, Any]]
    cross_cache: list[dict[str, Any]]
    scheduler: Any
    generator: torch.Generator
    next_latent_frame: int = 0
    emitted_frames: int = 0
    closed: bool = False


class ABotWorldInteractivePipeline(ABotWorldPipeline):
    """ABot pipeline whose model weights and one generation session stay on GPU."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._interactive_lock = threading.RLock()
        self._interactive_session: ABotWorldInteractiveSession | None = None
        self._models_preloaded = False

    def preload_models(self) -> None:
        """Place VAE, T5, and DiT on the configured GPU before accepting controls."""
        with self._interactive_lock:
            if self._models_preloaded:
                return
            for stage in self._get_stages():
                stage.model_runtime_config.offload_config.offload_type = WeightOffloadType.NO_CPU_OFFLOAD
                stage.onload_models()
                stage.onload_models_flag = True
            self._models_preloaded = True

    @torch.inference_mode()
    def create_interactive_session(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int = 42,
    ) -> ABotWorldInteractiveSession:
        """Encode the start image and allocate session-owned causal caches."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        with self._interactive_lock:
            if self._interactive_session is not None:
                self.close_interactive_session(self._interactive_session)
            self.preload_models()
            pixels = self.preprocess_image(image.convert("RGB"), self.config.height, self.config.width)
            start_latent, _ = self.vae_stage.process("encode_image", pixels, None, 1, concat_mask=False)
            first_frame_latent = start_latent.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            prompt_emb = self.text_encoding_stage.process([prompt])[0].to(device=self.device, dtype=self.torch_dtype)
            self_cache, cross_cache = self.denoise_stage._new_cache(
                first_frame_latent.shape[0], first_frame_latent.shape[-2], first_frame_latent.shape[-1]
            )
            session = ABotWorldInteractiveSession(
                prompt_emb=prompt_emb,
                first_frame_latent=first_frame_latent,
                self_cache=self_cache,
                cross_cache=cross_cache,
                scheduler=self.denoise_stage._scheduler(),
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            self._interactive_session = session
            return session

    @torch.inference_mode()
    def generate_next_block(
        self,
        session: ABotWorldInteractiveSession,
        actions: Mapping[str, bool] | None = None,
        control_latent_frames: int = 3,
    ) -> list[Image.Image]:
        """Generate one causal action-controlled latent group."""
        if control_latent_frames not in {1, 3}:
            raise ValueError("control_latent_frames must be 1 or 3")
        with self._interactive_lock:
            if session is not self._interactive_session or session.closed:
                raise RuntimeError("ABot interactive session is no longer active")
            frame_count = control_latent_frames
            latent_shape = session.first_frame_latent.shape
            noise = torch.randn(
                (latent_shape[0], latent_shape[1], frame_count, latent_shape[3], latent_shape[4]),
                generator=session.generator,
                device=self.device,
                dtype=torch.float32,
            )
            action_context = self.build_action_context(
                actions,
                latent_frames=frame_count,
                height=self.config.height,
                width=self.config.width,
                device=self.device,
                dtype=self.torch_dtype,
            )
            latents = self.denoise_stage._denoise_block(
                noise.to(dtype=self.torch_dtype),
                session.prompt_emb,
                action_context,
                session.first_frame_latent if session.next_latent_frame == 0 else None,
                session.self_cache,
                session.cross_cache,
                session.next_latent_frame,
                session.generator,
                session.scheduler,
            )
            decoded = self.vae_stage.process(
                "decode_video_cached",
                latents[0],
                session.next_latent_frame == 0,
                False,
            )
            if decoded.ndim == 5:
                decoded = decoded[0]
            frames = self.tensor2video(decoded)
            session.next_latent_frame += frame_count
            session.emitted_frames += len(frames)
            return frames

    def close_interactive_session(self, session: ABotWorldInteractiveSession | None = None) -> None:
        """Release retained cache references and reset the model-specific VAE stream cache."""
        with self._interactive_lock:
            target = self._interactive_session if session is None else session
            if target is None or target.closed:
                return
            target.closed = True
            target.self_cache.clear()
            target.cross_cache.clear()
            vae = self.vae_stage.vae
            if hasattr(vae, "_feat_cache"):
                vae._feat_cache = []
                vae._feat_idx = [0]
            if target is self._interactive_session:
                self._interactive_session = None

    def close(self) -> None:
        self.close_interactive_session()
        super().close()
