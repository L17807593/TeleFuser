"""Two-phase distilled denoising for LTX-2.5."""

from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25 import LTX25AVTransformer
from telefuser.models.ltx25.sampler import LTX25_STAGE1_DISTILLED_SIGMAS, LTX25_STAGE2_DISTILLED_SIGMAS
from telefuser.offload import AsyncOffloadManager

from .core import LTX25SimpleDenoiser, euler_ancestral_denoising_loop, euler_denoising_loop
from .latent import LatentState


class LTX25DenoisingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_denoising", config)
        self.transformer: LTX25AVTransformer = module_manager.fetch_module("ltx25_transformer")
        self.model_names = ["transformer"]
        self.offload_manager: AsyncOffloadManager | None = None
        if config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            self.offload_manager = AsyncOffloadManager(
                self.transformer.velocity_model.transformer_blocks,
                device=self.device,
                pin_cpu_memory=config.offload_config.pin_cpu_memory,
                offload_ratio=config.offload_config.offload_ratio,
                prefetch_size=config.offload_config.prefetch_size,
            )
            self.transformer.to(device=self.device, dtype=self.torch_dtype)
            self.onload_models_flag = True

    def _onload(self) -> None:
        if not self.onload_models_flag:
            self.transformer.to(self.device)
            self.onload_models_flag = True

    def _offload_between_phases(self) -> None:
        if self.model_runtime_config.offload_config.offload_type == WeightOffloadType.MODEL_CPU_OFFLOAD:
            self.transformer.cpu()
            self.onload_models_flag = False

    @torch.inference_mode()
    def denoise_stage1(
        self,
        video: LatentState,
        audio: LatentState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        *,
        noise_seed: int,
    ) -> tuple[LatentState, LatentState]:
        self._onload()
        result = euler_ancestral_denoising_loop(
            torch.tensor(LTX25_STAGE1_DISTILLED_SIGMAS, device=self.device),
            video,
            audio,
            self.transformer,
            LTX25SimpleDenoiser(video_context, audio_context),
            noise_seed=noise_seed,
            model_dtype=self.torch_dtype,
        )
        self._offload_between_phases()
        if result[0] is None or result[1] is None:
            raise RuntimeError("LTX-2.5 stage-one denoising requires video and audio outputs")
        return result[0], result[1]

    @torch.inference_mode()
    def denoise_stage2(
        self,
        video: LatentState,
        audio: LatentState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> tuple[LatentState, LatentState]:
        self._onload()
        result = euler_denoising_loop(
            torch.tensor(LTX25_STAGE2_DISTILLED_SIGMAS, device=self.device),
            video,
            audio,
            self.transformer,
            LTX25SimpleDenoiser(video_context, audio_context),
            model_dtype=self.torch_dtype,
        )
        if result[0] is None or result[1] is None:
            raise RuntimeError("LTX-2.5 stage-two denoising requires video and audio outputs")
        return result[0], result[1]

    def finish_request(self) -> None:
        if self.offload_manager is not None:
            self.offload_manager.release_all()
        elif self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
            self.transformer.cpu()
            self.onload_models_flag = False


__all__ = ["LTX25DenoisingStage"]
