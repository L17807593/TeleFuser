"""LTX-2.5 distilled pipeline stages using TeleFuser's standard lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.models.ltx25.audio import LTX25AudioVAEDecoder, LTX25AudioVocoder
from telefuser.models.ltx25.conv_video_vae import LTX25ConvVideoVAE
from telefuser.models.ltx25.diff_vae.diffusion_video_decoder import DiffusionVideoDecoder
from telefuser.models.ltx25.embeddings import LTX25EmbeddingsProcessor
from telefuser.models.ltx25.gemma4 import LTX25Gemma4TextEncoder
from telefuser.models.ltx25.spatial_upsampler import LTX25SpatialUpsampler
from telefuser.models.ltx25.transformer import LTX25AVTransformer

from .core import LTX25SimpleDenoiser, euler_ancestral_denoising_loop, euler_denoising_loop
from .latent import LatentState


class LTX25TextEncodingStage(BaseStage):
    def __init__(
        self,
        text_encoder: LTX25Gemma4TextEncoder,
        embeddings_processor: LTX25EmbeddingsProcessor,
        config: ModelRuntimeConfig,
    ) -> None:
        super().__init__("ltx25_text_encoding", config)
        self.text_encoder, self.embeddings_processor = text_encoder, embeddings_processor
        self.model_names = ["text_encoder", "embeddings_processor"]

    @with_model_offload(["text_encoder", "embeddings_processor"])
    def __call__(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, _, attention_mask = self.text_encoder.encode([prompt])
        encoded = self.embeddings_processor(hidden_states, attention_mask)
        return encoded.video_encoding, encoded.audio_encoding


class LTX25ImageConditioningStage(BaseStage):
    """Apply prebuilt image conditions without modifying sampling order."""

    def __init__(self, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_image_conditioning", config)

    def __call__(self, state: LatentState, conditions: Sequence[Callable[[LatentState], LatentState]]) -> LatentState:
        for condition in conditions:
            state = condition(state)
        return state


class LTX25DenoisingStage(BaseStage):
    def __init__(self, transformer: LTX25AVTransformer, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_denoising", config)
        self.transformer, self.model_names = transformer, ["transformer"]

    @with_model_offload(["transformer"])
    def __call__(
        self,
        video: LatentState,
        audio: LatentState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        ancestral: bool,
        noise_seed: int | None = None,
    ) -> tuple[LatentState, LatentState]:
        denoiser = LTX25SimpleDenoiser(video_context, audio_context)
        if ancestral:
            if noise_seed is None:
                raise ValueError("ancestral denoising requires noise_seed")
            result = euler_ancestral_denoising_loop(
                sigmas, video, audio, self.transformer, denoiser, noise_seed=noise_seed, model_dtype=self.torch_dtype
            )
        else:
            result = euler_denoising_loop(
                sigmas, video, audio, self.transformer, denoiser, model_dtype=self.torch_dtype
            )
        if result[0] is None or result[1] is None:
            raise RuntimeError("LTX-2.5 requires video and audio outputs")
        return result[0], result[1]


class LTX25UpsamplerStage(BaseStage):
    def __init__(self, upsampler: LTX25SpatialUpsampler, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_upsampler", config)
        self.upsampler, self.model_names = upsampler, ["upsampler"]

    @with_model_offload(["upsampler"])
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return self.upsampler(latent)


class LTX25VideoDecodeStage(BaseStage):
    def __init__(self, decoder: DiffusionVideoDecoder | LTX25ConvVideoVAE, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_video_decode", config)
        self.decoder, self.model_names = decoder, ["decoder"]

    @with_model_offload(["decoder"])
    def __call__(self, latent: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        if isinstance(self.decoder, DiffusionVideoDecoder):
            return tuple(self.decoder.decode_video(latent, generator=generator))
        return tuple(
            chunk[0].permute(1, 2, 3, 0).add(1).mul(0.5).clamp(0, 1)
            for chunk in self.decoder.decode(latent, generator=generator)
        )


class LTX25AudioDecodeStage(BaseStage):
    def __init__(self, decoder: LTX25AudioVAEDecoder, vocoder: LTX25AudioVocoder, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_audio_decode", config)
        self.decoder, self.vocoder, self.model_names = decoder, vocoder, ["decoder", "vocoder"]

    @with_model_offload(["decoder", "vocoder"])
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vocoder(self.decoder(latent)).squeeze(0).float()


__all__ = [
    "LTX25AudioDecodeStage",
    "LTX25DenoisingStage",
    "LTX25ImageConditioningStage",
    "LTX25TextEncodingStage",
    "LTX25UpsamplerStage",
    "LTX25VideoDecodeStage",
]
