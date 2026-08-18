"""Single-GPU LTX-2.5 distilled text-to-video and image-to-video pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.models.ltx25 import (
    DiffusionVideoDecoder,
    LTX25AVTransformer,
    LTX25ConvVideoVAE,
    LTX25DurationHead,
    LTX25EmbeddingsProcessor,
    LTX25Gemma4TextEncoder,
    LTX25ModelPaths,
    LTX25SpatialUpsampler,
    LTX25VideoEncoder,
    load_ltx25_audio_decoder_and_vocoder,
)
from telefuser.models.ltx25.diff_vae.types import VideoLatentShape
from telefuser.models.ltx25.sampler import (
    ANCESTRAL_NOISE_SEED_OFFSET,
    LTX25_STAGE1_DISTILLED_SIGMAS,
    LTX25_STAGE2_DISTILLED_SIGMAS,
)
from telefuser.models.ltx25.spatial_upsampler import load_video_latent_statistics
from telefuser.offload import AsyncOffloadManager

from .core import LTX25SimpleDenoiser, euler_ancestral_denoising_loop, euler_denoising_loop
from .image import default_image_crf, preprocess_ltx25_image
from .latent import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    LatentState,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    VideoLatentTools,
)


@dataclass(frozen=True, slots=True)
class LTX25ImageCondition:
    """A still-image condition expressed in output-frame coordinates."""

    image: Image.Image
    frame_idx: int = 0
    strength: float = 1.0
    crf: int | None = None


@dataclass(frozen=True, slots=True)
class LTX25DistilledOutput:
    """Decoded LTX-2.5 result, including audio and unpatchified latent states."""

    video_chunks: tuple[torch.Tensor, ...]
    audio: torch.Tensor
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    num_frames: int
    frame_rate: float


@dataclass(frozen=True, slots=True)
class LTX25DistilledConfig:
    """Construction settings for the reference distilled path."""

    model_root: str
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    video_vae: Literal["diff", "conv"] = "diff"
    offload: Literal["none", "cpu"] = "cpu"


class LTX25DistilledPipeline(BasePipeline):
    """Faithful two-stage distilled LTX-2.5 pipeline without legacy LTX imports."""

    def __init__(self, config: LTX25DistilledConfig) -> None:
        device = torch.device(config.device)
        super().__init__(device=device, torch_dtype=config.torch_dtype)
        self.config = config
        self.paths = LTX25ModelPaths.from_model_root(config.model_root)
        self._streamed_transformer: LTX25AVTransformer | None = None
        self._transformer_offload_manager: AsyncOffloadManager | None = None

    @classmethod
    def from_model_root(
        cls,
        model_root: str | Path,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_vae: Literal["diff", "conv"] = "diff",
        offload: Literal["none", "cpu"] = "cpu",
    ) -> "LTX25DistilledPipeline":
        """Create the pipeline using the official LTX-2.5 model-pack layout."""
        return cls(
            LTX25DistilledConfig(
                model_root=str(model_root),
                device=device,
                torch_dtype=torch_dtype,
                video_vae=video_vae,
                offload=offload,
            )
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        seed: int,
        height: int,
        width: int,
        num_frames: int | None = None,
        frame_rate: float = 24.0,
        images: Sequence[LTX25ImageCondition] = (),
    ) -> LTX25DistilledOutput:
        """Generate a decoded two-stage LTX-2.5 video and stereo waveform."""
        _validate_resolution(height, width, frame_rate)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video_context, audio_context = self._encode_prompt(prompt)
        if num_frames is None:
            num_frames = self._predict_num_frames(video_context, audio_context, frame_rate)
        _validate_request(height, width, num_frames, frame_rate)

        stage_1_tools = _video_tools(
            batch=1,
            frames=num_frames,
            height=height // 2,
            width=width // 2,
            frame_rate=frame_rate,
        )
        audio_tools = _audio_tools(num_frames=num_frames, frame_rate=frame_rate)
        stage_1_video = stage_1_tools.create_initial_state(self.device, self.torch_dtype)
        stage_1_video = self._apply_image_conditions(stage_1_video, stage_1_tools, images, height // 2, width // 2)
        stage_1_video = _noised_state(stage_1_video, 1.0, generator)
        stage_1_audio = _noised_state(audio_tools.create_initial_state(self.device, self.torch_dtype), 1.0, generator)

        transformer = self._load_transformer()
        denoiser = LTX25SimpleDenoiser(video_context, audio_context)
        stage_1_video, stage_1_audio = euler_ancestral_denoising_loop(
            torch.tensor(LTX25_STAGE1_DISTILLED_SIGMAS, device=self.device),
            stage_1_video,
            stage_1_audio,
            transformer,
            denoiser,
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            model_dtype=self.torch_dtype,
        )
        assert stage_1_video is not None and stage_1_audio is not None

        low_resolution_latent = stage_1_tools.unpatchify(stage_1_tools.clear_conditioning(stage_1_video)).latent
        upsampler = LTX25SpatialUpsampler.from_checkpoint(
            self.paths.spatial_upsampler_path, device=self.device, torch_dtype=self.torch_dtype
        )
        latent_statistics = load_video_latent_statistics(self._video_vae_path).to(
            device=self.device, dtype=self.torch_dtype
        )
        upscaled_video = latent_statistics.normalize(upsampler(latent_statistics.un_normalize(low_resolution_latent)))
        self._release(upsampler)

        stage_2_tools = _video_tools(batch=1, frames=num_frames, height=height, width=width, frame_rate=frame_rate)
        stage_2_video = stage_2_tools.create_initial_state(self.device, self.torch_dtype, upscaled_video)
        stage_2_video = self._apply_image_conditions(stage_2_video, stage_2_tools, images, height, width)
        stage_2_video = _noised_state(stage_2_video, LTX25_STAGE2_DISTILLED_SIGMAS[0], generator)
        stage_2_audio = _noised_state(
            audio_tools.create_initial_state(
                self.device,
                self.torch_dtype,
                audio_tools.unpatchify(stage_1_audio).latent,
            ),
            LTX25_STAGE2_DISTILLED_SIGMAS[0],
            generator,
        )
        stage_2_video, stage_2_audio = euler_denoising_loop(
            torch.tensor(LTX25_STAGE2_DISTILLED_SIGMAS, device=self.device),
            stage_2_video,
            stage_2_audio,
            transformer,
            denoiser,
            model_dtype=self.torch_dtype,
        )
        assert stage_2_video is not None and stage_2_audio is not None
        self._release_transformer(transformer)

        video_latent = stage_2_tools.unpatchify(stage_2_tools.clear_conditioning(stage_2_video)).latent
        audio_latent = audio_tools.unpatchify(stage_2_audio).latent
        video_decoder = self._load_video_decoder()
        video_chunks = tuple(self._decode_video(video_decoder, video_latent, generator))
        self._release(video_decoder)

        audio_decoder, vocoder = load_ltx25_audio_decoder_and_vocoder(
            self.paths.audio_vae_path, device=self.device, torch_dtype=self.torch_dtype
        )
        audio = vocoder(audio_decoder(audio_latent)).squeeze(0).float()
        self._release(audio_decoder, vocoder)
        return LTX25DistilledOutput(video_chunks, audio, video_latent, audio_latent, num_frames, frame_rate)

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        text_encoder = LTX25Gemma4TextEncoder.from_checkpoint(
            self.paths.text_encoder_path, device=self.device, torch_dtype=self.torch_dtype
        )
        embeddings = LTX25EmbeddingsProcessor.from_checkpoints(
            self.paths.transformer_path,
            self.paths.text_encoder_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        hidden_states, _, attention_mask = text_encoder.encode([prompt])
        encoded = embeddings(hidden_states, attention_mask)
        self._release(text_encoder, embeddings)
        return encoded.video_encoding, encoded.audio_encoding

    def _load_transformer(self) -> LTX25AVTransformer:
        """Load the denoiser using upstream-equivalent block streaming for CPU offload."""
        if self.config.offload != "cpu" or self.device.type != "cuda":
            return LTX25AVTransformer.from_checkpoint(
                self.paths.transformer_path, device=self.device, torch_dtype=self.torch_dtype
            )
        if getattr(self, "_streamed_transformer", None) is not None:
            return self._streamed_transformer

        transformer = LTX25AVTransformer.from_checkpoint(
            self.paths.transformer_path, device="cpu", torch_dtype=self.torch_dtype
        )
        blocks = transformer.velocity_model.transformer_blocks
        self._transformer_offload_manager = AsyncOffloadManager(
            blocks,
            device=self.device,
            offload_ratio=1.0,
            prefetch_size=1,
        )
        transformer.to(device=self.device, dtype=self.torch_dtype)
        self._streamed_transformer = transformer
        return transformer

    def _release_transformer(self, transformer: LTX25AVTransformer) -> None:
        """Release streamed blocks while retaining their CPU cache for the next request."""
        manager = getattr(self, "_transformer_offload_manager", None)
        if transformer is getattr(self, "_streamed_transformer", None) and manager is not None:
            manager.release_all()
            return
        self._release(transformer)

    def _predict_num_frames(
        self,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        frame_rate: float,
    ) -> int:
        duration_head = LTX25DurationHead.from_checkpoint(
            self.paths.duration_head_path, device=self.device, torch_dtype=self.torch_dtype
        )
        try:
            seconds = float(duration_head(video_context, audio_context).item())
        finally:
            self._release(duration_head)
        from telefuser.models.ltx25.duration import seconds_to_num_frames

        return seconds_to_num_frames(seconds, frame_rate=frame_rate)

    def _apply_image_conditions(
        self,
        state: LatentState,
        tools: VideoLatentTools,
        conditions: Sequence[LTX25ImageCondition],
        height: int,
        width: int,
    ) -> LatentState:
        if not conditions:
            return state
        encoder = LTX25VideoEncoder.from_checkpoint(
            self._video_vae_path, device=self.device, torch_dtype=self.torch_dtype
        )
        try:
            for condition in conditions:
                if condition.frame_idx < 0:
                    raise ValueError(f"image frame_idx must be non-negative, got {condition.frame_idx}")
                if not 0.0 <= condition.strength <= 1.0:
                    raise ValueError(f"image strength must be in [0, 1], got {condition.strength}")
                pixels = preprocess_ltx25_image(
                    condition.image,
                    height,
                    width,
                    default_image_crf(self._video_vae_path) if condition.crf is None else condition.crf,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                encoded = encoder(pixels)
                conditioning = (
                    VideoConditionByLatentIndex(encoded, condition.strength, 0)
                    if condition.frame_idx == 0
                    else VideoConditionByKeyframeIndex(encoded, condition.frame_idx, condition.strength)
                )
                state = conditioning.apply_to(state, tools)
            return state
        finally:
            self._release(encoder)

    def _release(self, *models: torch.nn.Module) -> None:
        """Apply the configured model-residency policy at a phase boundary."""
        if self.config.offload == "cpu":
            _release(*models)

    @property
    def _video_vae_path(self) -> Path:
        return self.paths.video_vae_path if self.config.video_vae == "diff" else self.paths.conv_video_vae_path

    def _load_video_decoder(self) -> DiffusionVideoDecoder | LTX25ConvVideoVAE:
        if self.config.video_vae == "diff":
            return DiffusionVideoDecoder.from_checkpoint(
                self._video_vae_path, device=self.device, torch_dtype=self.torch_dtype
            )
        return LTX25ConvVideoVAE.from_checkpoint(self._video_vae_path, device=self.device, torch_dtype=self.torch_dtype)

    def _decode_video(
        self,
        decoder: DiffusionVideoDecoder | LTX25ConvVideoVAE,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> Sequence[torch.Tensor]:
        if self.config.video_vae == "diff":
            return tuple(decoder.decode_video(latent, generator=generator))  # type: ignore[union-attr]
        return tuple(_conv_video_chunks_to_rgb(decoder.decode(latent, generator=generator)))


def _video_tools(*, batch: int, frames: int, height: int, width: int, frame_rate: float) -> VideoLatentTools:
    latent_shape = VideoLatentShape(
        batch=batch,
        channels=128,
        frames=(frames - 1) // 8 + 1,
        height=height // 32,
        width=width // 32,
    )
    return VideoLatentTools(VideoLatentPatchifier(1), latent_shape, frame_rate)


def _audio_tools(*, num_frames: int, frame_rate: float) -> AudioLatentTools:
    return AudioLatentTools(
        AudioPatchifier(1),
        AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate),
    )


def _noised_state(state: LatentState, noise_scale: float, generator: torch.Generator) -> LatentState:
    noise = torch.randn(state.latent.shape, generator=generator, dtype=state.latent.dtype, device=state.latent.device)
    latent = torch.lerp(state.latent.float(), noise.float(), noise_scale)
    latent = torch.lerp(state.clean_latent.float(), latent, state.denoise_mask)
    return replace(state, latent=latent.to(state.latent.dtype))


def _conv_video_chunks_to_rgb(chunks: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Convert ConvVAE decoder chunks from BCHW-video to DiffVAE's FHWC RGB contract."""
    output: list[torch.Tensor] = []
    for chunk in chunks:
        if chunk.ndim != 5 or chunk.shape[0] != 1 or chunk.shape[1] != 3:
            raise ValueError(f"LTX-2.5 ConvVAE decoder must return [1, 3, F, H, W], got {tuple(chunk.shape)}")
        output.append(chunk[0].permute(1, 2, 3, 0).add(1).mul(0.5).clamp(0, 1))
    return tuple(output)


def _release(*models: torch.nn.Module) -> None:
    for model in models:
        model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_resolution(height: int, width: int, frame_rate: float) -> None:
    if height <= 0 or width <= 0 or height % 64 or width % 64:
        raise ValueError("LTX-2.5 distilled height and width must be positive multiples of 64")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")


def _validate_request(height: int, width: int, num_frames: int, frame_rate: float) -> None:
    _validate_resolution(height, width, frame_rate)
    if num_frames < 1 or (num_frames - 1) % 8:
        raise ValueError("LTX-2.5 num_frames must satisfy num_frames = 8k + 1")


__all__ = [
    "LTX25DistilledConfig",
    "LTX25DistilledOutput",
    "LTX25DistilledPipeline",
    "LTX25ImageCondition",
]
