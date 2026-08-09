"""Sequential SwiftVR restoration pipeline.

The implementation follows H-oliday/SwiftVR commit
5ca168cef6ca7200f135fdfea85e5e13d12c5b53. Model execution remains sequential
to preserve the approved differential path.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import torch
from safetensors.torch import load_file

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.swiftvr_reae import ReAE
from telefuser.models.swiftvr_transformer import SwiftVRWanTransformer3DModel

from .chunk import ChunkSpec, ChunkType, build_chunk_specs
from .io import (
    crop_spatial_padding_ntchw,
    preprocess_clip_uint8,
)
from .streaming_dit import StreamingDiT
from .streaming_tae import StreamingTAE

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _as_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose float16, bfloat16, or float32.")
    return _DTYPES[key]


def aligned_pad(size: int, multiple: int = 32) -> int:
    """Return right/bottom padding needed to align a spatial dimension."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return (multiple - size % multiple) % multiple


@dataclass
class SwiftVRPipelineConfig:
    """Runtime configuration using existing TeleFuser model controls."""

    dit_config: ModelRuntimeConfig = field(
        default_factory=lambda: ModelRuntimeConfig(
            attention_config=AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        )
    )


class SwiftVRPipeline(BasePipeline):
    """Faithful single-device SwiftVR video-restoration pipeline."""

    upscale_mode = "bilinear"

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.device = torch.device(device)
        self._execution_lock = threading.RLock()

    def init(
        self,
        module_manager: ModuleManager,
        config: SwiftVRPipelineConfig,
        prompt_emb: torch.Tensor,
    ) -> None:
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.reae = module_manager.fetch_module("swiftvr_reae")
        self.transformer = module_manager.fetch_module("swiftvr_transformer")
        if self.reae is None or self.transformer is None:
            raise RuntimeError("SwiftVR requires both swiftvr_reae and swiftvr_transformer")
        self.prompt_emb = prompt_emb.to(dtype=self.torch_dtype)
        self._prepare_for_inference()

    def _prepare_for_inference(self) -> None:
        self.reae.to(device=self.device, dtype=self.torch_dtype).eval()
        self.transformer.to(device=self.device, dtype=self.torch_dtype).eval()
        if self.device.type == "cuda":
            # Match the released SwiftVR runtime before selecting convolution
            # and attention kernels; these flags are part of the frozen parity
            # configuration used by the upstream pipeline.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.transformer.prepare_for_inference(attention_config=self.config.dit_config.attention_config)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype | str = torch.bfloat16,
        attention_config: AttentionConfig | None = None,
    ) -> "SwiftVRPipeline":
        """Load the released local checkpoint through ModuleManager."""
        root = Path(model_id_or_path)
        dtype = _as_dtype(torch_dtype)
        transformer_dir = root / "transformer"
        config_path = transformer_dir / "config.json"
        with config_path.open(encoding="utf-8") as handle:
            transformer_config = json.load(handle)

        # Upstream constructs both models on CPU, then moves ReAE followed by
        # the transformer during pipeline preparation. Keep that allocation
        # order because cuDNN benchmarking can otherwise select a numerically
        # different convolution plan.
        module_manager = ModuleManager(torch_dtype=dtype, device="cpu")
        module_manager.load_model(
            str(root / "reae.safetensors"),
            device="cpu",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            name="swiftvr_reae",
            model_class=ReAE,
            model_resource="official",
        )
        module_manager.load_model(
            str(transformer_dir),
            device="cpu",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            name="swiftvr_transformer",
            model_class=SwiftVRWanTransformer3DModel,
            model_resource="diffusers",
            converter_kwargs={"config": transformer_config},
        )
        prompt_payload = load_file(str(root / "prompt_embedding.safetensors"), device="cpu")
        prompt_emb = prompt_payload["prompt_emb"][0]

        pipeline_config = SwiftVRPipelineConfig()
        if attention_config is not None:
            pipeline_config.dit_config.attention_config = attention_config
        pipeline = cls(device=device, torch_dtype=dtype)
        pipeline.init(module_manager, pipeline_config, prompt_emb)
        return pipeline

    @staticmethod
    def _validate_clip_len(clip_len: int) -> None:
        if clip_len <= 0 or clip_len % 4:
            raise ValueError(f"clip_len must be a positive multiple of 4, got {clip_len}")

    def _target_size(
        self,
        lq_h: int,
        lq_w: int,
        resolution: tuple[int, int] | None,
        upscale: int,
    ) -> tuple[int, int, int, int]:
        if resolution is not None:
            out_w, out_h = int(resolution[0]), int(resolution[1])
        else:
            if upscale <= 0:
                raise ValueError(f"upscale must be positive, got {upscale}")
            out_h, out_w = lq_h * upscale, lq_w * upscale
        return out_h, out_w, aligned_pad(out_h), aligned_pad(out_w)

    def _restored_chunks(
        self,
        chunks: Iterable[tuple[ChunkSpec, torch.Tensor]],
        *,
        clip_len: int,
        out_h: int,
        out_w: int,
        pad_h: int,
        pad_w: int,
        dit_overlap: int,
    ) -> Iterator[torch.Tensor]:
        tae_stream = StreamingTAE(self.reae)
        dit_stream = StreamingDiT(self.transformer, overlap=dit_overlap)
        n_lat = clip_len // 4
        prev_dit_out_cpu = None

        with self._execution_lock:
            for spec, frames_uint8 in chunks:
                gpu_frames = frames_uint8.to(device=self.device)
                clip = preprocess_clip_uint8(
                    gpu_frames,
                    out_h,
                    out_w,
                    self.upscale_mode,
                    pad_h,
                    pad_w,
                    self.torch_dtype,
                )
                encoded = tae_stream.encode_chunk_fixed(clip, spec)
                if spec.ctype == ChunkType.LAST:
                    denoised = dit_stream.denoise_last_chunk(
                        encoded,
                        spec,
                        self.prompt_emb,
                        prev_dit_out_cpu,
                        n_lat,
                        self.device,
                        self.torch_dtype,
                    )
                else:
                    encoded_bcfhw = encoded.permute(0, 2, 1, 3, 4).contiguous()
                    denoised_bcfhw = dit_stream.denoise(encoded_bcfhw, self.prompt_emb)
                    denoised = denoised_bcfhw.permute(0, 2, 1, 3, 4).contiguous()
                    prev_dit_out_cpu = encoded_bcfhw[:, :, -n_lat:].detach().cpu().clone()
                decoded = tae_stream.decode_chunk_fixed(denoised, spec)
                if decoded is not None and decoded.shape[1]:
                    yield crop_spatial_padding_ntchw(decoded, pad_h, pad_w)

    @torch.inference_mode()
    def __call__(
        self,
        frames_uint8: torch.Tensor,
        *,
        resolution: tuple[int, int] | None = None,
        upscale: int = 4,
        clip_len: int = 24,
        dit_overlap: int = 0,
    ) -> torch.Tensor:
        """Restore [T,H,W,3] uint8 frames and return [1,T,3,H,W]."""
        self._validate_clip_len(clip_len)
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3 or frames_uint8.dtype != torch.uint8:
            raise ValueError("frames_uint8 must have shape [T,H,W,3] and dtype uint8")
        total_frames = 4 * ((int(frames_uint8.shape[0]) - 1) // 4) + 1
        if total_frames <= 0:
            raise ValueError("frames_uint8 must contain at least one frame")
        frames_uint8 = frames_uint8[:total_frames]
        out_h, out_w, pad_h, pad_w = self._target_size(
            int(frames_uint8.shape[1]),
            int(frames_uint8.shape[2]),
            resolution,
            upscale,
        )
        chunks = (
            (
                spec,
                frames_uint8[spec.frame_start : spec.frame_start + spec.frame_count],
            )
            for spec in build_chunk_specs(total_frames, clip_len)
        )
        outputs = list(
            self._restored_chunks(
                chunks,
                clip_len=clip_len,
                out_h=out_h,
                out_w=out_w,
                pad_h=pad_h,
                pad_w=pad_w,
                dit_overlap=dit_overlap,
            )
        )
        if not outputs:
            return torch.empty((1, 0, 3, out_h, out_w), dtype=self.torch_dtype, device=self.device)
        return torch.cat(outputs, dim=1)

    def stream(
        self,
        *,
        clip_len: int = 24,
        resolution: tuple[int, int] | None = None,
        upscale: int = 4,
        dit_overlap: int = 1,
    ) -> "SwiftVRStreamSession":
        self._validate_clip_len(clip_len)
        return SwiftVRStreamSession(
            self,
            clip_len=clip_len,
            resolution=resolution,
            upscale=upscale,
            dit_overlap=dit_overlap,
        )


class SwiftVRStreamSession:
    """Per-session ReAE, overlap, and RoPE state for causal restoration."""

    def __init__(
        self,
        pipeline: SwiftVRPipeline,
        *,
        clip_len: int,
        resolution: tuple[int, int] | None,
        upscale: int,
        dit_overlap: int,
    ) -> None:
        self.pipeline = pipeline
        self.clip_len = clip_len
        self.resolution = resolution
        self.upscale = upscale
        self._sizes: tuple[int, int, int, int] | None = None
        self._tae = StreamingTAE(pipeline.reae)
        self._dit = StreamingDiT(pipeline.transformer, overlap=dit_overlap)
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SwiftVR stream session is closed")

    def _ensure_sizes(self, lq_h: int, lq_w: int) -> tuple[int, int, int, int]:
        if self._sizes is None:
            self._sizes = self.pipeline._target_size(lq_h, lq_w, self.resolution, self.upscale)
        return self._sizes

    def _run_latents(self, encoded: torch.Tensor) -> torch.Tensor:
        encoded_bcfhw = encoded.permute(0, 2, 1, 3, 4).contiguous()
        denoised = self._dit.denoise(encoded_bcfhw, self.pipeline.prompt_emb)
        return denoised.permute(0, 2, 1, 3, 4).contiguous()

    @torch.inference_mode()
    def step(self, frames_uint8: torch.Tensor) -> torch.Tensor | None:
        self._ensure_open()
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3 or frames_uint8.dtype != torch.uint8:
            raise ValueError("frames_uint8 must have shape [T,H,W,3] and dtype uint8")
        with self.pipeline._execution_lock:
            frames = frames_uint8.to(self.pipeline.device)
            out_h, out_w, pad_h, pad_w = self._ensure_sizes(int(frames.shape[1]), int(frames.shape[2]))
            clip = preprocess_clip_uint8(
                frames,
                out_h,
                out_w,
                self.pipeline.upscale_mode,
                pad_h,
                pad_w,
                self.pipeline.torch_dtype,
            )
            encoded = self._tae.encode_chunk(clip)
            if encoded is None:
                return None
            decoded = self._tae.decode_chunk(self._run_latents(encoded))
            return crop_spatial_padding_ntchw(decoded, pad_h, pad_w)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor | None:
        self._ensure_open()
        with self.pipeline._execution_lock:
            encoded = self._tae.flush_encoder()
            if encoded is None or self._sizes is None:
                return None
            _, _, pad_h, pad_w = self._sizes
            decoded = self._tae.decode_chunk(self._run_latents(encoded))
            return crop_spatial_padding_ntchw(decoded, pad_h, pad_w)

    def close(self) -> None:
        if self._closed:
            return
        self._tae.reset()
        self._dit.reset()
        self._dit._cond_cache = None
        self._dit._cond_cache_key = None
        self._sizes = None
        self._closed = True
