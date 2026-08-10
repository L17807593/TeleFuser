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
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    CompileConfig,
    ModelRuntimeConfig,
    ParallelConfig,
    QuantConfig,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.models.swiftvr_reae import ReAE
from telefuser.models.swiftvr_transformer import (
    SwiftVRWanTransformer3DModel,
    compile_transformer_blocks_with_config,
)
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker
from telefuser.worker.tensor_channel import WorkerTensorChannel

from .chunk import ChunkSpec, ChunkType, build_chunk_specs
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


_INTERP_NEEDS_ALIGN = ("linear", "bilinear", "bicubic", "trilinear")


def preprocess_clip_uint8(
    frames_uint8: torch.Tensor,
    out_h: int,
    out_w: int,
    mode: str,
    pad_h: int,
    pad_w: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert uint8 THWC frames to padded target-dtype NTCHW frames in [0, 1]."""
    frames = frames_uint8.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)
    _, _, height, width = frames.shape
    if (height, width) != (out_h, out_w):
        if mode in _INTERP_NEEDS_ALIGN:
            frames = F.interpolate(frames, size=(out_h, out_w), mode=mode, align_corners=False)
        else:
            frames = F.interpolate(frames, size=(out_h, out_w), mode=mode)
    frames = frames / 255.0
    if pad_h > 0 or pad_w > 0:
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return frames.unsqueeze(0)


def crop_spatial_padding_ntchw(video: torch.Tensor | None, pad_h: int = 0, pad_w: int = 0) -> torch.Tensor | None:
    """Remove bottom/right spatial padding from an NTCHW tensor."""
    if video is None:
        return None
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


def ntchw_to_pil_frames(video: torch.Tensor | None) -> list[Image.Image]:
    """Convert [0, 1] NTCHW output to PIL RGB frames on the host."""
    if video is None or video.numel() == 0 or video.shape[1] == 0:
        return []
    frames = (video[0].permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8)
    if frames.device.type == "cuda":
        cpu_frames = torch.empty_like(frames, device="cpu", pin_memory=True)
        cpu_frames.copy_(frames, non_blocking=True)
        torch.cuda.current_stream(frames.device).synchronize()
        frames = cpu_frames
    else:
        frames = frames.cpu()
    return [Image.fromarray(frame.numpy()) for frame in frames]


@dataclass
class SwiftVRPipelineConfig:
    """Runtime configuration using existing TeleFuser model controls."""

    encode_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(
        default_factory=lambda: ModelRuntimeConfig(
            attention_config=AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        )
    )
    decode_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    enable_stage_parallel: bool = False
    enable_stage_overlap: bool = False
    dit_overlap: int = 1
    tensor_channel_slots: int = 2


def _apply_cuda_runtime_flags(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _prepare_swiftvr_transformer(
    transformer: SwiftVRWanTransformer3DModel,
    runtime_config: ModelRuntimeConfig,
) -> None:
    transformer.prepare_for_inference(attention_config=runtime_config.attention_config)
    if runtime_config.quant_config.enabled:
        transformer.enable_quant(runtime_config.quant_config)
    if runtime_config.parallel_config.world_size == 1 and runtime_config.compile_config.enabled:
        compile_transformer_blocks_with_config(transformer, runtime_config.compile_config)


class _SwiftVRReAEEncodeStage(BaseStage):
    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.reae: ReAE = module_manager.fetch_module("swiftvr_reae")
        self.model_names = ["reae"]
        self._tae: StreamingTAE | None = None

    def _ensure_session(self) -> StreamingTAE:
        self.reae.to(device=self.device, dtype=self.torch_dtype).eval()
        _apply_cuda_runtime_flags(self.device)
        if self._tae is None:
            self._tae = StreamingTAE(self.reae)
        return self._tae

    @with_model_offload(["reae"])
    @torch.inference_mode()
    def process(
        self,
        frames_uint8: torch.Tensor,
        out_h: int,
        out_w: int,
        pad_h: int,
        pad_w: int,
        upscale_mode: str,
    ) -> torch.Tensor | None:
        tae = self._ensure_session()
        clip = preprocess_clip_uint8(frames_uint8, out_h, out_w, upscale_mode, pad_h, pad_w, self.torch_dtype)
        return tae.encode_chunk(clip)

    @with_model_offload(["reae"])
    @torch.inference_mode()
    def flush_encoder(self) -> torch.Tensor | None:
        return self._ensure_session().flush_encoder()

    def reset_session(self) -> None:
        if self._tae is not None:
            self._tae.reset()
        self._tae = None


class _SwiftVRDiTStage(BaseStage):
    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        prompt_emb: torch.Tensor,
        dit_overlap: int,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.transformer: SwiftVRWanTransformer3DModel = module_manager.fetch_module("swiftvr_transformer")
        self.prompt_emb = prompt_emb.to(dtype=self.torch_dtype)
        self.model_names = ["transformer"]
        self.dit_overlap = dit_overlap
        self._dit: StreamingDiT | None = None
        self._prepared = False

    def _ensure_session(self) -> StreamingDiT:
        self.transformer.to(device=self.device, dtype=self.torch_dtype).eval()
        _apply_cuda_runtime_flags(self.device)
        if not self._prepared:
            _prepare_swiftvr_transformer(self.transformer, self.model_runtime_config)
            self._prepared = True
        if self._dit is None:
            self._dit = StreamingDiT(self.transformer, overlap=self.dit_overlap)
        return self._dit

    @with_model_offload(["transformer"])
    @torch.inference_mode()
    def process(self, encoded: torch.Tensor) -> torch.Tensor:
        dit = self._ensure_session()
        encoded_bcfhw = encoded.permute(0, 2, 1, 3, 4).contiguous()
        denoised = dit.denoise(encoded_bcfhw, self.prompt_emb)
        return denoised.permute(0, 2, 1, 3, 4).contiguous()

    def reset_session(self) -> None:
        if self._dit is not None:
            self._dit.reset()
            self._dit._cond_cache = None
            self._dit._cond_cache_key = None
        self._dit = None


class _SwiftVRReAEDecodeStage(BaseStage):
    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.reae: ReAE = module_manager.fetch_module("swiftvr_reae")
        self.model_names = ["reae"]
        self._tae: StreamingTAE | None = None

    def _ensure_session(self) -> StreamingTAE:
        self.reae.to(device=self.device, dtype=self.torch_dtype).eval()
        _apply_cuda_runtime_flags(self.device)
        if self._tae is None:
            self._tae = StreamingTAE(self.reae)
        return self._tae

    @with_model_offload(["reae"])
    @torch.inference_mode()
    def process(self, denoised: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor | None:
        decoded = self._ensure_session().decode_chunk(denoised)
        return crop_spatial_padding_ntchw(decoded, pad_h, pad_w)

    def reset_session(self) -> None:
        if self._tae is not None:
            self._tae.reset()
        self._tae = None


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
        self._worker_tensor_channels: list[WorkerTensorChannel] = []
        self._stage_parallel_active_session = False
        self.encode_stage = None
        self.dit_stage = None
        self.decode_stage = None
        if config.enable_stage_parallel:
            self._init_stage_workers(module_manager, prompt_emb)
        else:
            self._prepare_for_inference()

    def _prepare_for_inference(self) -> None:
        self.reae.to(device=self.device, dtype=self.torch_dtype).eval()
        self.transformer.to(device=self.device, dtype=self.torch_dtype).eval()
        _apply_cuda_runtime_flags(self.device)
        _prepare_swiftvr_transformer(self.transformer, self.config.dit_config)

    def _init_stage_workers(self, module_manager: ModuleManager, prompt_emb: torch.Tensor) -> None:
        timeout = max(
            self.config.encode_config.parallel_config.timeout,
            self.config.dit_config.parallel_config.timeout,
            self.config.decode_config.parallel_config.timeout,
        )
        encode_to_dit = WorkerTensorChannel(
            self.config.dit_config.parallel_config.world_size,
            timeout=timeout,
            cuda_ipc_slots=self.config.tensor_channel_slots,
        )
        dit_to_decode = WorkerTensorChannel(
            self.config.decode_config.parallel_config.world_size,
            timeout=timeout,
            cuda_ipc_slots=self.config.tensor_channel_slots,
        )
        self._worker_tensor_channels.extend((encode_to_dit, dit_to_decode))
        self.encode_stage = ParallelWorker(
            _SwiftVRReAEEncodeStage("swiftvr_encode", module_manager, self.config.encode_config),
            tensor_output_channel=encode_to_dit,
            tensor_output_methods=("process", "flush_encoder"),
        )
        self.dit_stage = ParallelWorker(
            _SwiftVRDiTStage(
                "swiftvr_dit",
                module_manager,
                self.config.dit_config,
                prompt_emb,
                self.config.dit_overlap,
            ),
            tensor_output_channel=dit_to_decode,
            tensor_output_methods=("process",),
            tensor_input_channels=(encode_to_dit,),
        )
        self.decode_stage = ParallelWorker(
            _SwiftVRReAEDecodeStage("swiftvr_decode", module_manager, self.config.decode_config),
            tensor_input_channels=(dit_to_decode,),
        )
        logger.info("SwiftVR stage-parallel workers initialized with direct tensor channels")

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype | str = torch.bfloat16,
        attention_config: AttentionConfig | None = None,
        compile_config: CompileConfig | None = None,
        quant_config: QuantConfig | None = None,
        enable_stage_parallel: bool = False,
        enable_stage_overlap: bool = False,
        stage_dit_overlap: int = 1,
        stage_device_ids: tuple[int, int, int] | list[int] | None = None,
        tensor_channel_slots: int = 2,
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

        pipeline_config = SwiftVRPipelineConfig(
            enable_stage_parallel=enable_stage_parallel,
            enable_stage_overlap=enable_stage_overlap,
            dit_overlap=stage_dit_overlap,
            tensor_channel_slots=tensor_channel_slots,
        )
        runtime_configs = (pipeline_config.encode_config, pipeline_config.dit_config, pipeline_config.decode_config)
        for runtime_config in runtime_configs:
            runtime_config.torch_dtype = dtype
        if attention_config is not None:
            pipeline_config.dit_config.attention_config = attention_config
        if compile_config is not None:
            pipeline_config.dit_config.compile_config = compile_config
        if quant_config is not None:
            pipeline_config.dit_config.quant_config = quant_config
        if stage_device_ids is not None:
            if len(stage_device_ids) != 3:
                raise ValueError("stage_device_ids must contain encode, dit, and decode device ids")
            for runtime_config, device_id in zip(
                (pipeline_config.encode_config, pipeline_config.dit_config, pipeline_config.decode_config),
                stage_device_ids,
                strict=True,
            ):
                runtime_config.device_type = torch.device(device).type
                runtime_config.device_id = int(device_id)
                runtime_config.parallel_config = ParallelConfig(device_ids=[int(device_id)])
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
    ) -> list[Image.Image]:
        """Restore [T,H,W,3] uint8 frames and return PIL RGB frames."""
        self._validate_clip_len(clip_len)
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3 or frames_uint8.dtype != torch.uint8:
            raise ValueError("frames_uint8 must have shape [T,H,W,3] and dtype uint8")
        total_frames = 4 * ((int(frames_uint8.shape[0]) - 1) // 4) + 1
        if total_frames <= 0:
            raise ValueError("frames_uint8 must contain at least one frame")
        frames_uint8 = frames_uint8[:total_frames]
        if self.config.enable_stage_parallel:
            session = self.stream(
                clip_len=clip_len,
                resolution=resolution,
                upscale=upscale,
                dit_overlap=self.config.dit_overlap,
            )
            try:
                if self.config.enable_stage_overlap and isinstance(session, SwiftVRStagedStreamSession):
                    outputs = session.restore_chunks(frames_uint8, clip_len)
                else:
                    outputs = []
                    for start in range(0, int(frames_uint8.shape[0]), clip_len):
                        outputs.extend(session.step(frames_uint8[start : start + clip_len]))
                    outputs.extend(session.flush())
            finally:
                session.close()
            return outputs
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
            return []
        return ntchw_to_pil_frames(torch.cat(outputs, dim=1))

    def stream(
        self,
        *,
        clip_len: int = 24,
        resolution: tuple[int, int] | None = None,
        upscale: int = 4,
        dit_overlap: int = 1,
    ) -> "SwiftVRStreamSession | SwiftVRStagedStreamSession":
        self._validate_clip_len(clip_len)
        if self.config.enable_stage_parallel:
            if dit_overlap != self.config.dit_overlap:
                raise ValueError("stage-parallel SwiftVR requires dit_overlap to match the configured stage overlap")
            if self._stage_parallel_active_session:
                raise RuntimeError("stage-parallel SwiftVR supports one active stream session per pipeline")
            self._stage_parallel_active_session = True
            return SwiftVRStagedStreamSession(
                self,
                clip_len=clip_len,
                resolution=resolution,
                upscale=upscale,
            )
        return SwiftVRStreamSession(
            self,
            clip_len=clip_len,
            resolution=resolution,
            upscale=upscale,
            dit_overlap=dit_overlap,
        )

    def close(self) -> None:
        for stage in (self.decode_stage, self.dit_stage, self.encode_stage):
            if isinstance(stage, ParallelWorker):
                stage.close()
        for channel in getattr(self, "_worker_tensor_channels", ()):
            channel.close()
        self._worker_tensor_channels = []
        self._stage_parallel_active_session = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SwiftVRStagedStreamSession:
    """Stage-parallel SwiftVR session using WorkerTensorChannel between stages."""

    def __init__(
        self,
        pipeline: SwiftVRPipeline,
        *,
        clip_len: int,
        resolution: tuple[int, int] | None,
        upscale: int,
    ) -> None:
        self.pipeline = pipeline
        self.clip_len = clip_len
        self.resolution = resolution
        self.upscale = upscale
        self._sizes: tuple[int, int, int, int] | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SwiftVR stream session is closed")

    def _ensure_sizes(self, lq_h: int, lq_w: int) -> tuple[int, int, int, int]:
        if self._sizes is None:
            self._sizes = self.pipeline._target_size(lq_h, lq_w, self.resolution, self.upscale)
        return self._sizes

    def _run_encoded(self, encoded: object, pad_h: int, pad_w: int) -> list[Image.Image]:
        if encoded is None:
            return []
        if not isinstance(self.pipeline.dit_stage, ParallelWorker) or not isinstance(
            self.pipeline.decode_stage, ParallelWorker
        ):
            raise RuntimeError("SwiftVR stage workers are not initialized")
        denoised_wait = self.pipeline.dit_stage.process(encoded, _tensor_transport=True)
        denoised = denoised_wait()
        return ntchw_to_pil_frames(self.pipeline.decode_stage.process(denoised, pad_h, pad_w, sync=True))

    def _submit_encode(self, frames_uint8: torch.Tensor, pad: tuple[int, int, int, int]):
        if not isinstance(self.pipeline.encode_stage, ParallelWorker):
            raise RuntimeError("SwiftVR encode stage is not initialized")
        out_h, out_w, pad_h, pad_w = pad
        return self.pipeline.encode_stage.process(
            frames_uint8,
            out_h,
            out_w,
            pad_h,
            pad_w,
            self.pipeline.upscale_mode,
            _tensor_transport=True,
        )

    def restore_chunks(self, frames_uint8: torch.Tensor, clip_len: int) -> list[Image.Image]:
        self._ensure_open()
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3 or frames_uint8.dtype != torch.uint8:
            raise ValueError("frames_uint8 must have shape [T,H,W,3] and dtype uint8")
        if not isinstance(self.pipeline.dit_stage, ParallelWorker) or not isinstance(
            self.pipeline.decode_stage, ParallelWorker
        ):
            raise RuntimeError("SwiftVR stage workers are not initialized")
        pad = self._ensure_sizes(int(frames_uint8.shape[1]), int(frames_uint8.shape[2]))
        _, _, pad_h, pad_w = pad
        chunks = [frames_uint8[start : start + clip_len] for start in range(0, int(frames_uint8.shape[0]), clip_len)]
        outputs: list[Image.Image] = []
        decode_wait = None
        encode_wait = self._submit_encode(chunks[0], pad) if chunks else None

        for index, _chunk in enumerate(chunks):
            encoded = encode_wait() if encode_wait is not None else None
            next_index = index + 1
            encode_wait = self._submit_encode(chunks[next_index], pad) if next_index < len(chunks) else None
            if encoded is None:
                continue
            denoised_wait = self.pipeline.dit_stage.process(encoded, _tensor_transport=True)
            if decode_wait is not None:
                output = decode_wait()
                if output is not None:
                    outputs.extend(ntchw_to_pil_frames(output))
            denoised = denoised_wait()
            decode_wait = self.pipeline.decode_stage.process(denoised, pad_h, pad_w)

        if encode_wait is not None:
            encoded = encode_wait()
            if encoded is not None:
                denoised = self.pipeline.dit_stage.process(encoded, _tensor_transport=True)()
                if decode_wait is not None:
                    output = decode_wait()
                    if output is not None:
                        outputs.extend(ntchw_to_pil_frames(output))
                decode_wait = self.pipeline.decode_stage.process(denoised, pad_h, pad_w)

        if isinstance(self.pipeline.encode_stage, ParallelWorker):
            encoded = self.pipeline.encode_stage.flush_encoder(_tensor_transport=True)()
            if encoded is not None:
                denoised = self.pipeline.dit_stage.process(encoded, _tensor_transport=True)()
                if decode_wait is not None:
                    output = decode_wait()
                    if output is not None:
                        outputs.extend(ntchw_to_pil_frames(output))
                decode_wait = self.pipeline.decode_stage.process(denoised, pad_h, pad_w)

        if decode_wait is not None:
            output = decode_wait()
            if output is not None:
                outputs.extend(ntchw_to_pil_frames(output))
        return outputs

    @torch.inference_mode()
    def step(self, frames_uint8: torch.Tensor) -> list[Image.Image]:
        self._ensure_open()
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3 or frames_uint8.dtype != torch.uint8:
            raise ValueError("frames_uint8 must have shape [T,H,W,3] and dtype uint8")
        if not isinstance(self.pipeline.encode_stage, ParallelWorker):
            raise RuntimeError("SwiftVR encode stage is not initialized")
        out_h, out_w, pad_h, pad_w = self._ensure_sizes(int(frames_uint8.shape[1]), int(frames_uint8.shape[2]))
        encoded_wait = self.pipeline.encode_stage.process(
            frames_uint8,
            out_h,
            out_w,
            pad_h,
            pad_w,
            self.pipeline.upscale_mode,
            _tensor_transport=True,
        )
        return self._run_encoded(encoded_wait(), pad_h, pad_w)

    @torch.inference_mode()
    def flush(self) -> list[Image.Image]:
        self._ensure_open()
        if self._sizes is None:
            return []
        if not isinstance(self.pipeline.encode_stage, ParallelWorker):
            raise RuntimeError("SwiftVR encode stage is not initialized")
        _, _, pad_h, pad_w = self._sizes
        encoded_wait = self.pipeline.encode_stage.flush_encoder(_tensor_transport=True)
        return self._run_encoded(encoded_wait(), pad_h, pad_w)

    def close(self) -> None:
        if self._closed:
            return
        for stage in (self.pipeline.decode_stage, self.pipeline.dit_stage, self.pipeline.encode_stage):
            if isinstance(stage, ParallelWorker):
                stage.reset_session(sync=True)
        self._sizes = None
        self._closed = True
        self.pipeline._stage_parallel_active_session = False


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
    def step(self, frames_uint8: torch.Tensor) -> list[Image.Image]:
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
                return []
            decoded = self._tae.decode_chunk(self._run_latents(encoded))
            return ntchw_to_pil_frames(crop_spatial_padding_ntchw(decoded, pad_h, pad_w))

    @torch.inference_mode()
    def flush(self) -> list[Image.Image]:
        self._ensure_open()
        with self.pipeline._execution_lock:
            encoded = self._tae.flush_encoder()
            if encoded is None or self._sizes is None:
                return []
            _, _, pad_h, pad_w = self._sizes
            decoded = self._tae.decode_chunk(self._run_latents(encoded))
            return ntchw_to_pil_frames(crop_spatial_padding_ntchw(decoded, pad_h, pad_w))

    def close(self) -> None:
        if self._closed:
            return
        self._tae.reset()
        self._dit.reset()
        self._dit._cond_cache = None
        self._dit._cond_cache_key = None
        self._sizes = None
        self._closed = True
