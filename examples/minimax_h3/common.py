# SPDX-License-Identifier: Apache-2.0
"""Shared local-checkpoint loader and artifact writer for MiniMax H3 examples."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch

from telefuser.core.config import ModelRuntimeConfig, OffloadConfig, ParallelConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
from telefuser.models.minimax_h3_dit import MiniMaxH3DiT
from telefuser.models.minimax_h3_encoder import MiniMaxH3Encoder
from telefuser.models.minimax_h3_video_vae import MiniMaxH3VideoVAE
from telefuser.pipelines.minimax_h3.pipeline import (
    MiniMaxH3Generation,
    MiniMaxH3Pipeline,
    MiniMaxH3PipelineConfig,
)
from telefuser.utils.audio import save_wav
from telefuser.utils.video import save_video


def _checkpoint_shards(component: Path) -> list[str]:
    shards = sorted(str(path) for path in component.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model safetensor shards found in {component}")
    return shards


def load_minimax_h3_pipeline(
    model_root: str | Path,
    *,
    partition: str,
    device: str = "cuda:0",
    num_inference_steps: int = 50,
    ulysses_degree: int = 1,
) -> MiniMaxH3Pipeline:
    if partition not in {"FL2VA", "Ref2VA"}:
        raise ValueError("partition must be 'FL2VA' or 'Ref2VA'")
    if ulysses_degree not in {1, 2, 4}:
        raise ValueError("ulysses_degree must be 1, 2, or 4")
    component_root = Path(model_root) / partition
    if not component_root.is_dir():
        raise FileNotFoundError(f"MiniMax H3 partition not found: {component_root}")
    runtime_device = torch.device(device)
    offload = OffloadConfig(
        offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD,
        pin_cpu_memory=False,
    )
    bf16_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.bfloat16,
        offload_config=offload,
    )
    dit_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.bfloat16,
        offload_config=offload,
        parallel_config=ParallelConfig(
            device_ids=list(range(ulysses_degree)),
            sp_ulysses_degree=ulysses_degree,
            timeout=1800,
        ),
    )
    fp32_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.float32,
        offload_config=offload,
    )

    manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    transformer_dir = component_root / "transformer"
    manager.load_model(
        _checkpoint_shards(transformer_dir),
        device="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        name="minimax_h3_transformer",
        model_class=MiniMaxH3DiT,
        converter_kwargs={"config_path": transformer_dir / "config.json"},
    )
    encoder_dir = component_root / "text_encoder"
    manager.load_model(
        _checkpoint_shards(encoder_dir),
        device="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        name="minimax_h3_text_encoder",
        model_class=MiniMaxH3Encoder,
        converter_kwargs={"config_path": encoder_dir},
    )
    video_vae_dir = component_root / "video_vae"
    manager.load_model(
        str(video_vae_dir / "source" / "model.safetensors"),
        device="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        name="minimax_h3_video_vae",
        model_class=MiniMaxH3VideoVAE,
        converter_kwargs={"config_path": video_vae_dir},
    )
    audio_vae_dir = component_root / "audio_vae"
    manager.load_model(
        str(audio_vae_dir / "model.safetensors"),
        device="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        name="minimax_h3_audio_vae",
        model_class=MiniMaxH3AudioVAE,
        converter_kwargs={"config_path": audio_vae_dir},
    )

    pipeline = MiniMaxH3Pipeline(device=device)
    pipeline.init(
        manager,
        MiniMaxH3PipelineConfig(
            processor_path=str(component_root / "processor"),
            text_encoder_config=bf16_runtime,
            dit_config=dit_runtime,
            video_vae_config=fp32_runtime,
            audio_vae_config=fp32_runtime,
            num_inference_steps=num_inference_steps,
        ),
    )
    return pipeline


def save_generation(result: MiniMaxH3Generation, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = result.video[0].mul(255).clamp(0, 255).to(torch.uint8)
    waveform = result.audio[0]
    with tempfile.TemporaryDirectory() as directory:
        video_path = Path(directory) / "video.mp4"
        audio_path = Path(directory) / "audio.wav"
        save_wav(waveform, result.audio_sample_rate, str(audio_path))
        save_video(
            frames,
            str(video_path),
            fps=float(result.video_fps),
            quality=6,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(output),
            ],
            check=True,
            capture_output=True,
        )


__all__ = ["load_minimax_h3_pipeline", "save_generation"]
