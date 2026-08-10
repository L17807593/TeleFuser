"""SwiftVR streaming video-restoration example."""

from __future__ import annotations

import os
import time

import click
import numpy as np
import torch
from PIL import Image

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    CompileConfig,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
)
from telefuser.pipelines.swiftvr import SwiftVRPipeline
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import VideoData, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="swiftvr_restore_h100",
    model_root=TF_MODEL_ZOO_PATH + "/SwiftVR",
    scale=4,
    fps=16,
    video_quality=6,
    chunk_size=24,
    warmup_chunks=2,
    dit_overlap=0,
    attn_impl=AttnImplType.TORCH_SDPA,
    compile_mode="default",
)


def _parse_attn_impl(attn_impl: AttnImplType | str) -> AttnImplType:
    if isinstance(attn_impl, AttnImplType):
        return attn_impl
    key = attn_impl.upper().replace("-", "_")
    try:
        return AttnImplType[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported attention backend: {attn_impl}") from exc


def _quant_config(quantization: str | None) -> QuantConfig | None:
    if quantization is None:
        return None
    normalized = quantization.lower().replace("_", "-")
    if normalized == "torchao-fp8":
        return QuantConfig(enabled=True, quant_type=QuantType.TORCHAO_FP8, kernel_backend=QuantKernelBackend.TORCHAO)
    if normalized == "tf-kernel-fp8":
        return QuantConfig(enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL)
    raise ValueError("quantization must be 'torchao-fp8', 'tf-kernel-fp8', or None")


def _parse_stage_devices(stage_devices: str | None, parallelism: int) -> list[int] | None:
    if stage_devices:
        devices = [int(item.strip()) for item in stage_devices.split(",") if item.strip()]
    elif parallelism > 1:
        devices = list(range(parallelism))
    else:
        return None
    if len(devices) != 3:
        raise ValueError("SwiftVR stage parallelism requires exactly three devices: encode,dit,decode")
    return devices


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    attn_impl: AttnImplType | str = PPL_CONFIG["attn_impl"],
    compile_dit: bool = False,
    compile_mode: str = PPL_CONFIG["compile_mode"],
    quantization: str | None = None,
    enable_stage_parallel: bool = False,
    stage_devices: str | None = None,
    tensor_channel_slots: int = 2,
) -> SwiftVRPipeline:
    """Initialize the SwiftVR pipeline."""
    stage_device_ids = _parse_stage_devices(stage_devices, parallelism) if enable_stage_parallel else None
    if not enable_stage_parallel and parallelism != 1:
        raise ValueError("SwiftVR supports multiple GPUs only through --enable_stage_parallel")
    return SwiftVRPipeline.from_pretrained(
        model_root,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attention_config=AttentionConfig.dense_attention(_parse_attn_impl(attn_impl)),
        compile_config=CompileConfig(enabled=compile_dit, mode=compile_mode) if compile_dit else None,
        quant_config=_quant_config(quantization),
        enable_stage_parallel=enable_stage_parallel,
        enable_stage_overlap=enable_stage_parallel,
        stage_dit_overlap=PPL_CONFIG["dit_overlap"],
        stage_device_ids=stage_device_ids,
        tensor_channel_slots=tensor_channel_slots,
    )


def run(
    pipeline: SwiftVRPipeline,
    input_video: list[Image.Image],
    scale: int = PPL_CONFIG["scale"],
) -> list[Image.Image]:
    """Restore loaded PIL frames through a stateful streaming session."""
    if not input_video:
        raise ValueError("input_video must contain at least one frame")
    width, height = input_video[0].size
    crop_width, crop_height = width // 8 * 8, height // 8 * 8
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"input frames are too small: {width}x{height}")
    arrays = [
        np.asarray(frame.convert("RGB").crop((0, 0, crop_width, crop_height)), dtype=np.uint8) for frame in input_video
    ]
    frames = torch.from_numpy(np.stack(arrays)).contiguous()
    if pipeline.config.enable_stage_parallel:
        return pipeline(
            frames,
            upscale=scale,
            clip_len=PPL_CONFIG["chunk_size"],
            dit_overlap=PPL_CONFIG["dit_overlap"],
        )
    restored_frames: list[Image.Image] = []
    session = pipeline.stream(upscale=scale, dit_overlap=PPL_CONFIG["dit_overlap"])
    try:
        for start in range(0, len(frames), PPL_CONFIG["chunk_size"]):
            restored_frames.extend(session.step(frames[start : start + PPL_CONFIG["chunk_size"]]))
        restored_frames.extend(session.flush())
    finally:
        session.close()
    return restored_frames


def _warmup_frame_count(frame_count: int) -> int:
    full_chunks = PPL_CONFIG["chunk_size"] * PPL_CONFIG["warmup_chunks"]
    tail_frames = frame_count % PPL_CONFIG["chunk_size"]
    return min(frame_count, full_chunks + tail_frames)


@click.command()
@click.option(
    "--input_video",
    "-i",
    default=f"{os.path.dirname(__file__)}/../data/dag.mp4",
    help="Path to input low-quality video",
)
@click.option("--scale", "-s", default=PPL_CONFIG["scale"], type=int, help="Upscaling factor (default: 4)")
@click.option("--height", "-h", default=None, type=int, help="Input video height (default: auto-detect)")
@click.option("--width", "-w", default=None, type=int, help="Input video width (default: auto-detect)")
@click.option("--gpu_num", default=1, type=int, help="Number of GPUs to use (default: 1)")
@click.option(
    "--model_root",
    default=PPL_CONFIG["model_root"],
    help=f"Root directory containing model files (default: {PPL_CONFIG['model_root']})",
)
@click.option("--output", "-o", default=None, help="Output video path (default: auto-generated)")
@click.option(
    "--attn_impl",
    default=PPL_CONFIG["attn_impl"].name,
    type=click.Choice([item.name for item in AttnImplType], case_sensitive=False),
    help="DiT attention backend",
)
@click.option("--compile_dit/--no_compile_dit", default=False, help="Enable torch.compile for SwiftVR DiT blocks")
@click.option(
    "--compile_mode",
    default=PPL_CONFIG["compile_mode"],
    type=click.Choice(
        ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        case_sensitive=False,
    ),
    help="torch.compile mode / kernel fusion preset",
)
@click.option("--quantization", type=click.Choice(["torchao-fp8", "tf-kernel-fp8"]), default=None)
@click.option("--enable_stage_parallel", is_flag=True, help="Split encode, DiT, and decode into worker stages")
@click.option("--stage_devices", default=None, help="Comma-separated encode,dit,decode GPU ids, e.g. 0,1,2")
@click.option("--tensor_channel_slots", default=2, type=int, help="CUDA IPC slots per stage tensor-channel profile")
def main(
    input_video: str,
    scale: int,
    height: int | None,
    width: int | None,
    gpu_num: int,
    model_root: str,
    output: str | None,
    attn_impl: str,
    compile_dit: bool,
    compile_mode: str,
    quantization: str | None,
    enable_stage_parallel: bool,
    stage_devices: str | None,
    tensor_channel_slots: int,
) -> None:
    """SwiftVR streaming video restoration."""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if output is None:
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        filename = get_example_name(__file__).replace(".mp4", f"_scale{scale}_{gpu_num}gpu.mp4")
        output = os.path.join(output_dir, filename)

    click.echo(f"Input video: {input_video}")
    click.echo(f"Input resolution: {width or 'auto'}x{height or 'auto'}")
    click.echo(f"Scale: {scale}x")
    click.echo(f"GPUs: {gpu_num}")
    click.echo(f"Model root: {model_root}")
    click.echo(f"Attention: {attn_impl}")
    click.echo(f"Compile DiT: {compile_dit}")
    click.echo(f"Compile mode: {compile_mode}")
    click.echo(f"Quantization: {quantization or 'none'}")
    click.echo(f"Stage parallel: {enable_stage_parallel}")
    if stage_devices:
        click.echo(f"Stage devices: {stage_devices}")
    click.echo(f"Output: {output}")

    click.echo("Loading pipeline...")
    pipeline = get_pipeline(
        gpu_num,
        model_root,
        attn_impl=attn_impl,
        compile_dit=compile_dit,
        compile_mode=compile_mode,
        quantization=quantization,
        enable_stage_parallel=enable_stage_parallel,
        stage_devices=stage_devices,
        tensor_channel_slots=tensor_channel_slots,
    )

    click.echo("Loading video...")
    input_frames = VideoData(video_file=input_video, height=height, width=width).raw_data()
    click.echo(f"Total frames: {len(input_frames)}")

    warmup_frame_count = _warmup_frame_count(len(input_frames))
    click.echo(f"Warmup pass ({warmup_frame_count} frames)...")
    run(pipeline, input_frames[:warmup_frame_count], scale=scale)

    click.echo("Processing video...")
    start_time = time.perf_counter()
    restored = run(pipeline, input_frames, scale=scale)
    processing_seconds = time.perf_counter() - start_time
    processing_fps = len(restored) / processing_seconds
    click.echo(f"Processing time: {processing_seconds:.2f} seconds ({processing_fps:.2f} FPS)")

    click.echo(f"Saving to {output}...")
    encoding_started = time.perf_counter()
    save_video(restored, output, fps=PPL_CONFIG["fps"], quality=PPL_CONFIG["video_quality"])
    encoding_seconds = time.perf_counter() - encoding_started
    end_to_end_fps = len(restored) / (processing_seconds + encoding_seconds)
    click.echo(f"Encoding time: {encoding_seconds:.2f} seconds")
    click.echo(f"End-to-end throughput: {end_to_end_fps:.2f} FPS")
    click.echo("Done!")


if __name__ == "__main__":
    main()
