"""Run the faithful single-GPU LTX-2.5 distilled T2V or I2V pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import click
import torch
from PIL import Image

from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.ltx25_distilled import (
    LTX25DistilledOutput,
    LTX25DistilledPipeline,
    LTX25ImageCondition,
    build_ltx25_distilled_config,
    load_ltx25_distilled_modules,
)
from telefuser.utils.audio import save_wav
from telefuser.utils.video import save_video

PPL_CONFIG: dict[str, Any] = {
    "name": "ltx25_distilled_t2v_i2v_h100",
    "model_root": "/hhb-data/aigc/model_zoo/Lightricks/LTX-2.5/LTX-2.5",
    "height": 1024,
    "width": 1536,
    "num_frames": 121,
    "frame_rate": 24.0,
    "seed": 42,
    "video_vae": "diff",
}


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    video_vae: str = PPL_CONFIG["video_vae"],
    offload: str = "cpu",
) -> LTX25DistilledPipeline:
    """Load the isolated LTX-2.5 distilled pipeline on one H100."""
    if parallelism != 1:
        raise ValueError("LTX-2.5 distilled currently supports one GPU")
    if video_vae not in ("diff", "conv"):
        raise ValueError(f"video_vae must be 'diff' or 'conv', got {video_vae!r}")
    if offload not in ("none", "cpu"):
        raise ValueError(f"offload must be 'none' or 'cpu', got {offload!r}")
    selected_video_vae = cast(Literal["diff", "conv"], video_vae)
    selected_offload = cast(Literal["none", "cpu"], offload)
    module_manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    load_ltx25_distilled_modules(
        module_manager,
        model_root,
        video_vae=selected_video_vae,
        torch_dtype=torch.bfloat16,
    )
    pipeline = LTX25DistilledPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipeline.init(
        module_manager,
        build_ltx25_distilled_config(
            "cuda",
            torch.bfloat16,
            selected_video_vae,
            selected_offload,
        ),
    )
    return pipeline


def run(
    pipeline: LTX25DistilledPipeline,
    prompt: str,
    *,
    seed: int = PPL_CONFIG["seed"],
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    num_frames: int = PPL_CONFIG["num_frames"],
    frame_rate: float = PPL_CONFIG["frame_rate"],
    image_path: str | None = None,
    image_frame_index: int = 0,
    image_strength: float = 1.0,
) -> LTX25DistilledOutput:
    """Generate T2V, or I2V when ``image_path`` is supplied."""
    images = ()
    if image_path is not None:
        images = (
            LTX25ImageCondition(
                Image.open(image_path).convert("RGB"),
                frame_idx=image_frame_index,
                strength=image_strength,
            ),
        )
    return pipeline(
        prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=images,
    )


def run_with_file(
    pipeline: LTX25DistilledPipeline,
    prompt: str,
    output_path: str,
    seed: int = PPL_CONFIG["seed"],
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    num_frames: int = PPL_CONFIG["num_frames"],
    frame_rate: float = PPL_CONFIG["frame_rate"],
    image_path: str | None = None,
    image_frame_index: int = 0,
    image_strength: float = 1.0,
    **_: object,
) -> dict[str, str]:
    """Generate and save an MP4 with synchronized LTX-2.5 audio."""
    result = run(
        pipeline,
        prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        image_path=image_path,
        image_frame_index=image_frame_index,
        image_strength=image_strength,
    )
    frames = torch.cat(result.video_chunks).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
        audio_path = Path(stream.name)
    try:
        save_wav(result.audio, 48000, str(audio_path))
        save_video(list(frames), str(destination), fps=result.frame_rate, quality=6, audio_path=str(audio_path))
    finally:
        audio_path.unlink(missing_ok=True)
    return {"output_path": str(destination)}


@click.command()
@click.option("--prompt", required=True)
@click.option("--model-root", default=PPL_CONFIG["model_root"], show_default=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
@click.option("--image-path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--image-frame-index", default=0, show_default=True)
@click.option("--image-strength", default=1.0, show_default=True)
@click.option("--height", default=PPL_CONFIG["height"], show_default=True)
@click.option("--width", default=PPL_CONFIG["width"], show_default=True)
@click.option("--num-frames", default=PPL_CONFIG["num_frames"], show_default=True)
@click.option("--frame-rate", default=PPL_CONFIG["frame_rate"], show_default=True)
@click.option("--seed", default=PPL_CONFIG["seed"], show_default=True)
@click.option("--video-vae", type=click.Choice(["diff", "conv"]), default=PPL_CONFIG["video_vae"], show_default=True)
@click.option("--offload", type=click.Choice(["none", "cpu"]), default="cpu", show_default=True)
def main(
    prompt: str,
    model_root: str,
    output_path: Path,
    image_path: Path | None,
    image_frame_index: int,
    image_strength: float,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    video_vae: str,
    offload: str,
) -> None:
    """Generate an LTX-2.5 video with synchronized audio."""
    pipeline = get_pipeline(model_root=model_root, video_vae=video_vae, offload=offload)
    run_with_file(
        pipeline,
        prompt,
        str(output_path),
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        image_path=None if image_path is None else str(image_path),
        image_frame_index=image_frame_index,
        image_strength=image_strength,
    )


if __name__ == "__main__":
    main()
