# SPDX-License-Identifier: Apache-2.0
"""Calibrate AdaTaylorCache for the MiniMax H3 joint audio-video DiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from telefuser.feature_cache import AdaTaylorCacheCalibrator
from telefuser.pipelines.minimax_h3.example_utils import load_minimax_h3_pipeline, save_generation

try:
    from .minimax_h3_fl2va_h100 import PPL_CONFIG, run
except ImportError:
    from minimax_h3_fl2va_h100 import PPL_CONFIG, run

MODEL_TYPE = "MiniMax-H3-Base"
DEFAULT_MAX_CONSECUTIVE_SKIPS = 2
DEFAULT_RETENTION_RATIO = 0.2
DEFAULT_SCHEDULE_THRESHOLD = 0.03
DEFAULT_VIDEO_FLOW_SHIFT = 12.0
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "telefuser"
    / "feature_cache"
    / "ada_taylor_cache"
    / "params"
    / f"{MODEL_TYPE}.json"
)


def _apply_cache_profile(
    output_path: str | Path,
    *,
    max_consecutive_skips: int,
    retention_ratio: float,
    schedule_threshold: float,
) -> None:
    if max_consecutive_skips < 1:
        raise ValueError("max_consecutive_skips must be at least 1")
    if not 0.0 <= retention_ratio <= 1.0:
        raise ValueError("retention_ratio must be between 0 and 1")
    if schedule_threshold <= 0.0:
        raise ValueError("schedule_threshold must be positive")

    path = Path(output_path)
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "K": max_consecutive_skips,
            "retention_ratio": retention_ratio,
            "thresh": schedule_threshold,
        }
    )
    path.write_text(json.dumps(params, indent=4) + "\n", encoding="utf-8")


def calibrate(
    *,
    model_root: str,
    output_path: str,
    prompt: str = PPL_CONFIG["prompt"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    duration_seconds: float = 4.0,
    seed: int = PPL_CONFIG["seed"],
    flow_shift: float | None = PPL_CONFIG["flow_shift"],
    audio_flow_shift: float | None = PPL_CONFIG["audio_flow_shift"],
    max_consecutive_skips: int = DEFAULT_MAX_CONSECUTIVE_SKIPS,
    retention_ratio: float = DEFAULT_RETENTION_RATIO,
    schedule_threshold: float = DEFAULT_SCHEDULE_THRESHOLD,
    preview_output: str | None = None,
) -> None:
    """Run one full-compute request and write reusable H3 cache parameters."""
    pipeline = load_minimax_h3_pipeline(
        model_root,
        partition="FL2VA",
        device=PPL_CONFIG["device"],
        num_inference_steps=num_inference_steps,
        attn_impl=PPL_CONFIG["attn_impl"],
    )
    try:
        denoising_stage = pipeline.denoising_stage
        if denoising_stage is None or not hasattr(denoising_stage, "transformer"):
            raise RuntimeError("MiniMax H3 calibration requires the single-GPU denoising stage")
        sigma_shift = DEFAULT_VIDEO_FLOW_SHIFT if flow_shift is None else flow_shift
        denoising_steps = num_inference_steps - 1
        denoising_stage.transformer.set_ada_taylor_cache_calibrator(
            num_inference_steps=denoising_steps,
            sigma_shift=sigma_shift,
            model_name=MODEL_TYPE,
            output_path=output_path,
        )
        generation = run(
            pipeline,
            prompt=prompt,
            seed=seed,
            target_video_length=duration_seconds,
            mode="t2va",
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
        )
        cache = denoising_stage.transformer.feature_cache
        if (
            not isinstance(cache, AdaTaylorCacheCalibrator)
            or not cache.cond_calibrator.is_finished()
            or not cache.uncond_calibrator.is_finished()
            or not Path(output_path).is_file()
        ):
            raise RuntimeError("MiniMax H3 feature-cache calibration did not collect every denoising step")
        _apply_cache_profile(
            output_path,
            max_consecutive_skips=max_consecutive_skips,
            retention_ratio=retention_ratio,
            schedule_threshold=schedule_threshold,
        )
        if preview_output is not None:
            save_generation(generation, preview_output)
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MiniMax H3 AdaTaylorCache on one H100")
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument("--steps", type=int, default=PPL_CONFIG["num_inference_steps"])
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--flow-shift", type=float, default=PPL_CONFIG["flow_shift"])
    parser.add_argument("--audio-flow-shift", type=float, default=PPL_CONFIG["audio_flow_shift"])
    parser.add_argument("--max-consecutive-skips", type=int, default=DEFAULT_MAX_CONSECUTIVE_SKIPS)
    parser.add_argument("--retention-ratio", type=float, default=DEFAULT_RETENTION_RATIO)
    parser.add_argument("--schedule-threshold", type=float, default=DEFAULT_SCHEDULE_THRESHOLD)
    parser.add_argument("--preview-output")
    args = parser.parse_args()
    calibrate(
        model_root=args.model_root,
        output_path=args.output_path,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        duration_seconds=args.duration,
        seed=args.seed,
        flow_shift=args.flow_shift,
        audio_flow_shift=args.audio_flow_shift,
        max_consecutive_skips=args.max_consecutive_skips,
        retention_ratio=args.retention_ratio,
        schedule_threshold=args.schedule_threshold,
        preview_output=args.preview_output,
    )


if __name__ == "__main__":
    main()
