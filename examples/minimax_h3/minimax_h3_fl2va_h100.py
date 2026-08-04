# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

try:
    from examples.minimax_h3.common import (
        MINIMAX_H3_DEFAULT_FL2VA_IMAGE,
        load_minimax_h3_pipeline,
        save_generation,
    )
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from common import MINIMAX_H3_DEFAULT_FL2VA_IMAGE, load_minimax_h3_pipeline, save_generation


def build_fl2va_conditions(
    *,
    mode: str | None,
    image: str | None,
    last_image: str | None,
) -> list[dict[str, object]]:
    """Build the three supported FL2VA keyframe signatures, preserving legacy inference."""
    if mode is None:
        if image and last_image:
            mode = "first-last"
        elif image:
            mode = "first-frame"
        elif last_image:
            mode = "last-frame"
        else:
            mode = "t2va"
    if mode == "t2va":
        if image or last_image:
            raise ValueError("--mode t2va does not accept --image or --last-image")
        return []

    default_image = str(MINIMAX_H3_DEFAULT_FL2VA_IMAGE)
    if mode == "first-frame":
        return [{"type": "image", "role": "keyframe", "uri": image or default_image, "frame_index": 0}]
    if mode == "last-frame":
        return [
            {
                "type": "image",
                "role": "keyframe",
                "uri": last_image or image or default_image,
                "frame_index": -1,
            }
        ]
    if mode == "first-last":
        first = image or default_image
        last = last_image or image or default_image
        return [
            {"type": "image", "role": "keyframe", "uri": first, "frame_index": 0},
            {"type": "image", "role": "keyframe", "uri": last, "frame_index": -1},
        ]
    raise ValueError(f"unsupported FL2VA mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MiniMax H3 T2VA/FL2VA audio-video on H100 GPUs")
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument("--mode", choices=("t2va", "first-frame", "last-frame", "first-last"))
    parser.add_argument("--image")
    parser.add_argument("--last-image")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--aspect-ratio", choices=("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"), default="auto")
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--audio-flow-shift", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--enable-fsdp", action="store_true")
    parser.add_argument("--output", default="minimax_h3_fl2va.mp4")
    args = parser.parse_args()

    try:
        conditions = build_fl2va_conditions(mode=args.mode, image=args.image, last_image=args.last_image)
    except ValueError as exc:
        parser.error(str(exc))
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition="FL2VA",
        device=args.device,
        num_inference_steps=args.steps,
        ulysses_degree=args.ulysses_degree,
        enable_fsdp=args.enable_fsdp,
    )
    try:
        result = pipeline(
            task="fl2va" if conditions else "t2va",
            prompt=args.prompt,
            conditions=conditions,
            target={"short_edge": 768, "aspect_ratio": args.aspect_ratio, "duration_seconds": args.duration},
            seed=args.seed,
            flow_shift=args.flow_shift,
            audio_flow_shift=args.audio_flow_shift,
        )
        save_generation(result, args.output)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
