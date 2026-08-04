# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

try:
    from examples.minimax_h3.common import load_minimax_h3_pipeline, save_generation
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from common import load_minimax_h3_pipeline, save_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MiniMax H3 T2VA/FL2VA audio-video on H100 GPUs")
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument("--image")
    parser.add_argument("--last-image")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--output", default="minimax_h3_fl2va.mp4")
    args = parser.parse_args()

    if args.last_image and not args.image:
        parser.error("--last-image requires --image")
    conditions = []
    if args.image:
        conditions.append({"type": "image", "role": "keyframe", "uri": args.image, "frame_index": 0})
    if args.last_image:
        conditions.append({"type": "image", "role": "keyframe", "uri": args.last_image, "frame_index": -1})
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition="FL2VA",
        device=args.device,
        num_inference_steps=args.steps,
        ulysses_degree=args.ulysses_degree,
    )
    try:
        result = pipeline(
            task="fl2va" if conditions else "t2va",
            prompt=args.prompt,
            conditions=conditions,
            target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": args.duration},
            seed=args.seed,
        )
        save_generation(result, args.output)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
