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
    parser = argparse.ArgumentParser(description="Generate MiniMax H3 Ref2VA audio-video on H100 GPUs")
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--output", default="minimax_h3_ref2va.mp4")
    args = parser.parse_args()

    conditions = [
        *({"type": "image", "role": "reference", "uri": path} for path in args.image),
        *({"type": "video", "role": "reference", "uri": path} for path in args.video),
        *({"type": "audio", "role": "reference", "uri": path} for path in args.audio),
    ]
    if not conditions:
        parser.error("at least one --image, --video, or --audio reference is required")
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition="Ref2VA",
        device=args.device,
        num_inference_steps=args.steps,
        ulysses_degree=args.ulysses_degree,
    )
    try:
        result = pipeline(
            task="ref2va",
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
