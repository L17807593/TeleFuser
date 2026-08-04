# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

try:
    from examples.minimax_h3.common import (
        MINIMAX_H3_DEFAULT_REF2VA_AUDIO,
        MINIMAX_H3_DEFAULT_REF2VA_VIDEO,
        load_minimax_h3_pipeline,
        save_generation,
    )
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from common import (
        MINIMAX_H3_DEFAULT_REF2VA_AUDIO,
        MINIMAX_H3_DEFAULT_REF2VA_VIDEO,
        load_minimax_h3_pipeline,
        save_generation,
    )


def default_ref2va_conditions() -> list[dict[str, str]]:
    """Return the frozen video-then-audio reference order used by the example."""
    return [
        {"type": "video", "role": "reference", "uri": str(MINIMAX_H3_DEFAULT_REF2VA_VIDEO)},
        {"type": "audio", "role": "reference", "uri": str(MINIMAX_H3_DEFAULT_REF2VA_AUDIO)},
    ]


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
    parser.add_argument("--aspect-ratio", choices=("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"), default="auto")
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--audio-flow-shift", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--enable-fsdp", action="store_true")
    parser.add_argument("--output", default="minimax_h3_ref2va.mp4")
    args = parser.parse_args()

    conditions = [
        *({"type": "image", "role": "reference", "uri": path} for path in args.image),
        *({"type": "video", "role": "reference", "uri": path} for path in args.video),
        *({"type": "audio", "role": "reference", "uri": path} for path in args.audio),
    ]
    if not conditions:
        conditions = default_ref2va_conditions()
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition="Ref2VA",
        device=args.device,
        num_inference_steps=args.steps,
        ulysses_degree=args.ulysses_degree,
        enable_fsdp=args.enable_fsdp,
    )
    try:
        result = pipeline(
            task="ref2va",
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
