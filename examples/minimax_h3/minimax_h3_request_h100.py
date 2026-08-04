# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

try:
    from examples.minimax_h3.common import (
        MINIMAX_H3_DEFAULT_REQUEST,
        load_minimax_h3_pipeline,
        load_minimax_h3_request,
        partition_for_minimax_h3_request,
        save_generation,
    )
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from common import (
        MINIMAX_H3_DEFAULT_REQUEST,
        load_minimax_h3_pipeline,
        load_minimax_h3_request,
        partition_for_minimax_h3_request,
        save_generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a complete ordered MiniMax H3 JSON request on H100 GPUs")
    parser.add_argument("--request", default=str(MINIMAX_H3_DEFAULT_REQUEST))
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument("--steps", type=int, help="Override num_inference_steps from the request")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--enable-fsdp", action="store_true")
    parser.add_argument("--output", default="minimax_h3_request.mp4")
    args = parser.parse_args()

    request = load_minimax_h3_request(args.request)
    if args.steps is not None:
        request["num_inference_steps"] = args.steps
    configured_steps = int(request.get("num_inference_steps", 50))
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition=partition_for_minimax_h3_request(request),
        device=args.device,
        num_inference_steps=configured_steps,
        ulysses_degree=args.ulysses_degree,
        enable_fsdp=args.enable_fsdp,
    )
    try:
        result = pipeline(**request)
        save_generation(result, args.output)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
