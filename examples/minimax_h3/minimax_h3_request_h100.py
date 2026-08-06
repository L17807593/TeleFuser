# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from examples.minimax_h3.common import (
    MINIMAX_H3_DEFAULT_REQUEST,
    load_minimax_h3_pipeline,
    load_minimax_h3_request,
    partition_for_minimax_h3_request,
    save_generation,
)
from telefuser.pipelines.minimax_h3.pipeline import MiniMaxH3Generation, MiniMaxH3Pipeline

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG: dict[str, Any] = {
    "name": "minimax_h3_request_h100",
    "model_root": TF_MODEL_ZOO_PATH + "/MiniMaxAI_MiniMax-H3",
    "request_path": str(MINIMAX_H3_DEFAULT_REQUEST),
    "num_inference_steps": None,
    "device": "cuda:0",
    "enable_fsdp": None,
    "adaln_cache_path": None,
    "online_adaln_cache": False,
}


def _load_request(request_path: str, num_inference_steps: int | None) -> dict[str, Any]:
    request = load_minimax_h3_request(request_path)
    if num_inference_steps is not None:
        request["num_inference_steps"] = num_inference_steps
    return request


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    request_path: str = PPL_CONFIG["request_path"],
    device: str = PPL_CONFIG["device"],
    num_inference_steps: int | None = PPL_CONFIG["num_inference_steps"],
    enable_fsdp: bool | None = PPL_CONFIG["enable_fsdp"],
    adaln_cache_path: str | None = PPL_CONFIG["adaln_cache_path"],
    online_adaln_cache: bool = PPL_CONFIG["online_adaln_cache"],
) -> MiniMaxH3Pipeline:
    """Load the checkpoint partition required by a local JSON request."""
    request = _load_request(request_path, num_inference_steps)
    configured_steps = int(request.get("num_inference_steps", 50))
    tp_degree = 2 if parallelism == 4 else 1
    return load_minimax_h3_pipeline(
        model_root,
        partition=partition_for_minimax_h3_request(request),
        device=device,
        num_inference_steps=configured_steps,
        ulysses_degree=parallelism // tp_degree,
        tp_degree=tp_degree,
        text_encoder_tp_degree=parallelism,
        enable_fsdp=enable_fsdp,
        adaln_cache_path=adaln_cache_path,
        online_adaln_cache=online_adaln_cache,
    )


def run(
    pipeline: MiniMaxH3Pipeline,
    request_path: str = PPL_CONFIG["request_path"],
    num_inference_steps: int | None = PPL_CONFIG["num_inference_steps"],
) -> MiniMaxH3Generation:
    """Run a complete JSON request without changing its condition order."""
    return pipeline(**_load_request(request_path, num_inference_steps))


def run_with_file(
    pipeline: MiniMaxH3Pipeline,
    request_path: str = PPL_CONFIG["request_path"],
    output_path: str = "minimax_h3_request.mp4",
    num_inference_steps: int | None = PPL_CONFIG["num_inference_steps"],
    **_: object,
) -> dict[str, str]:
    """Run a complete JSON request and save its synchronized MP4."""
    result = run(pipeline, request_path, num_inference_steps)
    save_generation(result, output_path)
    return {"output_path": str(Path(output_path))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a complete ordered MiniMax H3 JSON request on H100 GPUs")
    parser.add_argument("--request", dest="request_path", default=PPL_CONFIG["request_path"])
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--steps", type=int, default=PPL_CONFIG["num_inference_steps"])
    parser.add_argument("--device", default=PPL_CONFIG["device"])
    parser.add_argument("--adaln-cache", dest="adaln_cache_path", default=PPL_CONFIG["adaln_cache_path"])
    parser.add_argument("--online-adaln-cache", action="store_true", default=PPL_CONFIG["online_adaln_cache"])
    parser.add_argument("--gpu-num", "--ulysses-degree", dest="gpu_num", type=int, choices=(1, 2, 4), default=1)
    fsdp_group = parser.add_mutually_exclusive_group()
    fsdp_group.add_argument("--enable-fsdp", dest="enable_fsdp", action="store_true")
    fsdp_group.add_argument("--disable-fsdp", dest="enable_fsdp", action="store_false")
    parser.set_defaults(enable_fsdp=PPL_CONFIG["enable_fsdp"])
    parser.add_argument("--output-path", "--output", dest="output_path", default="minimax_h3_request.mp4")
    args = parser.parse_args()

    pipeline = get_pipeline(
        args.gpu_num,
        args.model_root,
        request_path=args.request_path,
        device=args.device,
        num_inference_steps=args.steps,
        enable_fsdp=args.enable_fsdp,
        adaln_cache_path=args.adaln_cache_path,
        online_adaln_cache=args.online_adaln_cache,
    )
    try:
        result = run_with_file(
            pipeline,
            request_path=args.request_path,
            output_path=args.output_path,
            num_inference_steps=args.steps,
        )
        print("Output saved to {}".format(result["output_path"]))
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
