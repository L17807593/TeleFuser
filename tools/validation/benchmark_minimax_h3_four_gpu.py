# SPDX-License-Identifier: Apache-2.0
"""Benchmark the documented warm MiniMax H3 four-GPU profile."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import threading
import time
from pathlib import Path

from examples.minimax_h3.minimax_h3_fl2va_h100 import PPL_CONFIG, get_pipeline, run
from telefuser.core.config import AttnImplType


def _sample_device_memory(stop: threading.Event, peaks_mib: list[int]) -> None:
    while not stop.is_set():
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in output.splitlines():
            index, used_mib = (int(value.strip()) for value in line.split(","))
            if index < len(peaks_mib):
                peaks_mib[index] = max(peaks_mib[index], used_mib)
        stop.wait(0.1)


def _generate(pipeline: object) -> object:
    return run(
        pipeline,
        prompt=PPL_CONFIG["prompt"],
        seed=0,
        aspect_ratio="16:9",
        target_video_length=5,
        mode="t2va",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument(
        "--attention",
        choices=("FLASH_ATTN_4", "SAGE_ATTN_2_8_8_SM90"),
        default="FLASH_ATTN_4",
    )
    parser.add_argument("--feature-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pipeline = get_pipeline(
        4,
        args.model_root,
        num_inference_steps=50,
        online_adaln_cache=True,
        attn_impl=AttnImplType[args.attention],
        enable_feature_cache=args.feature_cache,
    )
    try:
        warmup = _generate(pipeline)
        del warmup
        gc.collect()

        peaks_mib = [0, 0, 0, 0]
        stop_sampling = threading.Event()
        sampler = threading.Thread(
            target=_sample_device_memory,
            args=(stop_sampling, peaks_mib),
            daemon=True,
        )
        sampler.start()
        try:
            started = time.perf_counter()
            result = _generate(pipeline)
            wall_seconds = time.perf_counter() - started
        finally:
            stop_sampling.set()
            sampler.join()

        payload = {
            "attention": args.attention,
            "feature_cache": args.feature_cache,
            "wall_seconds": wall_seconds,
            "peak_memory_mib": peaks_mib,
            "runtime_metrics": result.runtime_metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
