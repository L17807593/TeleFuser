"""Benchmark LingBot-World v2 steady chunks with SP/TP layouts."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import torch

from telefuser.service.livekit.pipeline_adapter import LiveKitPipelineAdapter
from telefuser.service.security.security_validator import SecurityLevel

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "examples/data/lingbot_world_fast/image.jpg"
DEFAULT_CONTROL_TRACE_PATH = PROJECT_ROOT / "benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json"
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
    "mountains under a bright blue sky with drifting white clouds. Gentle ripples reflect the tree and sky."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-degree", type=int, required=True)
    parser.add_argument("--frame-num", type=int, default=77)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--v2-model-root", type=Path, required=True)
    parser.add_argument("--gpu-num", type=int, default=4)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--control-trace", type=Path, default=DEFAULT_CONTROL_TRACE_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    p90_index = int(0.9 * (len(ordered) - 1))
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _sum_phase(profile: dict[str, Any], name: str) -> float | None:
    value = profile.get("phases", {}).get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        total = 0.0
        for item in value:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                return None
            total += float(item)
        return total
    return None


async def _send_controls(
    adapter: LiveKitPipelineAdapter,
    session_id: str,
    events: list[dict[str, Any]],
    started_at: float,
) -> None:
    for event in events:
        delay = started_at + float(event["delay_s"]) - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        adapter.push_chunk(session_id, dict(event["message"]))


def _write_pipeline_wrapper(args: argparse.Namespace, directory: Path) -> Path:
    example_path = PROJECT_ROOT / "examples/lingbot/lingbot_world_v2_image_to_video_h100.py"
    wrapper = directory / f"lingbot_world_v2_tp{args.tp_degree}_pipeline.py"
    wrapper.write_text(
        textwrap.dedent(
            f"""
            import runpy

            EXAMPLE_PATH = {str(example_path.resolve())!r}
            MODEL_ROOT = {str(args.model_root.resolve())!r}
            V2_MODEL_ROOT = {str(args.v2_model_root.resolve())!r}
            TP_DEGREE = {args.tp_degree!r}
            FRAME_NUM = {args.frame_num!r}
            example = runpy.run_path(EXAMPLE_PATH)


            def get_service(gpu_num: int = 4):
                service_factory = example["get_service"]
                service_globals = service_factory.__globals__
                service_globals["PPL_CONFIG"]["frame_num"] = FRAME_NUM
                original_get_pipeline = service_globals["get_pipeline"]

                def get_pipeline(
                    parallelism=service_globals["PPL_CONFIG"]["parallelism"],
                    model_root=None,
                    v2_model_root=None,
                    tp_degree=1,
                ):
                    return original_get_pipeline(
                        parallelism=parallelism,
                        model_root=MODEL_ROOT,
                        v2_model_root=V2_MODEL_ROOT,
                        tp_degree=TP_DEGREE,
                    )

                service_globals["get_pipeline"] = get_pipeline
                try:
                    return service_factory(gpu_num=gpu_num)
                finally:
                    service_globals["get_pipeline"] = original_get_pipeline
            """
        ).lstrip()
    )
    return wrapper


def _collect_stage_memory(adapter: LiveKitPipelineAdapter) -> dict[str, Any] | None:
    service = adapter.stream_service.service
    pipeline = getattr(service, "pipeline", None)
    if pipeline is None or not hasattr(pipeline, "stage_memory_snapshots"):
        return None
    return pipeline.stage_memory_snapshots()


async def _run(args: argparse.Namespace, pipeline_file: Path) -> dict[str, Any]:
    trace = json.loads(args.control_trace.read_text())
    events = trace["events"]
    adapter = LiveKitPipelineAdapter(security_level=SecurityLevel.NONE)
    adapter.start(str(pipeline_file), skip_validation=True, gpu_num=args.gpu_num)
    capacity = adapter.configure_session_capacity(2)
    session_id = adapter.create_session(
        {
            "prompt": args.prompt,
            "image_path": str(args.image.resolve()),
            "fps": args.fps,
            "chunk_size": args.chunk_size,
            "frame_num": args.frame_num,
            "max_duration_seconds": 60.0,
            "sample_shift": 10.0,
            "control_mode": "cam",
            "show_control_hud": False,
            "benchmark_metrics": True,
        }
    )
    started_at = time.monotonic()
    sender = asyncio.create_task(_send_controls(adapter, session_id, events, started_at))
    frames = 0
    chunk_profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    first_preview_at: float | None = None
    first_generated_frame_at: float | None = None
    stage_memory: dict[str, Any] | None = None
    try:
        async for payload in adapter.pull_chunks(session_id):
            payload_type = payload.get("type")
            if payload_type in {"preview", "chunk"}:
                payload_frames = payload.get("frames", [])
                if payload_type == "chunk":
                    frames += len(payload_frames)
                    if payload_frames and first_generated_frame_at is None:
                        first_generated_frame_at = time.monotonic()
                elif payload_frames and first_preview_at is None:
                    first_preview_at = time.monotonic()
            if payload_type != "status":
                continue
            if payload.get("stage") == "runtime_ready":
                runtime = payload.get("runtime")
            measurement = payload.get("measurement")
            if isinstance(measurement, dict) and "index" in measurement:
                chunk_profiles.append(measurement)
        await sender
        stage_memory = _collect_stage_memory(adapter)
    finally:
        if not sender.done():
            sender.cancel()
        await adapter.aclose()

    elapsed = time.monotonic() - started_at
    steady = [profile for profile in chunk_profiles if int(profile["index"]) > 0]
    compute_seconds = [float(profile["compute_seconds"]) for profile in steady]
    denoise_seconds = [
        value for profile in steady if (value := _sum_phase(profile, "denoise_gpu_span_seconds")) is not None
    ]
    dmd_step_seconds = [
        value for profile in steady if (value := _sum_phase(profile, "dmd_step_seconds")) is not None
    ]
    scalar_phase_names = sorted(
        {
            name
            for profile in steady
            for name, value in profile.get("phases", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    steady_frames = sum(float(profile["frames"]) for profile in steady)
    return {
        "environment": {
            "sys_executable": sys.executable,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "request": {
            "mode": f"sp{args.gpu_num // args.tp_degree}_tp{args.tp_degree}",
            "gpu_num": args.gpu_num,
            "tp_degree": args.tp_degree,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "chunk_size": args.chunk_size,
            "image": str(args.image.resolve()),
            "control_trace": str(args.control_trace.resolve()),
            "model_root": str(args.model_root.resolve()),
            "v2_model_root": str(args.v2_model_root.resolve()),
        },
        "transport": "direct pipeline service; no LiveKit room, pacing, codec, or client",
        "capacity": capacity,
        "runtime": runtime,
        "result": {
            "frames": frames,
            "chunks": len(chunk_profiles),
            "steady_chunks": len(steady),
            "elapsed_seconds": elapsed,
            "first_preview_seconds": None if first_preview_at is None else first_preview_at - started_at,
            "first_generated_frame_seconds": (
                None if first_generated_frame_at is None else first_generated_frame_at - started_at
            ),
            "steady_compute_seconds": sum(compute_seconds),
            "steady_compute_fps": steady_frames / sum(compute_seconds),
            "steady_denoise_gpu_seconds": sum(denoise_seconds),
            "steady_denoise_gpu_fps": steady_frames / sum(denoise_seconds),
            "steady_dmd_step_seconds": sum(dmd_step_seconds),
            "steady_dmd_step_fps": steady_frames / sum(dmd_step_seconds),
            "chunk_compute_seconds": _summary(compute_seconds),
            "denoise_gpu_span_seconds": _summary(denoise_seconds),
            "dmd_step_total_seconds": _summary(dmd_step_seconds),
            "phases": {
                name: _summary([float(profile["phases"][name]) for profile in steady if name in profile["phases"]])
                for name in scalar_phase_names
            },
        },
        "stage_memory": stage_memory,
        "chunk_profiles": chunk_profiles,
    }


def main() -> None:
    args = _parse_args()
    with tempfile.TemporaryDirectory(prefix="lingbot_world_v2_tp_bench_") as temp_dir:
        pipeline_file = _write_pipeline_wrapper(args, Path(temp_dir))
        result = asyncio.run(_run(args, pipeline_file))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["result"], indent=2, sort_keys=True))
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()
