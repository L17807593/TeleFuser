# SPDX-License-Identifier: Apache-2.0
"""Run a frozen MiniMax H3 request with the pinned SGLang reference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.validation.minimax_h3_validation_common import (
    json_sha256 as _json_sha256,
)
from tools.validation.minimax_h3_validation_common import (
    model_config_hashes as _model_config_hashes,
)
from tools.validation.minimax_h3_validation_common import (
    sha256 as _sha256,
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3"))
    parser.add_argument("--sglang-root", type=Path, default=Path("work_dirs/sglang"))
    parser.add_argument("--partition", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--target-duration-seconds", type=float)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument("--performance-mode", choices=("manual", "auto", "speed", "memory"), default="speed")
    parser.add_argument("--vae-cpu-offload", action="store_true")
    parser.add_argument("--dit-layerwise-offload", action="store_true")
    parser.add_argument("--attention-backend")
    parser.add_argument("--text-condition-artifact", type=Path)
    parser.add_argument("--trajectory-artifact", type=Path)
    parser.add_argument("--instrumentation-diff", type=Path)
    args = parser.parse_args()

    if args.num_inference_steps < 2:
        raise ValueError("MiniMax H3 requires at least two inference steps")
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("frozen request must be a JSON object")
    allowed_tasks = {"fl2va": {"t2va", "fl2va"}, "ref2va": {"ref2va"}}
    if request.get("task") not in allowed_tasks[args.partition]:
        raise ValueError(f"request task {request.get('task')!r} is incompatible with {args.partition}")
    if args.target_duration_seconds is not None:
        if not 4 <= args.target_duration_seconds <= 15:
            raise ValueError("target duration must be in [4, 15] seconds")
        request["target"] = {**request["target"], "duration_seconds": args.target_duration_seconds}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.instrumentation_diff is not None:
        instrumentation = subprocess.run(
            ["git", "-C", str(args.sglang_root.resolve()), "diff", "--binary"],
            check=True,
            capture_output=True,
        ).stdout
        if not instrumentation:
            raise ValueError("--instrumentation-diff requires a modified SGLang worktree")
        args.instrumentation_diff.parent.mkdir(parents=True, exist_ok=True)
        args.instrumentation_diff.write_bytes(instrumentation)

    import torch
    from sglang.multimodal_gen import DiffGenerator

    server_kwargs = {
        "model_path": str(args.model_root.resolve()),
        "model_id": "MiniMax-H3",
        "pipeline_class_name": "MiniMaxH3Pipeline",
        "model_variant": args.partition,
        "backend": "sglang",
        "performance_mode": args.performance_mode,
        "num_gpus": args.num_gpus,
        "tp_size": args.tp_size,
        "ulysses_degree": args.ulysses_degree,
        "enable_torch_compile": False,
        "enable_cfg_parallel": False,
        "warmup_mode": "off",
        "vae_cpu_offload": args.vae_cpu_offload,
        "dit_layerwise_offload": args.dit_layerwise_offload,
    }
    if args.attention_backend is not None:
        server_kwargs["attention_backend"] = args.attention_backend
    sampling_kwargs = {
        **request,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "save_output": True,
        "return_file_paths_only": args.artifact_dir is None,
        "output_path": str(args.output.resolve().parent),
        "output_file_name": args.output.name,
    }

    started_at = datetime.now(timezone.utc)
    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        result = generator.generate(sampling_params_kwargs=sampling_kwargs)
    finished_at = datetime.now(timezone.utc)
    if result is None or isinstance(result, list):
        raise RuntimeError(f"expected one SGLang result, got {type(result).__name__}")
    output_path = Path(result.output_file_path or args.output).resolve()
    if not output_path.is_file():
        raise RuntimeError(f"SGLang did not create the expected output: {output_path}")

    validation_artifacts: dict[str, dict[str, object]] = {}
    for name, destination, suffix in (
        ("text_condition", args.text_condition_artifact, ".text.pt"),
        ("trajectory", args.trajectory_artifact, ".trajectory.pt"),
    ):
        if destination is None:
            continue
        source = Path(f"{output_path}{suffix}")
        if not source.is_file():
            raise RuntimeError(f"instrumented SGLang did not create {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.move(source, destination)
        validation_artifacts[name] = {
            "path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    if args.instrumentation_diff is not None:
        validation_artifacts["instrumentation_diff"] = {
            "path": str(args.instrumentation_diff.resolve()),
            "bytes": args.instrumentation_diff.stat().st_size,
            "sha256": _sha256(args.instrumentation_diff),
        }

    raw_artifacts: dict[str, object] = {}
    if args.artifact_dir is not None:
        if not result.frames:
            raise RuntimeError("SGLang raw capture returned no frames")
        frames = np.stack([np.asarray(frame, dtype=np.uint8) for frame in result.frames])
        audio = (
            result.audio.detach().float().cpu().numpy()
            if isinstance(result.audio, torch.Tensor)
            else np.asarray(result.audio)
        )
        frames_path = args.artifact_dir / "frames_uint8.npy"
        audio_path = args.artifact_dir / "audio_float32.npy"
        np.save(frames_path, frames, allow_pickle=False)
        np.save(audio_path, audio.astype(np.float32, copy=False), allow_pickle=False)
        raw_artifacts = {
            "frames": {
                "path": str(frames_path.resolve()),
                "shape": list(frames.shape),
                "dtype": str(frames.dtype),
                "sha256": _sha256(frames_path),
            },
            "audio": {
                "path": str(audio_path.resolve()),
                "shape": list(audio.shape),
                "dtype": str(audio.dtype),
                "sha256": _sha256(audio_path),
            },
        }

    sglang_commit = subprocess.run(
        ["git", "-C", str(args.sglang_root.resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    packages = {}
    for name in ("sglang", "torch", "transformers", "diffusers", "flashinfer-python", "sglang-kernel"):
        packages[name] = importlib.metadata.version(name)
    device = torch.cuda.get_device_properties(0)
    partition_dir = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}[args.partition]
    manifest = {
        "schema_version": 2,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "sglang_commit": sglang_commit,
        "request": str(request_path),
        "source_request_sha256": _sha256(request_path),
        "request_sha256": _json_sha256(request),
        "model_config_sha256": _model_config_hashes(args.model_root.resolve() / partition_dir),
        "seed": request.get("seed"),
        "precision": {
            "text_encoder": "bfloat16",
            "transformer": "bfloat16",
            "video_vae_decode_autocast": "float16",
            "audio_vae": "float32",
        },
        "server_kwargs": server_kwargs,
        "sampling_kwargs": sampling_kwargs,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": {"name": device.name, "memory_bytes": device.total_memory, "count": torch.cuda.device_count()},
        "result": {
            "size": _json_value(result.size),
            "generation_time_seconds": result.generation_time,
            "peak_memory_mb": result.peak_memory_mb,
            "metrics": _json_value(result.metrics),
            "output_path": str(output_path),
            "output_bytes": output_path.stat().st_size,
            "output_sha256": _sha256(output_path),
            "raw_artifacts": raw_artifacts,
        },
        "validation_artifacts": validation_artifacts,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["result"], sort_keys=True))


if __name__ == "__main__":
    main()
