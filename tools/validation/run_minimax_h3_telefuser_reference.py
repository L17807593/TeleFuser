# SPDX-License-Identifier: Apache-2.0
"""Run a frozen MiniMax H3 request with TeleFuser and capture raw outputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import runpy
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__:
    from tools.validation.minimax_h3_validation_common import (
        json_sha256 as _json_sha256,
    )
    from tools.validation.minimax_h3_validation_common import (
        model_config_hashes as _model_config_hashes,
    )
    from tools.validation.minimax_h3_validation_common import (
        sha256 as _sha256,
    )
else:
    from minimax_h3_validation_common import (
        json_sha256 as _json_sha256,
    )
    from minimax_h3_validation_common import (
        model_config_hashes as _model_config_hashes,
    )
    from minimax_h3_validation_common import (
        sha256 as _sha256,
    )


def _array_summary(array: np.ndarray) -> dict[str, object]:
    values = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _cpu_capture(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {str(key): _cpu_capture(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_capture(item) for item in value)
    if isinstance(value, list):
        return [_cpu_capture(item) for item in value]
    return value


def _captured_hidden_states(value: object) -> torch.Tensor:
    if not isinstance(value, Mapping):
        raise TypeError("captured text condition must be a mapping")
    hidden_states = value.get("hidden_states")
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    positive = value.get("positive")
    if isinstance(positive, Mapping):
        hidden_states = positive.get("hidden_states")
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
    raise KeyError("captured text condition has no hidden_states tensor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3"))
    parser.add_argument("--partition", choices=("FL2VA", "Ref2VA"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--target-duration-seconds", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--text-condition-artifact", type=Path)
    parser.add_argument("--text-condition-input", type=Path)
    parser.add_argument("--denoise-boundary-artifact", type=Path)
    parser.add_argument("--dit-layer-artifact", type=Path)
    parser.add_argument("--dit-layer-input", type=Path)
    parser.add_argument("--trajectory-artifact", type=Path)
    parser.add_argument("--trajectory-max-updates", type=int)
    args = parser.parse_args()
    run_started = time.perf_counter()

    if args.trajectory_max_updates is not None and args.trajectory_artifact is None:
        parser.error("--trajectory-max-updates requires --trajectory-artifact")

    request_path = args.request.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("frozen request must be a JSON object")
    allowed_tasks = {"FL2VA": {"t2va", "fl2va"}, "Ref2VA": {"ref2va"}}
    if request.get("task") not in allowed_tasks[args.partition]:
        raise ValueError(f"request task {request.get('task')!r} is incompatible with {args.partition}")
    if args.target_duration_seconds is not None:
        if not 4 <= args.target_duration_seconds <= 15:
            raise ValueError("target duration must be in [4, 15] seconds")
        request["target"] = {**request["target"], "duration_seconds": args.target_duration_seconds}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_artifact is not None:
        from minimax_h3_trajectory_stage import MiniMaxH3TrajectoryDenoisingStage

        from telefuser.pipelines.minimax_h3 import pipeline as pipeline_module

        trajectory_path = args.trajectory_artifact.resolve()

        def build_trajectory_stage(module_manager: object, model_runtime_config: object) -> object:
            return MiniMaxH3TrajectoryDenoisingStage(
                module_manager,
                model_runtime_config,
                trajectory_path=trajectory_path,
                max_updates=args.trajectory_max_updates,
            )

        pipeline_module.MiniMaxH3DenoisingStage = build_trajectory_stage
    load_started = time.perf_counter()
    common = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "minimax_h3" / "common.py"))
    load_minimax_h3_pipeline = common["load_minimax_h3_pipeline"]
    save_generation = common["save_generation"]
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition=args.partition,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        ulysses_degree=args.ulysses_degree,
    )
    load_seconds = time.perf_counter() - load_started
    if args.text_condition_artifact is not None or args.text_condition_input is not None:
        if pipeline.text_stage is None:
            raise RuntimeError("MiniMax H3 text stage is not initialized")
        injected_condition = (
            None
            if args.text_condition_input is None
            else torch.load(args.text_condition_input, map_location="cpu", weights_only=True)
        )
        if args.text_condition_artifact is not None:
            args.text_condition_artifact.parent.mkdir(parents=True, exist_ok=True)
        original_encode = pipeline.text_stage.encode

        def capture_text_condition(**kwargs: object) -> object:
            ref2va_encoder_inputs: dict[str, object] = {}
            original_encode_ids = pipeline.text_stage.text_encoder.encode_ids
            if request["task"] == "ref2va":

                def capture_encode_ids(input_ids: torch.Tensor, **encoder_kwargs: object) -> torch.Tensor:
                    ref2va_encoder_inputs["input_ids"] = input_ids.detach().cpu()
                    ref2va_encoder_inputs["processor"] = _cpu_capture(encoder_kwargs)
                    return original_encode_ids(input_ids, **encoder_kwargs)

                pipeline.text_stage.text_encoder.encode_ids = capture_encode_ids
            try:
                condition = original_encode(**kwargs)
            finally:
                pipeline.text_stage.text_encoder.encode_ids = original_encode_ids
            if injected_condition is not None:
                hidden_states = _captured_hidden_states(injected_condition)
                if tuple(hidden_states.shape) != tuple(condition.hidden_states.shape):
                    raise ValueError(
                        "injected text hidden shape "
                        f"{tuple(hidden_states.shape)} != encoded shape {tuple(condition.hidden_states.shape)}"
                    )
                condition = replace(
                    condition,
                    hidden_states=hidden_states.to(dtype=condition.hidden_states.dtype),
                )
            payload = {
                "hidden_states": condition.hidden_states.detach().cpu(),
                "token_tags": condition.token_tags.detach().cpu(),
            }
            if request["task"] == "t2va":
                from telefuser.pipelines.minimax_h3.presentation import minimax_h3_text_only_ids

                payload["input_ids"] = minimax_h3_text_only_ids(
                    pipeline.text_stage.tokenizer,
                    request["prompt"],
                )
            elif request["task"] == "fl2va":
                from telefuser.pipelines.minimax_h3.presentation import minimax_h3_multi_image_presentation

                images = kwargs["images"]
                vision = pipeline.text_stage.processor.image_processor(images=images, return_tensors="pt")
                grids = vision["image_grid_thw"]
                merge = int(pipeline.text_stage.processor.image_processor.merge_size) ** 2
                counts = [int(grids[index].prod().item()) // merge for index in range(len(images))]
                input_ids, input_tags = minimax_h3_multi_image_presentation(
                    pipeline.text_stage.tokenizer,
                    prompt=request["prompt"],
                    image_token_counts=counts,
                )
                if not torch.equal(input_tags, condition.token_tags):
                    raise RuntimeError("reconstructed FL2VA token tags differ from the encoded condition")
                payload["input_ids"] = input_ids
                payload["processor"] = _cpu_capture(vision)
            elif request["task"] == "ref2va":
                payload.update(ref2va_encoder_inputs)
            if args.text_condition_artifact is not None:
                torch.save(payload, args.text_condition_artifact)
            return condition

        pipeline.text_stage.encode = capture_text_condition
    if args.denoise_boundary_artifact is not None:
        if args.ulysses_degree != 1:
            raise ValueError("--denoise-boundary-artifact currently requires --ulysses-degree 1")
        if pipeline.denoising_stage is None:
            raise RuntimeError("MiniMax H3 denoising stage is not initialized")
        args.denoise_boundary_artifact.parent.mkdir(parents=True, exist_ok=True)
        captured_boundaries: dict[str, object] = {}
        transformer = pipeline.denoising_stage.transformer
        original_transformer_forward = transformer.forward

        def capture_transformer_forward(*forward_args: object, **forward_kwargs: object) -> object:
            capture_first_step = "dit_inputs" not in captured_boundaries
            if capture_first_step:
                captured_boundaries["dit_inputs"] = {
                    "args": _cpu_capture(forward_args),
                    "kwargs": _cpu_capture(forward_kwargs),
                }
            output = original_transformer_forward(*forward_args, **forward_kwargs)
            if capture_first_step:
                captured_boundaries["dit_outputs"] = _cpu_capture(output)
            return output

        transformer.forward = capture_transformer_forward
        original_denoise = pipeline.denoising_stage.denoise

        def capture_denoise(**kwargs: object) -> object:
            denoised = original_denoise(**kwargs)
            captured_boundaries["final_video_latent"] = _cpu_capture(denoised.video_latent)
            captured_boundaries["final_audio_latent"] = _cpu_capture(denoised.audio_latent)
            captured_boundaries["packed"] = _cpu_capture(denoised.packed)
            torch.save(captured_boundaries, args.denoise_boundary_artifact)
            return denoised

        pipeline.denoising_stage.denoise = capture_denoise
    captured_layers: dict[str, object] = {}
    if args.dit_layer_artifact is not None or args.dit_layer_input is not None:
        if args.ulysses_degree != 1:
            raise ValueError("DiT layer capture and injection currently require --ulysses-degree 1")
        if pipeline.denoising_stage is None:
            raise RuntimeError("MiniMax H3 denoising stage is not initialized")
        if args.dit_layer_artifact is not None:
            args.dit_layer_artifact.parent.mkdir(parents=True, exist_ok=True)
        injected_layers = (
            None
            if args.dit_layer_input is None
            else torch.load(args.dit_layer_input, map_location="cpu", weights_only=False)
        )
        transformer = pipeline.denoising_stage.transformer

        def capture_output(name: str):
            def hook(_module: object, _inputs: object, output: object) -> None:
                if name not in captured_layers:
                    captured_layers[name] = _cpu_capture(output)

            return hook

        def capture_block_input(
            _module: object,
            inputs: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]] | None:
            if "block_0_input" in captured_layers:
                return
            hidden = inputs[0]
            effective_kwargs = kwargs
            if injected_layers is not None:
                injected_input = injected_layers["block_0_input"]
                hidden = injected_input["hidden"].to(device=hidden.device, dtype=hidden.dtype)
                effective_kwargs = {
                    **kwargs,
                    "adaln_input": injected_input["adaln_input"].to(
                        device=kwargs["adaln_input"].device,
                        dtype=kwargs["adaln_input"].dtype,
                    ),
                    "combined_indices": injected_input["combined_indices"].to(
                        device=kwargs["combined_indices"].device,
                        dtype=kwargs["combined_indices"].dtype,
                    ),
                }
            captured_layers["block_0_input"] = {
                "hidden": _cpu_capture(hidden),
                "adaln_input": _cpu_capture(effective_kwargs["adaln_input"]),
                "combined_indices": _cpu_capture(effective_kwargs["combined_indices"]),
                "rope_frequencies": _cpu_capture(effective_kwargs["rope_frequencies"]),
            }
            if injected_layers is not None:
                return (hidden,), effective_kwargs

        transformer.condition_proj.register_forward_hook(capture_output("condition_proj"))
        transformer.token_refiner.register_forward_hook(capture_output("token_refiner"))
        transformer.video_patch_proj.register_forward_hook(capture_output("video_patch_proj"))
        transformer.audio_patch_proj.register_forward_hook(capture_output("audio_patch_proj"))
        transformer.time_embedder.register_forward_hook(capture_output("time_embedder"))
        transformer.rope.register_forward_hook(capture_output("rope"))
        transformer.blocks[0].register_forward_pre_hook(capture_block_input, with_kwargs=True)
        transformer.blocks[0].attn.register_forward_hook(capture_output("block_0_attention"))
        transformer.blocks[0].mlp.register_forward_hook(capture_output("block_0_mlp"))
        for index in (0, len(transformer.blocks) // 2, len(transformer.blocks) - 1):
            transformer.blocks[index].register_forward_hook(capture_output(f"block_{index}_output"))
        transformer.final_layer.register_forward_hook(capture_output("final_layer"))

    torch.cuda.reset_peak_memory_stats(args.device)
    started_at = datetime.now(timezone.utc)
    generation_started = time.perf_counter()
    try:
        result = pipeline(
            task=request["task"],
            prompt=request["prompt"],
            conditions=request.get("conditions"),
            target=request["target"],
            seed=request.get("seed"),
            flow_shift=12.0,
            audio_flow_shift=3.0,
            num_inference_steps=args.num_inference_steps,
        )
        torch.cuda.synchronize(args.device)
        if args.dit_layer_artifact is not None:
            torch.save(captured_layers, args.dit_layer_artifact)
        generation_compute_seconds = time.perf_counter() - generation_started
    finally:
        shutdown_started = time.perf_counter()
        pipeline.stop()
        shutdown_seconds = time.perf_counter() - shutdown_started
    generation_seconds = time.perf_counter() - generation_started
    finished_at = datetime.now(timezone.utc)
    artifact_started = time.perf_counter()

    frames = result.video[0].mul(255).clamp(0, 255).to(torch.uint8).numpy()
    audio = result.audio[0].float().numpy()
    frames_path = args.artifact_dir / "frames_uint8.npy"
    audio_path = args.artifact_dir / "audio_float32.npy"
    np.save(frames_path, frames, allow_pickle=False)
    np.save(audio_path, audio, allow_pickle=False)
    save_generation(result, args.output)

    packages = {}
    for name in ("telefuser", "torch", "transformers", "diffusers", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 2,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "generation_seconds": generation_seconds,
        "telefuser_commit": git_commit,
        "request": str(request_path),
        "source_request_sha256": _sha256(request_path),
        "request_sha256": _json_sha256(request),
        "model_root": str(args.model_root.resolve()),
        "partition": args.partition,
        "model_config_sha256": _model_config_hashes(args.model_root.resolve() / args.partition),
        "seed": request.get("seed"),
        "precision": {
            "text_encoder": "bfloat16",
            "transformer": "bfloat16",
            "video_vae_weights": "float32",
            "video_vae_cuda_decode_autocast": "float16",
            "audio_vae": "float32",
        },
        "num_inference_steps": args.num_inference_steps,
        "device": args.device,
        "parallel": {"ulysses_degree": args.ulysses_degree, "device_count": args.ulysses_degree},
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        "plan": asdict(result.plan),
        "packed_sequence_length": result.packed_sequence_length,
        "runtime_metrics": result.runtime_metrics,
        "artifacts": {
            "frames": {**_array_summary(frames), "path": str(frames_path.resolve()), "sha256": _sha256(frames_path)},
            "audio": {**_array_summary(audio), "path": str(audio_path.resolve()), "sha256": _sha256(audio_path)},
            "mp4": {
                "path": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "sha256": _sha256(args.output),
                "mux_policy": "complete_generated_audio",
            },
        },
    }
    if args.text_condition_artifact is not None:
        manifest["artifacts"]["text_condition"] = {
            "path": str(args.text_condition_artifact.resolve()),
            "sha256": _sha256(args.text_condition_artifact),
        }
    if args.text_condition_input is not None:
        manifest["text_condition_input"] = {
            "path": str(args.text_condition_input.resolve()),
            "sha256": _sha256(args.text_condition_input),
        }
    if args.denoise_boundary_artifact is not None:
        manifest["artifacts"]["denoise_boundaries"] = {
            "path": str(args.denoise_boundary_artifact.resolve()),
            "sha256": _sha256(args.denoise_boundary_artifact),
        }
    if args.dit_layer_artifact is not None:
        manifest["artifacts"]["dit_layers"] = {
            "path": str(args.dit_layer_artifact.resolve()),
            "sha256": _sha256(args.dit_layer_artifact),
        }
    if args.dit_layer_input is not None:
        manifest["dit_layer_input"] = {
            "path": str(args.dit_layer_input.resolve()),
            "sha256": _sha256(args.dit_layer_input),
        }
    if args.trajectory_artifact is not None:
        manifest["artifacts"]["trajectory"] = {
            "path": str(args.trajectory_artifact.resolve()),
            "sha256": _sha256(args.trajectory_artifact),
        }
        manifest["trajectory_max_updates"] = args.trajectory_max_updates
    timings = {
        "load_seconds": load_seconds,
        "generation_compute_seconds": generation_compute_seconds,
        "shutdown_seconds": shutdown_seconds,
        "generation_seconds": generation_seconds,
        "artifact_seconds": time.perf_counter() - artifact_started,
        "total_seconds": time.perf_counter() - run_started,
    }
    manifest["timings"] = timings
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"timings": timings, "runtime_metrics": result.runtime_metrics, "artifacts": manifest["artifacts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
