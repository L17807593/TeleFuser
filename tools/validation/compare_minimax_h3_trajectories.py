# SPDX-License-Identifier: Apache-2.0
"""Compare mapped stable boundaries from official MiniMax H3 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


TensorPath = str | tuple[str, ...]


def _at(value: Any, path: TensorPath) -> Any:
    if isinstance(path, tuple):
        return torch.cat([_at(value, part) for part in path], dim=0)
    current = value
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, (list, tuple)) else current[part]
    return current


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        return {**result, "comparable": False, "reason": "shape_mismatch"}

    reference_flat = reference.detach().cpu().reshape(-1)
    candidate_flat = candidate.detach().cpu().reshape(-1)
    count = reference_flat.numel()
    exact = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_square = 0.0
    sum_relative = 0.0
    dot = 0.0
    reference_square = 0.0
    candidate_square = 0.0
    reference_min = math.inf
    reference_max = -math.inf
    relative_floor = 1e-6
    chunk_size = 1024 * 1024
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        reference_native = reference_flat[start:stop]
        candidate_native = candidate_flat[start:stop]
        exact += int(torch.eq(reference_native, candidate_native).sum())
        reference_chunk = reference_native.to(torch.float64)
        candidate_chunk = candidate_native.to(torch.float64)
        difference = candidate_chunk - reference_chunk
        absolute = difference.abs()
        max_abs = max(max_abs, float(absolute.max())) if absolute.numel() else max_abs
        sum_abs += float(absolute.sum())
        sum_square += float(difference.square().sum())
        sum_relative += float((absolute / reference_chunk.abs().clamp_min(relative_floor)).sum())
        dot += float(torch.dot(reference_chunk, candidate_chunk))
        reference_square += float(reference_chunk.square().sum())
        candidate_square += float(candidate_chunk.square().sum())
        if reference_chunk.numel():
            reference_min = min(reference_min, float(reference_chunk.min()))
            reference_max = max(reference_max, float(reference_chunk.max()))

    rmse = math.sqrt(sum_square / count) if count else 0.0
    denominator = math.sqrt(reference_square * candidate_square)
    cosine = dot / denominator if denominator else 1.0
    data_range = reference_max - reference_min
    psnr = math.inf if rmse == 0 else 20.0 * math.log10((data_range or 1.0) / rmse)
    return {
        **result,
        "comparable": True,
        "count": count,
        "exact_fraction": exact / count if count else 1.0,
        "max_abs_error": max_abs,
        "mean_abs_error": sum_abs / count if count else 0.0,
        "mean_relative_error": sum_relative / count if count else 0.0,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "psnr_db_reference_range": psnr,
        "reference_l2": math.sqrt(reference_square),
        "candidate_l2": math.sqrt(candidate_square),
    }


def _mapping(
    name: str,
    source: str,
    reference_path: TensorPath,
    candidate_path: TensorPath,
    *,
    expected_relation: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "reference_path": reference_path,
        "candidate_path": candidate_path,
        "expected_relation": expected_relation,
    }


def _mappings(case: str) -> list[dict[str, Any]]:
    mappings = [
        _mapping("processor.input_ids", "text", "input_ids", "input_ids", expected_relation="exact"),
        _mapping(
            "encoder.hidden_states",
            "text",
            "positive.hidden_states",
            "hidden_states",
            expected_relation="numeric",
        ),
        _mapping(
            "encoder.token_tags",
            "text",
            "positive.text_token_tags",
            "token_tags",
            expected_relation="exact",
        ),
    ]
    if case == "fl2va":
        mappings += [
            _mapping(
                "processor.pixel_values",
                "text",
                "processor.pixel_values",
                "processor.pixel_values",
                expected_relation="exact",
            ),
            _mapping(
                "processor.image_grid_thw",
                "text",
                "processor.image_grid_thw",
                "processor.image_grid_thw",
                expected_relation="exact",
            ),
            _mapping(
                "condition_vae.visual_pre_noise",
                "trajectory",
                "conditions_pre_noise.visual",
                "conditions_pre_noise.0.visual_rows",
                expected_relation="numeric",
            ),
            _mapping(
                "condition_vae.visual_post_noise",
                "trajectory",
                "conditions_post_noise.visual",
                "condition_rows_post_noise.visual",
                expected_relation="numeric",
            ),
        ]
    elif case == "ref2va":
        mappings += [
            _mapping(
                "processor.pixel_values_videos",
                "text",
                "processor.pixel_values_videos",
                "processor.pixel_values_videos",
                expected_relation="exact",
            ),
            _mapping(
                "processor.video_grid_thw",
                "text",
                "processor.video_grid_thw",
                "processor.video_grid_thw",
                expected_relation="exact",
            ),
            _mapping(
                "condition_vae.visual_pre_noise",
                "trajectory",
                "conditions_pre_noise.visual",
                "conditions_pre_noise.0.visual_rows",
                expected_relation="numeric",
            ),
            _mapping(
                "condition_vae.audio_pre_noise",
                "trajectory",
                "conditions_pre_noise.audio",
                ("conditions_pre_noise.0.audio_rows", "conditions_pre_noise.1.audio_rows"),
                expected_relation="numeric",
            ),
            _mapping(
                "condition_vae.visual_post_noise",
                "trajectory",
                "conditions_post_noise.visual",
                "condition_rows_post_noise.visual",
                expected_relation="numeric",
            ),
            _mapping(
                "condition_vae.audio_post_noise",
                "trajectory",
                "conditions_post_noise.audio",
                "condition_rows_post_noise.audio",
                expected_relation="numeric",
            ),
        ]

    for field in ("img_pos", "audio_pos", "text_pos", "update_mask", "img_position_ids", "cu_seqlens"):
        mappings.append(
            _mapping(
                f"packed.{field}",
                "trajectory",
                f"packed.{field}",
                f"packed.{field}",
                expected_relation="exact",
            )
        )
    mappings.append(
        _mapping(
            "packed.token_tags",
            "trajectory",
            "packed.token_tags",
            "transformer_layout.block_token_tags",
            expected_relation="exact",
        )
    )
    if case == "ref2va":
        mappings.append(
            _mapping(
                "packed.audio_update_mask",
                "trajectory",
                "packed.audio_update_mask",
                "packed.audio_update_mask",
                expected_relation="exact",
            )
        )
    for field in (
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "img_pos_info.position_ids",
        "audio_pos_info.position_ids",
        "text_pos_info.position_ids",
        "img_pos_for_infer_output_info.position_ids",
        "packed_seq_params.cu_seqlens_q",
    ):
        mappings.append(
            _mapping(
                f"transformer_layout.{field}",
                "trajectory",
                f"transformer_layout.{field}",
                f"transformer_layout.{field}",
                expected_relation="exact",
            )
        )
    mappings += [
        _mapping(
            "initial_noise.video",
            "trajectory",
            "initial_rows.visual",
            "steps.0.scheduler_input.input_visual_latent",
            expected_relation="exact",
        ),
        _mapping(
            "initial_noise.audio",
            "trajectory",
            "initial_rows.audio",
            "steps.0.scheduler_input.input_audio_latent",
            expected_relation="exact",
        ),
    ]
    for step in ("0", "24", "48"):
        relation = "exact" if step == "0" else "numeric"
        video_prediction_path = (
            f"steps.{step}.dit_output.0" if case == "ref2va" else f"steps.{step}.scheduler_input.noise_pred_visual"
        )
        audio_prediction_path = (
            f"steps.{step}.dit_output.1" if case == "ref2va" else f"steps.{step}.scheduler_input.noise_pred_audio"
        )
        mappings += [
            _mapping(
                f"step.{step}.pre_model.video",
                "trajectory",
                f"steps.{step}.pre_model.video_target",
                f"steps.{step}.scheduler_input.input_visual_latent",
                expected_relation=relation,
            ),
            _mapping(
                f"step.{step}.pre_model.audio",
                "trajectory",
                f"steps.{step}.pre_model.audio_target",
                f"steps.{step}.scheduler_input.input_audio_latent",
                expected_relation=relation,
            ),
            _mapping(
                f"step.{step}.dit_prediction.video",
                "trajectory",
                f"steps.{step}.dit_output.0",
                video_prediction_path,
                expected_relation="numeric",
            ),
            _mapping(
                f"step.{step}.dit_prediction.audio",
                "trajectory",
                f"steps.{step}.dit_output.1",
                audio_prediction_path,
                expected_relation="numeric",
            ),
            _mapping(
                f"step.{step}.scheduler_output.video",
                "trajectory",
                f"steps.{step}.scheduler_output.video_target",
                f"steps.{step}.scheduler_output.output_visual_latent",
                expected_relation="numeric",
            ),
            _mapping(
                f"step.{step}.scheduler_output.audio",
                "trajectory",
                f"steps.{step}.scheduler_output.audio_target",
                f"steps.{step}.scheduler_output.output_audio_latent",
                expected_relation="numeric",
            ),
        ]
    mappings += [
        _mapping(
            "final_latent.video",
            "trajectory",
            "final_video_latent",
            "final_video_latent",
            expected_relation="numeric",
        ),
        _mapping(
            "final_latent.audio",
            "trajectory",
            "final_audio_latent",
            "final_audio_latent",
            expected_relation="numeric",
        ),
    ]
    return mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument("--reference-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument("--reference-text", type=Path, required=True)
    parser.add_argument("--candidate-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts = {
        "trajectory": (
            torch.load(args.reference_trajectory, map_location="cpu", weights_only=True),
            torch.load(args.candidate_trajectory, map_location="cpu", weights_only=True),
        ),
        "text": (
            torch.load(args.reference_text, map_location="cpu", weights_only=True),
            torch.load(args.candidate_text, map_location="cpu", weights_only=True),
        ),
    }
    comparisons = {}
    exact_checks = {}
    for item in _mappings(args.case):
        reference, candidate = artifacts[item["source"]]
        metrics = _tensor_metrics(
            _at(reference, item["reference_path"]),
            _at(candidate, item["candidate_path"]),
        )
        metrics["expected_relation"] = item["expected_relation"]
        comparisons[item["name"]] = metrics
        if item["expected_relation"] == "exact":
            exact_checks[item["name"]] = bool(metrics.get("comparable") and metrics.get("exact_fraction") == 1.0)

    report = {
        "schema_version": 1,
        "case": args.case,
        "reference": "pinned_sglang",
        "candidate": "telefuser",
        "artifacts": {
            "reference_trajectory": {
                "path": str(args.reference_trajectory.resolve()),
                "sha256": _sha256(args.reference_trajectory),
            },
            "candidate_trajectory": {
                "path": str(args.candidate_trajectory.resolve()),
                "sha256": _sha256(args.candidate_trajectory),
            },
            "reference_text": {
                "path": str(args.reference_text.resolve()),
                "sha256": _sha256(args.reference_text),
            },
            "candidate_text": {
                "path": str(args.candidate_text.resolve()),
                "sha256": _sha256(args.candidate_text),
            },
        },
        "exact_checks": exact_checks,
        "all_exact_boundaries_passed": all(exact_checks.values()),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=True))
    if not report["all_exact_boundaries_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
