# SPDX-License-Identifier: Apache-2.0
"""Validation-only MiniMax H3 denoising stage with trajectory capture."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.minimax_h3 import denoising as denoising_module
from telefuser.pipelines.minimax_h3.denoising import (
    MiniMaxH3DenoiseResult,
    MiniMaxH3DenoisingStage,
)
from telefuser.pipelines.minimax_h3.resolved_plan import MiniMaxH3ResolvedPlan
from telefuser.pipelines.minimax_h3.text_encoding import MiniMaxH3TextCondition
from telefuser.pipelines.minimax_h3.vae import MiniMaxH3PreparedCondition


def _cpu_capture(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _cpu_capture(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_capture(item) for item in value)
    if isinstance(value, list):
        return [_cpu_capture(item) for item in value]
    return value


def _condition_payload(condition: MiniMaxH3PreparedCondition) -> dict[str, Any]:
    return {
        "material": asdict(condition.material),
        "kind": condition.kind,
        "visual_rows": _cpu_capture(condition.visual_rows),
        "audio_rows": _cpu_capture(condition.audio_rows),
        "latent_t": condition.latent_t,
        "latent_h": condition.latent_h,
        "latent_w": condition.latent_w,
        "ref_audio_t": condition.ref_audio_t,
        "has_audio": condition.has_audio,
    }


class MiniMaxH3TrajectoryDenoisingStage(MiniMaxH3DenoisingStage):
    """Capture stable full-trajectory boundaries from the worker that owns the DiT."""

    def __init__(
        self,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        *,
        trajectory_path: str | Path,
        max_updates: int | None = None,
    ) -> None:
        super().__init__(module_manager, model_runtime_config)
        if max_updates is not None and max_updates < 1:
            raise ValueError("max_updates must be at least 1 when provided")
        self.trajectory_path = str(Path(trajectory_path).resolve())
        self.max_updates = max_updates

    def denoise(
        self,
        *,
        plan: MiniMaxH3ResolvedPlan,
        text: MiniMaxH3TextCondition,
        conditions: list[MiniMaxH3PreparedCondition],
        num_inference_steps: int,
    ) -> MiniMaxH3DenoiseResult:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        should_capture = rank == 0
        configured_num_updates = num_inference_steps - 1
        num_updates = min(configured_num_updates, self.max_updates or configured_num_updates)
        selected_steps = sorted({0, num_updates // 2, num_updates - 1})
        payload: dict[str, Any] = {
            "schema_version": 1,
            "rank": rank,
            "world_size": world_size,
            "num_inference_steps": num_inference_steps,
            "num_updates": num_updates,
            "configured_num_updates": configured_num_updates,
            "trajectory_truncated": num_updates != configured_num_updates,
            "selected_steps": selected_steps,
            "plan": asdict(plan),
            "text": {
                "hidden_states": _cpu_capture(text.hidden_states),
                "token_tags": _cpu_capture(text.token_tags),
            },
            "conditions_pre_noise": [_condition_payload(condition) for condition in conditions],
            "transformer_layout": None,
            "steps": {},
        }

        transformer_step = 0
        scheduler_step = 0
        original_transformer_forward = self.transformer.forward
        original_scheduler_step = self.scheduler.step_denoising
        original_time_shift_sigmas = denoising_module.minimax_h3_time_shift_sigmas

        def limited_time_shift_sigmas(*args: Any, **kwargs: Any) -> torch.Tensor:
            sigmas = original_time_shift_sigmas(*args, **kwargs)
            return sigmas[: num_updates + 1]

        def capture_transformer_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal transformer_step
            step = transformer_step
            if should_capture and payload["transformer_layout"] is None:
                layout_keys = (
                    "img_position_ids",
                    "unique_timesteps",
                    "inverse_indices",
                    "update_mask",
                    "update_audio_mask",
                    "img_pos_info",
                    "audio_pos_info",
                    "text_pos_info",
                    "img_pos_for_infer_output_info",
                    "packed_seq_params",
                    "block_token_tags",
                    "skip_mask_out_condition",
                )
                payload["transformer_layout"] = {key: _cpu_capture(kwargs[key]) for key in layout_keys if key in kwargs}
                video_positions = kwargs["img_pos_info"]["position_ids"]
                audio_positions = kwargs["audio_pos_info"]["position_ids"]
                video_rows = kwargs["x"][0].index_select(0, video_positions)
                audio_rows = kwargs["audio_x"][0].index_select(0, audio_positions)
                payload["condition_rows_post_noise"] = {
                    "visual": _cpu_capture(video_rows[~kwargs["update_mask"]]),
                    "audio": _cpu_capture(audio_rows[~kwargs["update_audio_mask"]]),
                }
            output = original_transformer_forward(*args, **kwargs)
            if should_capture and step in selected_steps:
                step_payload = payload["steps"].setdefault(str(step), {})
                step_payload["dit_output"] = _cpu_capture(output)
            transformer_step += 1
            return output

        def capture_scheduler_step(**kwargs: Any) -> dict[str, torch.Tensor]:
            nonlocal scheduler_step
            step = scheduler_step
            output = original_scheduler_step(**kwargs)
            if should_capture and step in selected_steps:
                scheduler_keys = (
                    "input_visual_latent",
                    "input_audio_latent",
                    "timestep",
                    "video_timestep",
                    "audio_timestep",
                    "noise_pred_visual",
                    "noise_pred_audio",
                    "sigma_curr",
                    "sigma_next",
                    "video_sigma_curr",
                    "video_sigma_next",
                    "audio_sigma_curr",
                    "audio_sigma_next",
                )
                step_payload = payload["steps"].setdefault(str(step), {})
                step_payload["scheduler_input"] = {
                    key: _cpu_capture(kwargs[key]) for key in scheduler_keys if key in kwargs
                }
                step_payload["scheduler_output"] = _cpu_capture(output)
            scheduler_step += 1
            return output

        self.transformer.forward = capture_transformer_forward
        self.scheduler.step_denoising = capture_scheduler_step
        if num_updates != configured_num_updates:
            denoising_module.minimax_h3_time_shift_sigmas = limited_time_shift_sigmas
        try:
            result = super().denoise(
                plan=plan,
                text=text,
                conditions=conditions,
                num_inference_steps=num_inference_steps,
            )
        finally:
            self.transformer.forward = original_transformer_forward
            self.scheduler.step_denoising = original_scheduler_step
            denoising_module.minimax_h3_time_shift_sigmas = original_time_shift_sigmas

        if should_capture:
            payload["packed"] = _cpu_capture(result.packed)
            payload["final_video_latent"] = _cpu_capture(result.video_latent)
            payload["final_audio_latent"] = _cpu_capture(result.audio_latent)
            payload["observed_transformer_steps"] = transformer_step
            payload["observed_scheduler_steps"] = scheduler_step
            trajectory_path = Path(self.trajectory_path)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, trajectory_path)
        return result


__all__ = ["MiniMaxH3TrajectoryDenoisingStage"]
