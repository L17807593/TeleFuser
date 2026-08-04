# SPDX-License-Identifier: Apache-2.0
"""Packed single-branch MiniMax H3 denoising stage."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.distributed.fsdp import shard_model
from telefuser.utils.logging import logger

from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .resolved_plan import MiniMaxH3ResolvedPlan
from .scheduler import MiniMaxH3EulerAncestralEta0SchedulerAdapter
from .text_encoding import MiniMaxH3TextCondition
from .time_request import minimax_h3_time_shift_sigmas
from .vae import MiniMaxH3PreparedCondition

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0


@dataclass(frozen=True)
class MiniMaxH3DenoiseResult:
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    packed: dict[str, torch.Tensor]
    runtime_metrics: dict[str, float | int]


class MiniMaxH3DenoisingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__("minimax_h3_denoising", model_runtime_config)
        self.transformer = module_manager.fetch_module("minimax_h3_transformer")
        if self.transformer is None:
            raise ValueError("ModuleManager must contain 'minimax_h3_transformer'")
        self.scheduler = MiniMaxH3EulerAncestralEta0SchedulerAdapter()
        self.model_names = ["transformer"]

    def parallel_models(self) -> None:
        parallel_config = self.model_runtime_config.parallel_config
        unsupported = {
            "cfg_degree": parallel_config.cfg_degree,
            "sp_ring_degree": parallel_config.sp_ring_degree,
            "pp_degree": parallel_config.pp_degree,
            "tp_degree": parallel_config.tp_degree,
        }
        invalid = {name: degree for name, degree in unsupported.items() if degree != 1}
        if invalid:
            raise NotImplementedError(f"MiniMax H3 does not support these parallel degrees yet: {invalid}")
        device_mesh = create_device_mesh_from_config(parallel_config)
        self.transformer.device_mesh = device_mesh
        self.transformer.set_attention_config(self.model_runtime_config.attention_config)
        if parallel_config.sp_ulysses_degree > 1:
            self.transformer.enable_usp(device_mesh)
        if parallel_config.enable_fsdp:
            logger.info(f"Enabling FSDP for {self.name}")
            fp32_parameters = [
                parameter for parameter in self.transformer.parameters() if parameter.dtype == torch.float32
            ]
            self.transformer = shard_model(
                module=self.transformer,
                device_id=self.device,
                wrap_module_names=self.transformer.get_fsdp_module_names(),
                param_dtype=self.torch_dtype,
                reduce_dtype=self.torch_dtype,
                buffer_dtype=torch.float32,
                ignored_states=fp32_parameters,
            )
            self.onload_models_flag = True

    @staticmethod
    def _reference_blocks(
        conditions: list[MiniMaxH3PreparedCondition],
    ) -> tuple[list[dict[str, object]], torch.Tensor | None, torch.Tensor | None]:
        blocks: list[dict[str, object]] = []
        visual: list[torch.Tensor] = []
        audio: list[torch.Tensor] = []
        for condition in conditions:
            if condition.kind == "image":
                if condition.visual_rows is None:
                    raise ValueError("reference image is missing visual VAE rows")
                blocks.append(
                    {
                        "kind": "image",
                        "latent_h": condition.latent_h,
                        "latent_w": condition.latent_w,
                    }
                )
                visual.append(condition.visual_rows)
            elif condition.kind == "audio":
                if condition.audio_rows is None:
                    raise ValueError("reference audio is missing audio VAE rows")
                blocks.append({"kind": "audio", "ref_audio_t": condition.ref_audio_t})
                audio.append(condition.audio_rows)
            elif condition.kind in {"video", "video_audio"}:
                if condition.visual_rows is None:
                    raise ValueError("reference video is missing visual VAE rows")
                blocks.append(
                    {
                        "kind": condition.kind,
                        "ref_audio_t": condition.ref_audio_t,
                        "latent_t": condition.latent_t,
                        "latent_h": condition.latent_h,
                        "latent_w": condition.latent_w,
                    }
                )
                visual.append(condition.visual_rows)
                if condition.audio_rows is not None:
                    audio.append(condition.audio_rows)
            else:
                raise ValueError(f"unsupported reference block kind {condition.kind!r}")
        visual_rows = None if not visual else torch.cat(visual, dim=0)
        audio_rows = None if not audio else torch.cat(audio, dim=0)
        return blocks, visual_rows, audio_rows

    @with_model_offload(["transformer"])
    @torch.inference_mode()
    def denoise(
        self,
        *,
        plan: MiniMaxH3ResolvedPlan,
        text: MiniMaxH3TextCondition,
        conditions: list[MiniMaxH3PreparedCondition],
        num_inference_steps: int,
    ) -> MiniMaxH3DenoiseResult:
        shape = plan.shape
        latent_t = int(shape["video_latent_t"])
        latent_h = int(shape["height"]) // 16
        latent_w = int(shape["width"]) // 16
        audio_t = int(shape["audio_latent_t"])
        seed = 42 if plan.seed is None else int(plan.seed)
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")

        ref_blocks: list[dict[str, object]] | None = None
        if plan.task == "ref2va":
            ref_blocks, visual_cond, audio_cond = self._reference_blocks(conditions)
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=text.text_len,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        else:
            visual_conditions = [condition for condition in conditions if condition.visual_rows is not None]
            visual_cond = (
                None
                if not visual_conditions
                else torch.cat([condition.visual_rows for condition in visual_conditions], dim=0)
            )
            audio_cond = None
            semantic_indices = tuple(
                int(condition.material.frame_index)
                for condition in visual_conditions
                if condition.material.frame_index is not None
            )
            packed = minimax_h3_packed_sequence(
                text_len=text.text_len,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=bool(visual_conditions),
                keyframe_frame_indices=semantic_indices if visual_conditions else None,
                frame_count=int(shape["frame_count"]) if visual_conditions else None,
            )

        token_tags = packed["token_tags"].clone()
        token_tags[: text.text_len] = text.token_tags
        condition_shapes = [
            (condition.latent_t, condition.latent_h, condition.latent_w)
            for condition in conditions
            if condition.visual_rows is not None
        ]
        if visual_cond is not None:
            visual_cond = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_cond,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )
        audio_lengths = [condition.ref_audio_t for condition in conditions if condition.audio_rows is not None]
        if audio_cond is not None:
            audio_cond = minimax_h3_audio_cond_noise_aug_rows(
                audio_cond,
                condition_audio_t=audio_lengths,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )

        video_generator = torch.Generator(device="cpu").manual_seed(seed)
        video_native = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=video_generator,
            dtype=torch.float32,
        )
        video_target = minimax_h3_patchify_video_latent(video_native, patch_size=(1, 2, 2))
        audio_generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_target = torch.randn(audio_t * 2, 32, generator=audio_generator, dtype=torch.float32)

        device = next(self.transformer.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        reset_communication_metrics = getattr(self.transformer, "reset_communication_metrics", None)
        if callable(reset_communication_metrics):
            reset_communication_metrics()
        denoising_started = time.perf_counter()
        video_rows = torch.zeros(len(packed["img_pos"]), 96, dtype=torch.float32, device=device)
        audio_rows = torch.zeros(len(packed["audio_pos"]), 32, dtype=torch.float32, device=device)
        video_update_cpu = packed["update_mask"]
        audio_update_cpu = packed.get("audio_update_mask", torch.ones(len(packed["audio_pos"]), dtype=torch.bool))
        video_update = video_update_cpu.to(device)
        audio_update = audio_update_cpu.to(device)
        video_rows[video_update] = video_target.to(device)
        audio_rows[audio_update] = audio_target.to(device)
        if visual_cond is not None:
            video_rows[~video_update] = visual_cond.to(device)
        if audio_cond is not None:
            audio_rows[~audio_update] = audio_cond.to(device)

        video_shift = plan.flow_shift or plan.default_flow_shift
        audio_shift = plan.audio_flow_shift or plan.default_audio_flow_shift
        video_sigmas = minimax_h3_time_shift_sigmas(num_steps=num_inference_steps, shift_scale=video_shift)
        audio_sigmas = minimax_h3_time_shift_sigmas(num_steps=num_inference_steps, shift_scale=audio_shift)
        img_pos_cpu = packed["img_pos"]
        audio_pos_cpu = packed["audio_pos"]
        img_pos = img_pos_cpu.to(device)
        audio_pos = audio_pos_cpu.to(device)
        text_pos = packed["text_pos"].to(device)
        target_img_pos = img_pos[video_update]
        target_audio_row_start = int((~audio_update_cpu).sum())
        condition_img_pos = img_pos_cpu[~video_update_cpu]
        target_audio_pos = audio_pos_cpu[audio_update_cpu]
        condition_audio_pos = audio_pos_cpu[~audio_update_cpu]

        seq_len = int(packed["seq_len"])
        row_timesteps = torch.empty(seq_len, dtype=torch.float32)
        x = torch.zeros(1, seq_len, 96, dtype=torch.float32, device=device)
        audio_x = torch.zeros(1, seq_len, 32, dtype=torch.float32, device=device)
        img_position_ids = packed["img_position_ids"].unsqueeze(0).float().to(device)
        prompt_embeds = text.hidden_states.to(device)
        cu_seqlens = packed["cu_seqlens"].to(device)
        block_token_tags = token_tags.to(device)

        for step in range(len(video_sigmas) - 1):
            t_video = float(1.0 - video_sigmas[step])
            t_audio = float(1.0 - audio_sigmas[step])
            row_timesteps.fill_(t_video)
            row_timesteps[condition_img_pos] = max(t_video, MINIMAX_H3_IMGVID_COND_TIMESTEP)
            row_timesteps[target_audio_pos] = t_audio
            row_timesteps[condition_audio_pos] = max(t_audio, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP)
            unique_timesteps, inverse_indices = torch.unique(
                row_timesteps,
                sorted=True,
                return_inverse=True,
            )
            x[0].index_copy_(0, img_pos, video_rows)
            audio_x[0].index_copy_(0, audio_pos, audio_rows)
            video_timestep = torch.tensor(t_video, device=device)
            audio_timestep = torch.tensor(t_audio, device=device)
            video_velocity, audio_velocity = self.transformer(
                x=x,
                audio_x=audio_x,
                img_position_ids=img_position_ids,
                unique_timesteps=unique_timesteps.to(device),
                inverse_indices=inverse_indices.to(device),
                update_mask=video_update,
                update_audio_mask=audio_update,
                prompt_embeds=prompt_embeds,
                img_pos_info={"position_ids": img_pos},
                audio_pos_info={"position_ids": audio_pos},
                text_pos_info={"position_ids": text_pos},
                img_pos_for_infer_output_info={"position_ids": target_img_pos},
                packed_seq_params={"cu_seqlens_q": cu_seqlens},
                block_token_tags=block_token_tags,
                skip_mask_out_condition=True,
            )
            stepped = self.scheduler.step_denoising(
                input_visual_latent=video_rows[video_update],
                input_audio_latent=audio_rows[audio_update],
                timestep=video_timestep,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,
                noise_pred_visual=video_velocity,
                noise_pred_audio=audio_velocity[target_audio_row_start:],
                sigma_curr=video_sigmas[step],
                sigma_next=video_sigmas[step + 1],
                video_sigma_curr=video_sigmas[step],
                video_sigma_next=video_sigmas[step + 1],
                audio_sigma_curr=audio_sigmas[step],
                audio_sigma_next=audio_sigmas[step + 1],
            )
            video_rows[video_update] = stepped["output_visual_latent"]
            audio_rows[audio_update] = stepped["output_audio_latent"]

        video_latent = minimax_h3_unpatchify_video_tokens(
            video_rows[video_update].cpu(),
            latent_shape=(latent_t, latent_h // 2, latent_w // 2, 24),
            patch_size=(1, 2, 2),
        )
        audio_latent = minimax_h3_unpack_audio_tokens(
            audio_rows[audio_update].cpu(),
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        get_communication_seconds = getattr(self.transformer, "communication_seconds", None)
        communication_seconds = float(get_communication_seconds()) if callable(get_communication_seconds) else 0.0
        runtime_metrics: dict[str, float | int] = {
            "denoising_seconds": time.perf_counter() - denoising_started,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "communication_seconds": communication_seconds,
        }
        return MiniMaxH3DenoiseResult(video_latent, audio_latent, packed, runtime_metrics)


__all__ = ["MiniMaxH3DenoiseResult", "MiniMaxH3DenoisingStage"]
