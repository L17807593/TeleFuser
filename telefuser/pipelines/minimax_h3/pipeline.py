# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 local T2VA, FL2VA, and Ref2VA pipeline."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoProcessor

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.worker import ParallelWorker

from .data import (
    minimax_h3_validate_canonical_request,
    minimax_h3_validate_reference_media_facts,
)
from .denoising import MiniMaxH3DenoisingStage
from .material_io import (
    MiniMaxH3MaterialFacts,
    minimax_h3_localize_material,
    minimax_h3_probe_material,
)
from .resolved_plan import (
    MiniMaxH3ResolvedPlan,
    minimax_h3_resolve_plan,
    minimax_h3_resolve_spatial_shape,
)
from .text_encoding import MiniMaxH3TextEncodingStage
from .vae import MiniMaxH3PreparedCondition, MiniMaxH3VAEStage


def _fp32_runtime_config() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(torch_dtype=torch.float32)


@dataclass
class MiniMaxH3PipelineConfig:
    processor_path: str
    text_encoder_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    video_vae_config: ModelRuntimeConfig = field(default_factory=_fp32_runtime_config)
    audio_vae_config: ModelRuntimeConfig = field(default_factory=_fp32_runtime_config)
    num_inference_steps: int = 50

    def __post_init__(self) -> None:
        if not self.processor_path:
            raise ValueError("processor_path is required")
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")


@dataclass(frozen=True)
class MiniMaxH3Generation:
    video: torch.Tensor
    audio: torch.Tensor
    video_fps: int
    audio_sample_rate: int
    plan: MiniMaxH3ResolvedPlan
    packed_sequence_length: int
    runtime_metrics: dict[str, float | int]


class MiniMaxH3Pipeline(BasePipeline):
    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: MiniMaxH3PipelineConfig | None = None
        self.text_stage: MiniMaxH3TextEncodingStage | None = None
        self.vae_stage: MiniMaxH3VAEStage | None = None
        self.denoising_stage: MiniMaxH3DenoisingStage | ParallelWorker | None = None

    def init(self, module_manager: ModuleManager, config: MiniMaxH3PipelineConfig) -> None:
        self.config = config
        processor = AutoProcessor.from_pretrained(
            config.processor_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.text_stage = MiniMaxH3TextEncodingStage(
            module_manager,
            config.text_encoder_config,
            processor=processor,
        )
        self.vae_stage = MiniMaxH3VAEStage(
            module_manager,
            config.video_vae_config,
            config.audio_vae_config,
        )
        denoising_stage = MiniMaxH3DenoisingStage(module_manager, config.dit_config)
        if config.dit_config.parallel_config.world_size > 1 and not dist.is_initialized():
            self.denoising_stage = ParallelWorker(denoising_stage)
        else:
            self.denoising_stage = denoising_stage
            if dist.is_initialized():
                denoising_stage.parallel_models()
        self._model_info = module_manager.get_model_info()

    def _get_stages(self) -> list[object]:
        return [stage for stage in (self.text_stage, self.vae_stage, self.denoising_stage) if stage is not None]

    def stop(self) -> None:
        for stage in self._get_stages():
            close = getattr(stage, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _resolve_stage_result(value: Any) -> Any:
        return value() if callable(value) else value

    @staticmethod
    def _resolve_deferred_plan(
        canonical: dict[str, Any],
        facts: dict[int, MiniMaxH3MaterialFacts],
    ) -> tuple[dict[str, Any], MiniMaxH3ResolvedPlan]:
        if canonical["target"].get("duration_seconds") is None:
            duration_sources = [
                (index, condition)
                for index, condition in enumerate(canonical["conditions"])
                if condition["type"] in {"audio", "video", "video_audio"}
            ]
            index, condition = duration_sources[0]
            duration = facts[index].duration_seconds
            if duration is None:
                raise ValueError(f"conditions[{index}] has no probed duration")
            effective = duration - float(condition.get("start_time_seconds", 0.0))
            canonical = minimax_h3_validate_canonical_request(
                task=canonical["task"],
                prompt=canonical["prompt"],
                conditions=canonical["conditions"],
                target={**canonical["target"], "duration_seconds": effective},
                flow_shift=canonical.get("flow_shift"),
                audio_flow_shift=canonical.get("audio_flow_shift"),
                seed=canonical.get("seed"),
            )
        plan = minimax_h3_resolve_plan(canonical)
        if plan.shape.get("geometry") == "deferred":
            first = plan.materials[0]
            item_facts = facts[int(first.condition_index)]
            if item_facts.width is None or item_facts.height is None:
                raise ValueError("deferred FL2VA geometry requires image width and height")
            shape = dict(plan.shape)
            shape.update(
                minimax_h3_resolve_spatial_shape(
                    width=item_facts.width,
                    height=item_facts.height,
                )
            )
            plan = replace(plan, shape=shape)
        if plan.shape.get("geometry") != "resolved_v2":
            raise ValueError("MiniMax H3 target geometry must resolve before model execution")
        return canonical, plan

    @staticmethod
    def _condition_labels(
        prepared: list[MiniMaxH3PreparedCondition],
    ) -> list[tuple[str, int]]:
        counters = {"image": 0, "audio": 0, "video": 0}
        labels: list[tuple[str, int]] = []
        for condition in prepared:
            if condition.kind == "image":
                counters["image"] += 1
                labels.append(("image", counters["image"]))
            elif condition.kind == "audio":
                counters["audio"] += 1
                labels.append(("audio", counters["audio"]))
            elif condition.kind in {"video", "video_audio"}:
                if condition.has_audio:
                    counters["audio"] += 1
                    labels.append(("audio", counters["audio"]))
                counters["video"] += 1
                labels.append(("video", counters["video"]))
        return labels

    @torch.inference_mode()
    def __call__(
        self,
        *,
        task: str,
        prompt: str,
        conditions: list[dict[str, Any]] | None,
        target: dict[str, Any],
        seed: int | None = None,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
        num_inference_steps: int | None = None,
    ) -> MiniMaxH3Generation:
        if self.config is None or self.text_stage is None or self.vae_stage is None or self.denoising_stage is None:
            raise RuntimeError("MiniMaxH3Pipeline.init must be called before generation")
        canonical = minimax_h3_validate_canonical_request(
            task=task,
            prompt=prompt,
            conditions=[] if conditions is None else conditions,
            target=target,
            seed=seed,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
        )
        with ExitStack() as stack:
            paths: dict[int, Path] = {}
            facts: dict[int, MiniMaxH3MaterialFacts] = {}
            for index, condition in enumerate(canonical["conditions"]):
                path = stack.enter_context(minimax_h3_localize_material(condition["uri"]))
                paths[index] = path
                facts[index] = minimax_h3_probe_material(path, condition["type"])
                duration = facts[index].duration_seconds
                start = float(condition.get("start_time_seconds", 0.0))
                if duration is not None and start >= duration:
                    raise ValueError(f"conditions[{index}].start_time_seconds must be less than media duration")
            if task == "ref2va":
                duration_facts = {
                    index: float(item.duration_seconds)
                    for index, item in facts.items()
                    if item.duration_seconds is not None
                }
                minimax_h3_validate_reference_media_facts(canonical["conditions"], duration_facts)
            canonical, plan = self._resolve_deferred_plan(canonical, facts)
            prepared = self.vae_stage.prepare_media(plan, paths, facts)
            images = [item.image for item in prepared if item.image is not None]
            videos = [item.video_frames for item in prepared if item.video_frames is not None]
            text = self.text_stage.encode(
                task=plan.task,
                prompt=plan.prompt,
                images=images,
                videos=videos,
                condition_labels=self._condition_labels(prepared),
            )
            if any(item.image is not None or item.video_frames is not None for item in prepared):
                prepared = self.vae_stage.encode_visual(prepared)
            duration_seconds = float(plan.shape["frame_count"]) / float(plan.shape["fps"])
            if any(item.has_audio for item in prepared):
                prepared = self.vae_stage.encode_audio(prepared, paths, facts, duration_seconds)
            steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
                raise ValueError("num_inference_steps must be an integer of at least 2")
            denoised = self._resolve_stage_result(
                self.denoising_stage.denoise(
                    plan=plan,
                    text=text,
                    conditions=prepared,
                    num_inference_steps=steps,
                )
            )
            video = self.vae_stage.decode_video(denoised.video_latent)
            audio = self.vae_stage.decode_audio(denoised.audio_latent)
            return MiniMaxH3Generation(
                video=video[:, : int(plan.shape["frame_count"])],
                audio=audio,
                video_fps=24,
                audio_sample_rate=32_000,
                plan=plan,
                packed_sequence_length=int(denoised.packed["seq_len"]),
                runtime_metrics=denoised.runtime_metrics,
            )


__all__ = [
    "MiniMaxH3Generation",
    "MiniMaxH3Pipeline",
    "MiniMaxH3PipelineConfig",
]
