"""Runtime construction for single-replica LingBot-VLA v2 inference."""

from __future__ import annotations

import torch
from transformers import AutoProcessor

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.lingbot_vla_v2_loader import load_lingbot_vla_v2

from .pipeline import LingBotVlaV2Pipeline, LingBotVlaV2PipelineConfig


def get_lingbot_vla_v2_pipeline(
    model_root: str,
    qwen3vl_root: str,
    device: str = "cuda:0",
    *,
    warmup: bool = False,
) -> LingBotVlaV2Pipeline:
    """Load one official 6B base checkpoint replica for inference."""
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable")
        device_index = target_device.index or 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; visible device count is {torch.cuda.device_count()}"
            )
        target_device = torch.device("cuda", device_index)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(qwen3vl_root, local_files_only=True, padding_side="right")
    manager = ModuleManager(torch_dtype=dtype, device="cpu")
    manager.add_module(processor, "lingbot_vla_v2_processor", path=qwen3vl_root)
    load_lingbot_vla_v2(manager, model_root, qwen3vl_root, torch_dtype=dtype)
    pipeline = LingBotVlaV2Pipeline(device=str(target_device), torch_dtype=dtype)
    pipeline.init(
        manager,
        LingBotVlaV2PipelineConfig(
            policy_config=ModelRuntimeConfig(
                device_type=target_device.type,
                device_id=target_device.index or 0,
                torch_dtype=dtype,
            ),
        ),
    )
    pipeline.prepare_for_inference()
    if warmup:
        pipeline.warmup()
    return pipeline
