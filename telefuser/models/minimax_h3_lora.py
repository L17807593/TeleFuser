# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Turbo LoRA mapping rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from telefuser.core.config import LoraConfig
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader, LoRATarget

MINIMAX_H3_LORA_KEY_MAPPING_RULES = [
    (r"^(?:base_model\.model\.|model\.diffusion_model\.|diffusion_model\.|transformer\.|model\.)", ""),
    (r"^transformer_blocks\.", "blocks."),
    (r"^token_refiner\.refiner_blocks\.", "token_refiner.blocks."),
    (r"\.attn\.to_out\.0\.", ".attn.out_proj."),
    (r"\.ff\.net\.0\.proj\.", ".mlp.fc1."),
    (r"\.ff\.net\.2\.", ".mlp.fc2."),
]


def minimax_h3_lora_target(
    model_key: str,
    weights: Mapping[str, torch.Tensor],
) -> LoRATarget | None:
    """Resolve Diffusers H3 projections to native parameters or fused QKV slices."""
    for source, offset in ((".attn.to_q.", 0), (".attn.to_k.", 1), (".attn.to_v.", 2)):
        if source not in model_key:
            continue
        target_key = model_key.replace(source, ".attn.qkv_proj.")
        parameter = weights.get(target_key)
        if parameter is None or parameter.shape[0] % 3:
            return None
        width = parameter.shape[0] // 3
        return LoRATarget(target_key, parameter[offset * width : (offset + 1) * width])
    parameter = weights.get(model_key)
    return None if parameter is None else LoRATarget(model_key, parameter)


class MiniMaxH3LoraAdapter:
    """Configure the generic loader for the released H3 Turbo LoRA."""

    DEFAULT_ALPHA = 128.0

    @classmethod
    def apply(cls, model: torch.nn.Module, configs: Iterable[LoraConfig]) -> int:
        if getattr(model, "quant_type", None) is not None:
            raise ValueError("MiniMax H3 Turbo LoRA requires original DiT weights during merging")
        loader = LoRALoader(
            MINIMAX_H3_LORA_KEY_MAPPING_RULES,
            target_resolver=minimax_h3_lora_target,
            strict=True,
            default_alpha=cls.DEFAULT_ALPHA,
            stream_safetensors=True,
            merge_dtype=torch.float32,
        )
        total = 0
        for config in configs:
            applied = loader.apply_lora(model, config.path, strength=config.strength)
            total += applied
            logger.info(
                "Loaded MiniMax H3 Turbo LoRA: {} (strength={}, layers={})", config.path, config.strength, applied
            )
        return total


__all__ = [
    "MINIMAX_H3_LORA_KEY_MAPPING_RULES",
    "MiniMaxH3LoraAdapter",
    "minimax_h3_lora_target",
]
