# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Qwen3-VL layer-50 encoder."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen3VLConfig, Qwen3VLModel

from telefuser.core.base_model import BaseModel

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
_LAYER_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def _is_unconsumed_checkpoint_weight(name: str) -> bool:
    if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
        return True
    match = _LAYER_WEIGHT_RE.match(name)
    return bool(match and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER)


def load_minimax_h3_encoder_config(path: str | Path) -> Qwen3VLConfig:
    config_path = Path(path)
    source = config_path.parent if config_path.is_file() else config_path
    config = Qwen3VLConfig.from_pretrained(source, local_files_only=True)
    config.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    config.text_config.output_hidden_states = False
    config.text_config.use_cache = False
    return config


class MiniMaxH3Encoder(BaseModel):
    """Qwen3-VL multimodal backbone ending at unnormalized layer 50."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        if int(config.text_config.num_hidden_layers) != MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER:
            raise ValueError("MiniMax H3 encoder config must be trimmed to 50 language layers")
        if int(config.text_config.hidden_size) != MINIMAX_H3_QWEN3VL_HIDDEN_DIM:
            raise ValueError("MiniMax H3 encoder hidden size must be 5120")
        self.model = Qwen3VLModel(config)
        self.model.language_model.norm = nn.Identity()
        self.config = config
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self.selected_lm_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM
        self.layer_name_list = ["model"]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return outputs.last_hidden_state

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError(f"input_ids must be one-dimensional, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be provided together")

        host_ids = input_ids.to(device="cpu", dtype=torch.long).unsqueeze(0)
        host_image_grid = None if image_grid_thw is None else image_grid_thw.to(device="cpu", dtype=torch.long)
        host_video_grid = None if video_grid_thw is None else video_grid_thw.to(device="cpu", dtype=torch.long)
        position_ids = None
        if host_image_grid is not None or host_video_grid is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid,
                host_video_grid,
                attention_mask=torch.ones_like(host_ids),
            )
        call_kwargs: dict[str, Any] = {
            "input_ids": host_ids.to(self.device),
            "attention_mask": torch.ones_like(host_ids).to(self.device),
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
            "use_cache": False,
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            call_kwargs["image_grid_thw"] = host_image_grid
        if pixel_values_videos is not None:
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(self.device, torch.bfloat16)
            call_kwargs["video_grid_thw"] = host_video_grid
        hidden = self.model(**call_kwargs).last_hidden_state[0].to(torch.bfloat16)
        expected = (input_ids.numel(), self.hidden_dim)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"unexpected MiniMax H3 encoder shape {tuple(hidden.shape)}, expected {expected}")
        return hidden

    @staticmethod
    def state_dict_converter(config_path: str | Path) -> MiniMaxH3EncoderStateDictConverter:
        return MiniMaxH3EncoderStateDictConverter(config_path)


class MiniMaxH3EncoderStateDictConverter:
    def __init__(self, config_path: str | Path) -> None:
        self.config = load_minimax_h3_encoder_config(config_path)

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        converted = {
            name: value
            for name, value in state_dict.items()
            if "rotary_emb.inv_freq" not in name and not _is_unconsumed_checkpoint_weight(name)
        }
        return converted, {"config": self.config}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self.from_official(state_dict)


__all__ = [
    "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MiniMaxH3Encoder",
    "MiniMaxH3EncoderStateDictConverter",
    "load_minimax_h3_encoder_config",
]
