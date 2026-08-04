# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 f16/t4/d24 visual VAE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from telefuser.core.base_model import BaseModel

from .minimax_h3_video import AutoencoderKLLegacy


@dataclass(frozen=True)
class MiniMaxH3VideoVAEConfig:
    architecture: dict[str, Any]
    clip_length: int
    token_drop: int
    encoder_tiling: bool
    decoder_tiling: bool
    tile_size: int
    tile_overlap_min: int
    chunk_dim: int
    latent_channels: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> MiniMaxH3VideoVAEConfig:
        component_dir = Path(path)
        if component_dir.is_file():
            component_dir = component_dir.parent
        component = json.loads((component_dir / "config.json").read_text(encoding="utf-8"))
        source_dir = component_dir / component["source_path"]
        architecture = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
        config = cls(
            architecture=architecture,
            clip_length=int(component["vae_clip_length"]),
            token_drop=int(component["vae_token_drop"]),
            encoder_tiling=bool(component["vae_encoder_tiling"]),
            decoder_tiling=bool(component["vae_decoder_tiling"]),
            tile_size=int(component["vae_tile_size"]),
            tile_overlap_min=int(component["vae_tile_overlap_min"]),
            chunk_dim=int(component["vae_chunk_dim"]),
            latent_channels=int(component["latent_channels"]),
            latents_mean=tuple(float(value) for value in component["latents_mean"]),
            latents_std=tuple(float(value) for value in component["latents_std"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        spatial_ratio = int(self.architecture.get("vae_ratio", 0))
        temporal_ratio = int(self.architecture.get("vae_ratio_t", 0))
        if (spatial_ratio, temporal_ratio, self.latent_channels) != (16, 4, 24):
            raise ValueError(
                "MiniMax H3 visual VAE requires f16/t4/d24 geometry, got "
                f"f{spatial_ratio}/t{temporal_ratio}/d{self.latent_channels}"
            )
        if int(self.architecture.get("embed_dim", 0)) != self.latent_channels:
            raise ValueError("visual VAE embed_dim must equal latent_channels")
        if self.clip_length != 17 or self.token_drop != 3:
            raise ValueError("MiniMax H3 visual VAE requires clip_length=17 and token_drop=3")
        if len(self.latents_mean) != self.latent_channels or len(self.latents_std) != self.latent_channels:
            raise ValueError("visual VAE latent statistics must contain one value per channel")
        if any(value <= 0 for value in self.latents_std):
            raise ValueError("visual VAE latent standard deviations must be positive")

    def model_kwargs(self) -> dict[str, Any]:
        ignored = {"_class_name", "_diffusers_version", "vae_ratio", "vae_ratio_t"}
        kwargs = {key: value for key, value in self.architecture.items() if key not in ignored}
        kwargs.update(
            {
                "clip_length": self.clip_length,
                "token_drop": self.token_drop,
                "encoder_tiling": self.encoder_tiling,
                "decoder_tiling": self.decoder_tiling,
                "parallel_tiling": False,
                "tile_size": self.tile_size,
                "tile_overlap_min": self.tile_overlap_min,
                "encoder_parallel": False,
                "decoder_parallel": False,
                "chunk_dim": self.chunk_dim,
            }
        )
        return kwargs


class MiniMaxH3VideoVAE(BaseModel):
    """Checkpoint-backed 3D CNN encoder and ViT decoder."""

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__()
        config.validate()
        self.model = AutoencoderKLLegacy(**config.model_kwargs())
        self.config = config
        self.layer_name_list = ["model"]

    @property
    def processor(self) -> Any:
        return self.model.processor

    @torch.no_grad()
    def encode_images(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        return self.model.encode_images(*args, **kwargs)

    @torch.no_grad()
    def encode_videos(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        return self.model.encode_videos(*args, **kwargs)

    @torch.no_grad()
    def decode_base(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model.decode_base(*args, **kwargs)

    def prepare_decoder_autocast_weights(self, dtype: torch.dtype) -> int:
        return self.model.decoder.prepare_autocast_linear_weights(dtype)

    @torch.no_grad()
    def decode_normalized(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != self.config.latent_channels:
            raise ValueError(f"visual latent must be [B, {self.config.latent_channels}, T, H, W]")
        mean = latent.new_tensor(self.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latent.new_tensor(self.config.latents_std).view(1, -1, 1, 1, 1)
        frames = self.model.decode_base(latent.mul(std).add(mean))
        return self.model.processor.revert_tensor(frames)

    @staticmethod
    def state_dict_converter(config_path: str | Path) -> MiniMaxH3VideoVAEStateDictConverter:
        return MiniMaxH3VideoVAEStateDictConverter(config_path)


class MiniMaxH3VideoVAEStateDictConverter:
    def __init__(self, config_path: str | Path) -> None:
        self.config = MiniMaxH3VideoVAEConfig.from_path(config_path)

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        converted = {f"model.{name}": value for name, value in state_dict.items()}
        return converted, {"config": self.config}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self.from_official(state_dict)


__all__ = [
    "MiniMaxH3VideoVAE",
    "MiniMaxH3VideoVAEConfig",
    "MiniMaxH3VideoVAEStateDictConverter",
]
