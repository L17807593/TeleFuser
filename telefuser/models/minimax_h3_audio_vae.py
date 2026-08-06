# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 32 kHz stereo audio VAE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from telefuser.core.base_model import BaseModel

from .minimax_h3_audio import DacAudioVAE


@dataclass(frozen=True)
class MiniMaxH3AudioVAEConfig:
    encoder_dim: int
    encoder_rates: tuple[int, ...]
    latent_dim: int
    decoder_dim: int
    decoder_rates: tuple[int, ...]
    sample_rate: int
    latent_channels: int
    output_channels: int
    attn_proj: bool
    decoder_type: str
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> MiniMaxH3AudioVAEConfig:
        component_dir = Path(path)
        if component_dir.is_file():
            component_dir = component_dir.parent
        component = json.loads((component_dir / "config.json").read_text(encoding="utf-8"))
        metadata_path = component_dir / component["source_metadata_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["metadata"]["kwargs"]
        config = cls(
            encoder_dim=int(metadata["encoder_dim"]),
            encoder_rates=tuple(int(rate) for rate in metadata["encoder_rates"]),
            latent_dim=int(metadata["latent_dim"]),
            decoder_dim=int(metadata["decoder_dim"]),
            decoder_rates=tuple(int(rate) for rate in metadata["decoder_rates"]),
            sample_rate=int(metadata["sample_rate"]),
            latent_channels=int(metadata["vae_latent_channels"]),
            output_channels=int(component["output_channel"]),
            attn_proj=bool(metadata["attn_proj"]),
            decoder_type=str(metadata["decoder_type"]),
            latents_mean=tuple(float(value) for value in component["latents_mean"]),
            latents_std=tuple(float(value) for value in component["latents_std"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.sample_rate != 32_000:
            raise ValueError(f"MiniMax H3 audio VAE requires 32000 Hz, got {self.sample_rate}")
        if self.latent_channels != 32:
            raise ValueError(f"MiniMax H3 audio VAE requires 32 latent channels, got {self.latent_channels}")
        if self.output_channels != 2:
            raise ValueError(f"MiniMax H3 output requires stereo audio, got {self.output_channels} channels")
        if len(self.latents_mean) != self.latent_channels or len(self.latents_std) != self.latent_channels:
            raise ValueError("audio VAE latent statistics must contain one value per latent channel")
        if any(value <= 0 for value in self.latents_std):
            raise ValueError("audio VAE latent standard deviations must be positive")


class MiniMaxH3AudioVAE(DacAudioVAE, BaseModel):
    """DAC-lineage waveform encoder and BigVGAN decoder used by MiniMax H3."""

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        config.validate()
        super().__init__(
            encoder_dim=config.encoder_dim,
            encoder_rates=list(config.encoder_rates),
            latent_dim=config.latent_dim,
            decoder_dim=config.decoder_dim,
            decoder_rates=list(config.decoder_rates),
            sample_rate=config.sample_rate,
            vae_latent_channels=config.latent_channels,
            attn_proj=config.attn_proj,
            decoder_type=config.decoder_type,
        )
        self.config = config
        self.layer_name_list = ["encoder", "decoder"]

    @torch.no_grad()
    def encode_mean(self, waveform: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Encode independent waveform channels to deterministic normalized latents."""
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must be [channels, 1, samples]")
        waveform = self.preprocess(waveform, sample_rate)
        parameter = next(self.parameters())
        waveform = waveform.to(device=parameter.device, dtype=parameter.dtype)
        hidden = self.encoder(waveform)
        if self.attn_proj:
            hidden = self.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
        latent = self.mean_proj(hidden)
        mean = latent.new_tensor(self.config.latents_mean).view(1, -1, 1)
        std = latent.new_tensor(self.config.latents_std).view(1, -1, 1)
        return latent.sub(mean).div(std)

    @torch.no_grad()
    def decode_normalized(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode normalized [2, 32, T] latents to [1, 2, samples] stereo."""
        if latent.ndim != 3 or latent.shape[0] != self.config.output_channels:
            raise ValueError("audio latent must be [2, 32, T]")
        if latent.shape[1] != self.config.latent_channels:
            raise ValueError(f"audio latent channel dimension must be {self.config.latent_channels}")
        parameter = next(self.parameters())
        latent = latent.to(device=parameter.device, dtype=parameter.dtype)
        mean = latent.new_tensor(self.config.latents_mean).view(1, -1, 1)
        std = latent.new_tensor(self.config.latents_std).view(1, -1, 1)
        waveform = self.decode(latent.mul(std).add(mean))
        return waveform.transpose(0, 1).contiguous()

    @staticmethod
    def state_dict_converter(config_path: str | Path) -> MiniMaxH3AudioVAEStateDictConverter:
        return MiniMaxH3AudioVAEStateDictConverter(config_path)


class MiniMaxH3AudioVAEStateDictConverter:
    def __init__(self, config_path: str | Path) -> None:
        self.config = MiniMaxH3AudioVAEConfig.from_path(config_path)

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        converted: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            if name.endswith(".weight_g"):
                name = name.removesuffix(".weight_g") + ".parametrizations.weight.original0"
            elif name.endswith(".weight_v"):
                name = name.removesuffix(".weight_v") + ".parametrizations.weight.original1"
            converted[name] = value
        return converted, {"config": self.config}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self.from_official(state_dict)


__all__ = [
    "MiniMaxH3AudioVAE",
    "MiniMaxH3AudioVAEConfig",
    "MiniMaxH3AudioVAEStateDictConverter",
]
