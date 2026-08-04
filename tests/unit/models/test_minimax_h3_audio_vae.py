import json
from pathlib import Path

import pytest
import torch

from telefuser.models.minimax_h3_audio_vae import (
    MiniMaxH3AudioVAE,
    MiniMaxH3AudioVAEConfig,
    MiniMaxH3AudioVAEStateDictConverter,
)


def _config() -> MiniMaxH3AudioVAEConfig:
    return MiniMaxH3AudioVAEConfig(
        encoder_dim=8,
        encoder_rates=(2,),
        latent_dim=16,
        decoder_dim=32,
        decoder_rates=(2,),
        sample_rate=32_000,
        latent_channels=8,
        output_channels=2,
        attn_proj=True,
        decoder_type="bigvgan",
        latents_mean=(0.0,) * 8,
        latents_std=(1.0,) * 8,
    )


def test_audio_vae_config_reads_released_component(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "source_metadata_path": "metadata.json",
                "output_channel": 2,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(
        '{"metadata":{"kwargs":{"encoder_dim":64,"encoder_rates":[2,4,4,5,5],'
        '"latent_dim":2048,"decoder_dim":1024,"decoder_rates":[5,5,2,2,2,2,2],'
        '"sample_rate":32000,"vae_latent_channels":32,"attn_proj":true,'
        '"decoder_type":"bigvgan"}}}',
        encoding="utf-8",
    )
    config = MiniMaxH3AudioVAEConfig.from_path(tmp_path)
    assert config.sample_rate == 32_000
    assert config.encoder_rates == (2, 4, 4, 5, 5)
    assert config.output_channels == 2


def test_audio_vae_config_rejects_non_h3_latent_width() -> None:
    config = _config()
    with pytest.raises(ValueError, match="32 latent channels"):
        config.validate()


def test_audio_vae_converter_maps_legacy_weight_norm_keys() -> None:
    converter = MiniMaxH3AudioVAEStateDictConverter.__new__(MiniMaxH3AudioVAEStateDictConverter)
    converter.config = _config()
    state = {
        "encoder.block.0.bias": torch.zeros(1),
        "encoder.block.0.weight_g": torch.ones(1, 1, 1),
        "encoder.block.0.weight_v": torch.ones(1, 1, 3),
    }
    converted, kwargs = converter.from_official(state)
    assert set(converted) == {
        "encoder.block.0.bias",
        "encoder.block.0.parametrizations.weight.original0",
        "encoder.block.0.parametrizations.weight.original1",
    }
    assert kwargs == {"config": converter.config}


def test_decode_normalized_requires_stereo() -> None:
    model = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(model)
    model.config = _config()
    with pytest.raises(ValueError, match=r"\[2, 32, T\]"):
        model.decode_normalized(torch.zeros(1, 8, 4))


def test_decode_normalized_casts_fp32_denoise_output_to_model_dtype() -> None:
    model = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(model)
    model.config = _config()
    model.probe = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    observed: list[torch.dtype] = []

    def decode(latent: torch.Tensor) -> torch.Tensor:
        observed.append(latent.dtype)
        return latent[:, :1]

    model.decode = decode
    output = model.decode_normalized(torch.zeros(2, 8, 4, dtype=torch.float32))
    assert observed == [torch.bfloat16]
    assert output.shape == (1, 2, 4)
