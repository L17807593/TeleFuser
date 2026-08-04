from pathlib import Path

import pytest
import torch

from telefuser.models.minimax_h3_video_vae import (
    MiniMaxH3VideoVAEConfig,
    MiniMaxH3VideoVAEStateDictConverter,
)


def _config() -> MiniMaxH3VideoVAEConfig:
    return MiniMaxH3VideoVAEConfig(
        architecture={"vae_ratio": 16, "vae_ratio_t": 4, "embed_dim": 24},
        clip_length=17,
        token_drop=3,
        encoder_tiling=True,
        decoder_tiling=True,
        tile_size=256,
        tile_overlap_min=64,
        chunk_dim=-1,
        latent_channels=24,
        latents_mean=(0.0,) * 24,
        latents_std=(1.0,) * 24,
    )


def test_video_vae_config_reads_released_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "config.json").write_text(
        '{"source_path":"source","vae_clip_length":17,"vae_token_drop":3,'
        '"vae_encoder_tiling":1,"vae_decoder_tiling":1,"vae_tile_size":256,'
        '"vae_tile_overlap_min":64,"vae_chunk_dim":-1,"latent_channels":24,'
        '"latents_mean":[' + ",".join(["0"] * 24) + "],"
        '"latents_std":[' + ",".join(["1"] * 24) + "]}",
        encoding="utf-8",
    )
    (source / "config.json").write_text(
        '{"vae_ratio":16,"vae_ratio_t":4,"embed_dim":24}',
        encoding="utf-8",
    )
    config = MiniMaxH3VideoVAEConfig.from_path(tmp_path)
    assert config.latent_channels == 24
    assert config.model_kwargs()["parallel_tiling"] is False


def test_video_vae_config_rejects_wrong_geometry() -> None:
    config = _config()
    config.architecture["vae_ratio_t"] = 8
    with pytest.raises(ValueError, match="f16/t4/d24"):
        config.validate()


def test_video_vae_converter_prefixes_composed_model() -> None:
    converter = MiniMaxH3VideoVAEStateDictConverter.__new__(MiniMaxH3VideoVAEStateDictConverter)
    converter.config = _config()
    state = {"encoder.conv_in.conv.weight": torch.zeros(1)}
    converted, kwargs = converter.from_official(state)
    assert set(converted) == {"model.encoder.conv_in.conv.weight"}
    assert kwargs == {"config": converter.config}
