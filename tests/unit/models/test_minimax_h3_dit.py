from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.models.minimax_h3_dit import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiT,
    MiniMaxH3DiTConfig,
    _reorder_grouped_qkv_to_qkv,
)


def _small_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )


def test_released_architecture_has_exact_parameter_contract() -> None:
    config = MiniMaxH3DiTConfig()
    with torch.device("meta"):
        model = MiniMaxH3DiT(config)
    assert len(model.state_dict()) == 535
    assert model.video_patch_proj.weight.shape == (5376, 96)
    assert model.blocks[49].attn.qkv_proj.weight.shape == (3 * 56 * 128, 5376)
    assert model.token_refiner.blocks[1].mlp.fc1.weight.shape == (2 * 14336, 5376)


def test_mixed_precision_boundaries_match_upstream_contract() -> None:
    with torch.device("meta"):
        model = MiniMaxH3DiT(_small_config())
    state = model.state_dict()
    for name, tensor in state.items():
        if name in MINIMAX_H3_FP32_PARAM_NAMES | MINIMAX_H3_FP32_BUFFER_NAMES:
            assert tensor.dtype == torch.float32, name
        elif tensor.is_floating_point():
            assert tensor.dtype == torch.bfloat16, name


def test_to_preserves_fp32_boundary_values_during_dtype_conversion() -> None:
    model = MiniMaxH3DiT(_small_config())
    boundary_names = MINIMAX_H3_FP32_PARAM_NAMES | MINIMAX_H3_FP32_BUFFER_NAMES
    with torch.no_grad():
        for index, name in enumerate(sorted(boundary_names), start=1):
            tensor = model.state_dict()[name]
            tensor.fill_(index * 0.123456789)
    expected = {name: model.state_dict()[name].clone() for name in boundary_names}

    model.to(dtype=torch.bfloat16)

    state = model.state_dict()
    for name, value in expected.items():
        assert state[name].dtype == torch.float32
        assert torch.equal(state[name], value), name
    assert model.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16


def test_grouped_qkv_reorder_matches_sglang_vector() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    actual = _reorder_grouped_qkv_to_qkv(
        weight,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=2,
    )
    expected = torch.tensor([0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11], dtype=torch.float32).reshape(12, 1)
    torch.testing.assert_close(actual, expected)


def test_small_packed_forward_returns_video_and_audio_rows() -> None:
    torch.manual_seed(0)
    model = MiniMaxH3DiT(_small_config()).eval()
    sequence = 8
    video_positions = torch.arange(4, 8)
    audio_positions = torch.arange(2, 4)
    text_positions = torch.arange(0, 2)
    video, audio = model(
        x=torch.randn(1, sequence, 8),
        audio_x=torch.randn(1, sequence, 2),
        img_position_ids=torch.zeros(1, sequence, 3, dtype=torch.float64),
        unique_timesteps=torch.tensor([0.5]),
        inverse_indices=torch.zeros(sequence, dtype=torch.long),
        update_mask=torch.ones(video_positions.numel(), dtype=torch.bool),
        token_tags=torch.tensor([1, 1, 2, 2, 0, 0, 0, 0]),
        prompt_embeds=torch.randn(2, 16),
        img_pos_info={"position_ids": video_positions},
        audio_pos_info={"position_ids": audio_positions},
        text_pos_info={"position_ids": text_positions},
        img_pos_for_infer_output_info={"position_ids": video_positions},
        packed_seq_params={"cu_seqlens_q": torch.tensor([0, sequence], dtype=torch.int32)},
    )
    assert video.shape == (4, 8)
    assert audio.shape == (2, 2)
    assert torch.isfinite(video).all()
    assert torch.isfinite(audio).all()


def test_enable_usp_rejects_uneven_head_partition() -> None:
    model = MiniMaxH3DiT(_small_config())
    with (
        patch("telefuser.models.minimax_h3_dit.get_ulysses_world_size", return_value=3),
        pytest.raises(ValueError, match="must be divisible"),
    ):
        model.enable_usp(MagicMock())
