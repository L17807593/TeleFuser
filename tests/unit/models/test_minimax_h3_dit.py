import json
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
from telefuser.ops.rotary import apply_qk_norm_rope_neox, apply_rotary_emb_neox


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


def test_partial_neox_rope_matches_h3_reference() -> None:
    torch.manual_seed(0)
    query = torch.randn(3, 2, 8)
    key = torch.randn(3, 2, 8)
    angles = torch.randn(3, 3)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

    actual_query, actual_key = apply_rotary_emb_neox(query, key, cache)

    def reference(value: torch.Tensor) -> torch.Tensor:
        first, second = value[..., :6].chunk(2, dim=-1)
        cosine = angles.cos().unsqueeze(1)
        sine = angles.sin().unsqueeze(1)
        rotary = torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)
        return torch.cat((rotary, value[..., 6:]), dim=-1)

    torch.testing.assert_close(actual_query, reference(query))
    torch.testing.assert_close(actual_key, reference(key))


def test_qk_norm_rope_native_fallback_matches_split_reference() -> None:
    torch.manual_seed(0)
    query = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    q_weight = torch.randn(8, dtype=torch.bfloat16)
    k_weight = torch.randn(8, dtype=torch.bfloat16)
    angles = torch.randn(3, 3)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(-1, keepdim=True)
        return (value * torch.rsqrt(variance + 1e-5)).to(weight.dtype) * weight

    expected_query, expected_key = apply_rotary_emb_neox(
        normalize(query, q_weight),
        normalize(key, k_weight),
        cache,
    )
    actual_query, actual_key = apply_qk_norm_rope_neox(
        query,
        key,
        q_weight,
        k_weight,
        cache,
        eps=1e-5,
    )

    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


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


def test_enable_tp_shards_fused_projections_by_logical_section() -> None:
    model = MiniMaxH3DiT(_small_config())
    block = model.blocks[0]
    original_qkv = block.attn.qkv_proj.weight.detach().clone()
    original_out = block.attn.out_proj.weight.detach().clone()
    original_fc1 = block.mlp.fc1.weight.detach().clone()
    original_fc2 = block.mlp.fc2.weight.detach().clone()
    original_adaln = block.adaln_proj.linear.weight.detach().clone()
    tp_group = MagicMock()

    with (
        patch("telefuser.models.minimax_h3_dit.get_tp_world_size", return_value=2),
        patch("telefuser.models.minimax_h3_dit.get_tp_rank", return_value=1),
        patch("telefuser.models.minimax_h3_dit.get_tp_group", return_value=tp_group),
    ):
        model.enable_tp(MagicMock())

    q, k, v = original_qkv.chunk(3, dim=0)
    expected_qkv = torch.cat((q.chunk(2)[1], k.chunk(2)[1], v.chunk(2)[1]), dim=0)
    gate, up = original_fc1.chunk(2, dim=0)
    expected_fc1 = torch.cat((gate.chunk(2)[1], up.chunk(2)[1]), dim=0)
    torch.testing.assert_close(block.attn.qkv_proj.weight, expected_qkv)
    torch.testing.assert_close(block.attn.out_proj.weight, original_out.chunk(2, dim=1)[1])
    torch.testing.assert_close(block.mlp.fc1.weight, expected_fc1)
    torch.testing.assert_close(block.mlp.fc2.weight, original_fc2.chunk(2, dim=1)[1])
    torch.testing.assert_close(block.adaln_proj.linear.weight, original_adaln.chunk(2, dim=0)[1])
    assert block.attn.num_heads == 2
    assert block.attn.inner_dim == 16
    assert block.mlp.intermediate_size == 32
    assert model.tp_flag is True


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


def _small_packed_inputs() -> dict[str, object]:
    sequence = 8
    video_positions = torch.arange(4, 8)
    audio_positions = torch.arange(2, 4)
    return {
        "x": torch.randn(1, sequence, 8),
        "audio_x": torch.randn(1, sequence, 2),
        "img_position_ids": torch.zeros(1, sequence, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(sequence, dtype=torch.long),
        "update_mask": torch.ones(video_positions.numel(), dtype=torch.bool),
        "token_tags": torch.tensor([1, 1, 2, 2, 0, 0, 0, 0]),
        "prompt_embeds": torch.randn(2, 16),
        "img_pos_info": {"position_ids": video_positions},
        "audio_pos_info": {"position_ids": audio_positions},
        "text_pos_info": {"position_ids": torch.arange(0, 2)},
        "img_pos_for_infer_output_info": {"position_ids": video_positions},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, sequence], dtype=torch.int32)},
    }


def test_feature_cache_skips_joint_blocks_and_keeps_both_output_modalities() -> None:
    class _ComputeThenReuseCache:
        def __init__(self) -> None:
            self.calls = 0
            self.residual = None
            self.input_was_preserved = False

        def should_compute(self, is_cond: bool) -> bool:
            assert is_cond is True
            self.calls += 1
            return self.calls == 1

        def update(self, output: torch.Tensor, ori_input: torch.Tensor, is_cond: bool) -> None:
            assert is_cond is True
            self.input_was_preserved = output.data_ptr() != ori_input.data_ptr()
            self.residual = (output - ori_input).detach()

        def approximate(self, input: torch.Tensor, is_cond: bool) -> torch.Tensor:
            assert is_cond is True
            return input + self.residual

    torch.manual_seed(0)
    model = MiniMaxH3DiT(_small_config()).eval()
    cache = _ComputeThenReuseCache()
    model._feature_cache = cache
    inputs = _small_packed_inputs()

    with patch.object(model.blocks[0], "forward", wraps=model.blocks[0].forward) as first_block:
        first_video, first_audio = model(**inputs)
        inputs["x"] = inputs["x"] + 0.1
        inputs["audio_x"] = inputs["audio_x"] - 0.1
        second_video, second_audio = model(**inputs)

    assert first_block.call_count == 1
    assert cache.calls == 2
    assert cache.input_was_preserved
    assert first_video.shape == second_video.shape == (4, 8)
    assert first_audio.shape == second_audio.shape == (2, 2)
    assert torch.isfinite(second_video).all()
    assert torch.isfinite(second_audio).all()


def test_h3_calibrator_writes_shared_parameter_format_for_single_branch(tmp_path) -> None:
    torch.manual_seed(0)
    model = MiniMaxH3DiT(_small_config()).eval()
    output_path = tmp_path / "MiniMax-H3-Base.json"
    model.set_ada_taylor_cache_calibrator(
        num_inference_steps=2,
        sigma_shift=5.0,
        model_name="MiniMax-H3-Base",
        output_path=str(output_path),
    )

    model(**_small_packed_inputs())
    model(**_small_packed_inputs())

    params = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(params["cond_mag_ratios"]) == 2
    assert params["uncond_mag_ratios"] == params["cond_mag_ratios"]


def test_tp_forward_batches_block_adaln_all_gather() -> None:
    torch.manual_seed(0)
    model = MiniMaxH3DiT(_small_config()).eval()
    tp_group = MagicMock()
    with (
        patch("telefuser.models.minimax_h3_dit.get_tp_world_size", return_value=2),
        patch("telefuser.models.minimax_h3_dit.get_tp_rank", return_value=0),
        patch("telefuser.models.minimax_h3_dit.get_tp_group", return_value=tp_group),
    ):
        model.enable_tp(MagicMock())

    sequence = 8
    video_positions = torch.arange(4, 8)
    audio_positions = torch.arange(2, 4)

    def gather_rank_copies(tensor: torch.Tensor, *, dim: int, **_: object) -> torch.Tensor:
        return torch.cat((tensor, tensor), dim=dim)

    with (
        patch("telefuser.models.minimax_h3_dit.all_gather_cat", side_effect=gather_rank_copies) as gather,
        patch("telefuser.models.minimax_h3_dit.all_reduce_sum_"),
    ):
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
            text_pos_info={"position_ids": torch.arange(0, 2)},
            img_pos_for_infer_output_info={"position_ids": video_positions},
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, sequence], dtype=torch.int32)},
        )

    assert video.shape == (4, 8)
    assert audio.shape == (2, 2)
    assert gather.call_count == 2
    assert gather.call_args_list[0].args[0].shape[0] == len(model.blocks)


def test_request_static_inputs_are_reused_across_denoising_steps() -> None:
    torch.manual_seed(0)
    model = MiniMaxH3DiT(_small_config()).eval()
    sequence = 8
    video_positions = torch.arange(4, 8)
    audio_positions = torch.arange(2, 4)
    inputs = {
        "x": torch.randn(1, sequence, 8),
        "audio_x": torch.randn(1, sequence, 2),
        "img_position_ids": torch.zeros(1, sequence, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(sequence, dtype=torch.long),
        "update_mask": torch.ones(video_positions.numel(), dtype=torch.bool),
        "token_tags": torch.tensor([1, 1, 2, 2, 0, 0, 0, 0]),
        "prompt_embeds": torch.randn(2, 16),
        "img_pos_info": {"position_ids": video_positions},
        "audio_pos_info": {"position_ids": audio_positions},
        "text_pos_info": {"position_ids": torch.arange(0, 2)},
        "img_pos_for_infer_output_info": {"position_ids": video_positions},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, sequence], dtype=torch.int32)},
    }

    with (
        patch.object(model.token_refiner, "forward", wraps=model.token_refiner.forward) as token_refiner,
        patch.object(model.rope, "forward", wraps=model.rope.forward) as rope,
    ):
        model(static_cache_key=1, **inputs)
        model(static_cache_key=1, **inputs)
        assert token_refiner.call_count == 1
        assert rope.call_count == 1

        model(static_cache_key=2, **inputs)
        assert token_refiner.call_count == 2
        assert rope.call_count == 2


def test_time_embedder_reuses_device_frequency_cache() -> None:
    model = MiniMaxH3DiT(_small_config()).eval()
    timestep = torch.tensor([0.5])

    model.time_embedder(timestep)
    cached = model.time_embedder._frequency_cache[timestep.device]
    model.time_embedder(timestep)

    assert model.time_embedder._frequency_cache[timestep.device] is cached


def test_enable_usp_rejects_uneven_head_partition() -> None:
    model = MiniMaxH3DiT(_small_config())
    with (
        patch("telefuser.models.minimax_h3_dit.get_ulysses_world_size", return_value=3),
        pytest.raises(ValueError, match="must be divisible"),
    ):
        model.enable_usp(MagicMock())
