from unittest.mock import patch

import torch

from telefuser.models.minimax_h3_encoder import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Encoder,
    _is_unconsumed_checkpoint_weight,
    _shard_linear_input,
    _shard_linear_output,
)


def test_encoder_filters_tail_norm_and_lm_head_but_keeps_layer_49() -> None:
    assert not _is_unconsumed_checkpoint_weight("model.language_model.layers.49.self_attn.q_proj.weight")
    assert _is_unconsumed_checkpoint_weight("model.language_model.layers.50.self_attn.q_proj.weight")
    assert _is_unconsumed_checkpoint_weight("model.language_model.layers.63.mlp.down_proj.weight")
    assert _is_unconsumed_checkpoint_weight("model.language_model.norm.weight")
    assert _is_unconsumed_checkpoint_weight("lm_head.weight")
    assert MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER == 50


def test_encoder_filter_does_not_drop_visual_or_embedding_weights() -> None:
    for name in (
        "model.language_model.embed_tokens.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.blocks.26.attn.qkv.weight",
    ):
        assert not _is_unconsumed_checkpoint_weight(name)
    assert torch.bfloat16.is_floating_point


def test_encode_ids_enters_encoder_root_forward() -> None:
    encoder = MiniMaxH3Encoder.__new__(MiniMaxH3Encoder)
    torch.nn.Module.__init__(encoder)
    encoder.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
    encoder.hidden_dim = 4
    input_ids = torch.tensor([1, 2, 3])
    expected = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)

    with patch.object(encoder, "forward", return_value=expected) as forward:
        actual = encoder.encode_ids(input_ids)

    torch.testing.assert_close(actual, expected[0])
    call = forward.call_args.kwargs
    assert torch.equal(call["input_ids"], input_ids.unsqueeze(0))
    assert torch.equal(call["attention_mask"], torch.ones(1, 3, dtype=torch.long))
    assert set(call) == {"input_ids", "attention_mask"}


def test_encoder_tp_shards_fused_columns_by_logical_section() -> None:
    linear = torch.nn.Linear(4, 12, bias=True)
    original_weight = linear.weight.detach().clone()
    original_bias = linear.bias.detach().clone()

    _shard_linear_output(linear, rank=1, world_size=2, sections=(4, 4, 4))

    expected_weight = torch.cat(tuple(section.chunk(2)[1] for section in original_weight.split((4, 4, 4))))
    expected_bias = torch.cat(tuple(section.chunk(2)[1] for section in original_bias.split((4, 4, 4))))
    torch.testing.assert_close(linear.weight, expected_weight)
    torch.testing.assert_close(linear.bias, expected_bias)
    assert linear.out_features == 6


def test_encoder_tp_row_shard_adds_bias_on_rank_zero_only() -> None:
    rank_zero = torch.nn.Linear(4, 3, bias=True)
    rank_one = torch.nn.Linear(4, 3, bias=True)
    rank_one.load_state_dict(rank_zero.state_dict())
    original_weight = rank_zero.weight.detach().clone()
    original_bias = rank_zero.bias.detach().clone()

    _shard_linear_input(rank_zero, rank=0, world_size=2)
    _shard_linear_input(rank_one, rank=1, world_size=2)

    torch.testing.assert_close(rank_zero.weight, original_weight.chunk(2, dim=1)[0])
    torch.testing.assert_close(rank_one.weight, original_weight.chunk(2, dim=1)[1])
    torch.testing.assert_close(rank_zero.bias, original_bias)
    torch.testing.assert_close(rank_one.bias, torch.zeros_like(original_bias))
    assert rank_zero.in_features == rank_one.in_features == 2
