import torch

from telefuser.models.minimax_h3_encoder import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    _is_unconsumed_checkpoint_weight,
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
