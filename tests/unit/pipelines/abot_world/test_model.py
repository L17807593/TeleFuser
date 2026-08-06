from __future__ import annotations

import pytest
import torch

from telefuser.models.abot_world_dit import (
    ABotWorldDiT,
    CausalWanSelfAttention,
    _rope_apply,
)
from telefuser.models.wan_video_dit import precompute_freqs_cis_3d


def _tiny_dit() -> ABotWorldDiT:
    return ABotWorldDiT(
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        downscale_factor_control_adapter=2,
    )


def test_official_state_dict_converter_is_native() -> None:
    state_dict = {"blocks.0.self_attn.q.weight": torch.zeros(4, 4)}

    converted, metadata = ABotWorldDiT.state_dict_converter().from_official(state_dict)

    assert converted is state_dict
    assert metadata == {}


def test_causal_window_updates_every_transformer_block() -> None:
    model = _tiny_dit()

    model.set_causal_attention_window(local_attn_size=18, sink_size=6)

    assert model.local_attn_size == 18
    assert model.sink_size == 6
    assert all(block.self_attn.local_attn_size == 18 for block in model.blocks)
    assert all(block.self_attn.sink_size == 6 for block in model.blocks)


def test_causal_window_rejects_invalid_sink_configuration() -> None:
    model = _tiny_dit()

    with pytest.raises(ValueError, match="sink_size"):
        model.set_causal_attention_window(local_attn_size=6, sink_size=6)


def test_sink_cache_retains_prefix_and_rolls_tail() -> None:
    attention = CausalWanSelfAttention(dim=4, num_heads=1, local_attn_size=3, sink_size=1)
    cache = {
        "k": torch.zeros(1, 3, 1, 4),
        "v": torch.zeros(1, 3, 1, 4),
        "global_end_index": torch.zeros(1, dtype=torch.long),
        "local_end_index": torch.zeros(1, dtype=torch.long),
    }

    for frame in range(4):
        value = torch.full((1, 1, 1, 4), float(frame))
        attention._update_cache(cache, value, value, current_start=frame, frame_tokens=1)

    # The first frame is the sink; the rolling tail contains the newest frames.
    assert torch.equal(cache["k"][0, :, 0, 0], torch.tensor([0.0, 2.0, 3.0]))
    assert int(cache["global_end_index"].item()) == 4
    assert int(cache["local_end_index"].item()) == 3


def test_rope_applies_frame_indices_at_the_supported_boundary() -> None:
    freqs = torch.cat(precompute_freqs_cis_3d(8), dim=1)
    values = torch.randn(1, 4, 2, 8)

    output = _rope_apply(values, (2, 1, 2), freqs, torch.tensor([0, 1023]))

    assert output.shape == values.shape
    assert torch.isfinite(output).all()


def test_rope_rejects_positions_outside_the_precomputed_table() -> None:
    freqs = torch.cat(precompute_freqs_cis_3d(8), dim=1)
    values = torch.randn(1, 2, 1, 8)

    with pytest.raises(ValueError, match="frame indices"):
        _rope_apply(values, (2, 1, 1), freqs, torch.tensor([0, 1024]))


def test_sink_attention_uses_bounded_positions_for_long_sessions() -> None:
    attention = CausalWanSelfAttention(dim=8, num_heads=1, local_attn_size=3, sink_size=1)
    freqs = torch.cat(precompute_freqs_cis_3d(8), dim=1)
    cache = {
        "k": torch.zeros(1, 3, 1, 8),
        "v": torch.zeros(1, 3, 1, 8),
        "global_end_index": torch.tensor([2048]),
        "local_end_index": torch.tensor([3]),
    }

    output = attention(
        torch.randn(1, 1, 8),
        (1, 1, 1),
        freqs,
        cache,
        current_start=2048,
    )

    assert output.shape == (1, 1, 8)
    assert torch.isfinite(output).all()
