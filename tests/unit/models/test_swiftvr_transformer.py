from types import SimpleNamespace

import torch

from telefuser.models.swiftvr_transformer import (
    _WindowRuntimeMetaCache,
    _make_hw_starts,
    get_1d_rotary_pos_embed,
)
from telefuser.pipelines.swiftvr.streaming_dit import _ensure_rope_cache_len, _rope_with_offset


def _rope(length: int = 4) -> SimpleNamespace:
    parts = [get_1d_rotary_pos_embed(4, length) for _ in range(3)]
    return SimpleNamespace(
        t_dim=4,
        h_dim=4,
        w_dim=4,
        freqs_cos=torch.cat([part[0] for part in parts], dim=1),
        freqs_sin=torch.cat([part[1] for part in parts], dim=1),
    )


def test_window_starts_cover_boundary_without_redundant_interior() -> None:
    h_starts, w_starts = _make_hw_starts(11, 10, 4, 4, False)

    assert h_starts.tolist() == [0, 4, 7]
    assert w_starts.tolist() == [0, 4, 6]

    shifted_h, shifted_w = _make_hw_starts(11, 10, 4, 4, True)
    assert shifted_h.tolist() == [0, 2, 6, 7]
    assert shifted_w.tolist() == [0, 2, 6]


def test_shifted_window_owner_scatter_returns_each_global_token_once() -> None:
    for shifted, prefer_front in ((False, True), (True, False)):
        meta = _WindowRuntimeMetaCache.get(
            2,
            5,
            6,
            4,
            4,
            do_shift=shifted,
            prefer_front=prefer_front,
            device=torch.device("cpu"),
        )
        gathered_global_indices = meta.lin_flat
        restored = torch.index_select(gathered_global_indices, 0, meta.owner_pos)
        assert torch.equal(restored, torch.arange(meta.THW))


def test_rope_extension_preserves_existing_values() -> None:
    rope = _rope()
    original_cos = rope.freqs_cos.clone()
    original_sin = rope.freqs_sin.clone()

    _ensure_rope_cache_len(rope, 12)

    assert rope.freqs_cos.shape == (12, 12)
    assert rope.freqs_sin.shape == (12, 12)
    torch.testing.assert_close(rope.freqs_cos[:4], original_cos, rtol=0, atol=0)
    torch.testing.assert_close(rope.freqs_sin[:4], original_sin, rtol=0, atol=0)


def test_rope_offset_uses_global_temporal_position() -> None:
    rope = _rope()
    cos, sin = _rope_with_offset(rope, 3, 2, 2, t_off=5)
    cos_grid = cos.view(3, 2, 2, 12)
    sin_grid = sin.view(3, 2, 2, 12)

    torch.testing.assert_close(cos_grid[:, 0, 0, :4], rope.freqs_cos[5:8, :4])
    torch.testing.assert_close(sin_grid[:, 0, 0, :4], rope.freqs_sin[5:8, :4])
