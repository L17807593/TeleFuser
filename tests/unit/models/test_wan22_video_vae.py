from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from telefuser.models import wan22_video_vae


class _RecordingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_chunk_flags: list[bool] = []

    def forward(self, x, feat_cache, feat_idx, first_chunk: bool = False):
        self.first_chunk_flags.append(first_chunk)
        return x


def test_cached_decode_marks_only_the_first_frame_of_the_first_clip(monkeypatch) -> None:
    decoder = _RecordingDecoder()
    fake_vae = SimpleNamespace(
        model=SimpleNamespace(conv2=lambda value: value, decoder=decoder),
        z_dim=1,
        _feat_cache=[],
        _feat_idx=[0],
        _get_scale_on_device=lambda _device, _dtype: [torch.zeros(1), torch.ones(1)],
    )
    monkeypatch.setattr(wan22_video_vae, "_count_conv3d", lambda _decoder: 1)
    monkeypatch.setattr(wan22_video_vae, "unpatchify", lambda video, patch_size: video)

    first = torch.ones(1, 1, 2, 1, 1)
    second = torch.ones(1, 1, 1, 1, 1)
    wan22_video_vae.Wan22VideoVAE.cached_decode_withflag(
        fake_vae,
        first,
        device=torch.device("cpu"),
        is_first_clip=True,
        is_last_clip=False,
    )
    wan22_video_vae.Wan22VideoVAE.cached_decode_withflag(
        fake_vae,
        second,
        device=torch.device("cpu"),
        is_first_clip=False,
        is_last_clip=False,
    )

    assert decoder.first_chunk_flags == [True, False, False]
