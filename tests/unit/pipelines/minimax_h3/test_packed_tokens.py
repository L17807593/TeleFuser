import torch

from telefuser.pipelines.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)


def test_video_patchify_round_trip_preserves_native_layout() -> None:
    latent = torch.arange(2 * 3 * 2 * 4 * 6).reshape(2, 3, 2, 4, 6)
    rows = minimax_h3_patchify_video_latent(latent, patch_size=(1, 2, 2))
    restored = minimax_h3_unpatchify_video_tokens(rows, latent_shape=(2, 2, 3, 3), patch_size=(1, 2, 2))
    assert torch.equal(restored, latent)


def test_audio_unpack_restores_channel_major_latent() -> None:
    rows = torch.arange(12).reshape(6, 2)
    actual = minimax_h3_unpack_audio_tokens(rows, audio_t=6, audio_channel=2)
    assert actual.shape == (2, 2, 3)
    assert torch.equal(actual[0], rows[:3].T)
    assert torch.equal(actual[1], rows[3:].T)
