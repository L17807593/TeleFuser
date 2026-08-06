from __future__ import annotations

import pytest
import torch

from telefuser.pipelines.abot_world import ABotWorldPipeline


def test_action_context_uses_official_wasd_ijkl_channel_layout() -> None:
    action = ABotWorldPipeline.build_action_context(
        {"W": True, "D": True, "L": True},
        latent_frames=4,
        height=32,
        width=64,
        device="cpu",
        dtype=torch.float32,
    )
    assert action.shape == (1, 32, 4, 32, 64)
    # Every key is expanded into four contiguous channels, in W,A,S,D,I,J,K,L order.
    assert torch.all(action[:, 0:4] == 1)
    assert torch.all(action[:, 12:16] == 1)
    assert torch.all(action[:, 28:32] == 1)
    assert torch.all(action[:, 4:12] == 0)
    assert torch.all(action[:, 16:28] == 0)


def test_action_context_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown ABot action keys"):
        ABotWorldPipeline.build_action_context(
            {"SPACE": True}, latent_frames=1, height=32, width=32, device="cpu", dtype=torch.float32
        )
