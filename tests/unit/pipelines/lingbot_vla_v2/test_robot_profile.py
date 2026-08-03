from __future__ import annotations

import pytest
import torch

from telefuser.pipelines.lingbot_vla_v2.robot_profile import RobotWinProfile


def _stats() -> dict[str, dict[str, list[float]]]:
    return {
        "observation.state.arm.position": {"q01": [0.0] * 12, "q99": [2.0] * 12},
        "observation.state.effector.position": {"q01": [-1.0] * 2, "q99": [1.0] * 2},
        "action.arm.position": {"q01": [0.0] * 12, "q99": [2.0] * 12},
        "action.effector.position": {"q01": [-1.0] * 2, "q99": [1.0] * 2},
    }


def test_normalize_state_uses_robotwin_joint_order() -> None:
    profile = RobotWinProfile(_stats())
    state = torch.arange(14, dtype=torch.float32) / 10.0

    canonical = profile.normalize_state(state)

    arm = torch.cat((state[0:6], state[7:13]))
    effector = state[[6, 13]]
    assert canonical.shape == (55,)
    assert torch.allclose(canonical[0:12], arm / (2.0 + 1e-6) * 2.0 - 1.0)
    assert torch.allclose(canonical[28:30], (effector + 1.0) / (2.0 + 1e-6) * 2.0 - 1.0)
    assert torch.count_nonzero(canonical[12:28]) == 0
    assert torch.count_nonzero(canonical[30:]) == 0


def test_structure_actions_reconstructs_raw_robotwin_layout() -> None:
    profile = RobotWinProfile(_stats())
    canonical = torch.zeros(1, 3, 55)

    chunk = profile.structure_actions(canonical, include_canonical=True)

    arm = chunk.fields["action.arm.position"]
    effector = chunk.fields["action.effector.position"]
    assert arm.shape == (3, 12)
    assert effector.shape == (3, 2)
    assert torch.allclose(arm, torch.full_like(arm, 1.0000005))
    assert torch.allclose(effector, torch.zeros_like(effector), atol=1e-6)
    assert torch.equal(chunk.raw_actions[:, 0:6], arm[:, 0:6])
    assert torch.equal(chunk.raw_actions[:, 6], effector[:, 0])
    assert torch.equal(chunk.raw_actions[:, 7:13], arm[:, 6:12])
    assert torch.equal(chunk.raw_actions[:, 13], effector[:, 1])
    assert chunk.horizon == 3
    assert chunk.canonical_normalized_actions is not None


def test_action_chunk_is_marked_unverified() -> None:
    chunk = RobotWinProfile(_stats()).structure_actions(torch.zeros(2, 55))

    assert chunk.policy_verified is False
    assert chunk.verification_status == "unverified_official_6b_base"
    assert chunk.robot_profile == "robotwin"
    assert chunk.action_mask.shape == (55,)
    assert chunk.action_mask.nonzero().flatten().tolist() == list(range(12)) + [28, 29]
    assert chunk.canonical_normalized_actions is None


def test_default_profile_loads_bundled_upstream_stats() -> None:
    profile = RobotWinProfile.default()

    canonical = profile.normalize_state(torch.zeros(14))
    chunk = profile.structure_actions(torch.zeros(1, 55))

    assert canonical.shape == (55,)
    assert torch.isfinite(canonical).all()
    assert chunk.raw_actions.shape == (1, 14)
    assert torch.isfinite(chunk.raw_actions).all()


def test_profile_rejects_invalid_state_and_action_shapes() -> None:
    profile = RobotWinProfile(_stats())

    with pytest.raises(ValueError, match="state must have shape"):
        profile.normalize_state(torch.zeros(13))
    with pytest.raises(ValueError, match="canonical actions must have shape"):
        profile.structure_actions(torch.zeros(2, 54))
