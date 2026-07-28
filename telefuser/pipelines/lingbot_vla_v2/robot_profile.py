"""RobotWin feature mapping for LingBot-VLA v2 inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch


ROBOTWIN_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ROBOTWIN_STATE_DIM = 14
CANONICAL_DIM = 55
ARM_SLICE = slice(0, 12)
EFFECTOR_SLICE = slice(28, 30)


@dataclass(frozen=True)
class LingBotVlaV2ActionChunk:
    """Structured RobotWin action chunk returned by the SDK."""

    fields: Mapping[str, torch.Tensor]
    raw_actions: torch.Tensor
    action_mask: torch.Tensor
    horizon: int
    robot_profile: str = "robotwin"
    policy_verified: bool = False
    verification_status: str = "unverified_official_6b_base"
    canonical_normalized_actions: torch.Tensor | None = None


class RobotWinProfile:
    """Map RobotWin observations and actions to LingBot's canonical space."""

    name = "robotwin"
    camera_keys = ROBOTWIN_CAMERA_KEYS
    canonical_dim = CANONICAL_DIM
    raw_state_dim = ROBOTWIN_STATE_DIM
    _REQUIRED_STATS = (
        "observation.state.arm.position",
        "observation.state.effector.position",
        "action.arm.position",
        "action.effector.position",
    )

    def __init__(self, norm_stats: Mapping[str, Mapping[str, object]]) -> None:
        self._stats = {
            key: {
                stat_name: torch.as_tensor(stat_value, dtype=torch.float32)
                for stat_name, stat_value in values.items()
            }
            for key, values in norm_stats.items()
        }
        self._validate_stats()

    @classmethod
    def from_json(cls, path: str | Path) -> "RobotWinProfile":
        """Load RobotWin normalization statistics from an upstream-format JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        norm_stats = payload.get("norm_stats")
        if not isinstance(norm_stats, dict):
            raise ValueError("RobotWin normalization file must contain a norm_stats object")
        return cls(norm_stats)

    @classmethod
    def default(cls) -> "RobotWinProfile":
        """Load the RobotWin statistics bundled with TeleFuser."""
        path = Path(__file__).with_name("assets") / "robotwin_norm_stats.json"
        return cls.from_json(path)

    @property
    def action_mask(self) -> torch.Tensor:
        """Return the canonical dimensions used by RobotWin actions."""
        mask = torch.zeros(self.canonical_dim, dtype=torch.bool)
        mask[ARM_SLICE] = True
        mask[EFFECTOR_SLICE] = True
        return mask

    def normalize_state(self, raw_state: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Convert one raw 14-D RobotWin state to normalized canonical 55-D space."""
        state = torch.as_tensor(raw_state, dtype=torch.float32, device="cpu")
        if state.shape != (self.raw_state_dim,):
            raise ValueError(f"RobotWin state must have shape ({self.raw_state_dim},), got {tuple(state.shape)}")
        if not torch.isfinite(state).all():
            raise ValueError("RobotWin state must contain only finite values")

        arm = torch.cat((state[0:6], state[7:13]))
        effector = state[[6, 13]]
        canonical = torch.zeros(self.canonical_dim, dtype=torch.float32)
        canonical[ARM_SLICE] = self._normalize("observation.state.arm.position", arm)
        canonical[EFFECTOR_SLICE] = self._normalize("observation.state.effector.position", effector)
        return canonical

    def structure_actions(
        self,
        canonical_normalized_actions: torch.Tensor,
        *,
        include_canonical: bool = False,
    ) -> LingBotVlaV2ActionChunk:
        """Convert a normalized canonical action chunk to RobotWin action fields."""
        actions = torch.as_tensor(canonical_normalized_actions, dtype=torch.float32, device="cpu")
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError("RobotWin structured output currently supports a single observation")
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] != self.canonical_dim:
            raise ValueError(
                f"canonical actions must have shape [H,{self.canonical_dim}] or [1,H,{self.canonical_dim}], "
                f"got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("canonical actions must contain only finite values")

        arm = self._unnormalize("action.arm.position", actions[:, ARM_SLICE])
        effector = self._unnormalize("action.effector.position", actions[:, EFFECTOR_SLICE])
        raw = torch.empty(actions.shape[0], self.raw_state_dim, dtype=torch.float32)
        raw[:, 0:6] = arm[:, 0:6]
        raw[:, 6] = effector[:, 0]
        raw[:, 7:13] = arm[:, 6:12]
        raw[:, 13] = effector[:, 1]
        fields = MappingProxyType(
            {
                "action.arm.position": arm,
                "action.effector.position": effector,
                "action": raw,
            }
        )
        return LingBotVlaV2ActionChunk(
            fields=fields,
            raw_actions=raw,
            action_mask=self.action_mask,
            horizon=int(actions.shape[0]),
            canonical_normalized_actions=actions.clone() if include_canonical else None,
        )

    def _validate_stats(self) -> None:
        expected_dims = {
            "observation.state.arm.position": 12,
            "observation.state.effector.position": 2,
            "action.arm.position": 12,
            "action.effector.position": 2,
        }
        missing = [key for key in self._REQUIRED_STATS if key not in self._stats]
        if missing:
            raise ValueError(f"RobotWin normalization statistics are missing keys: {missing}")
        for key, expected_dim in expected_dims.items():
            values = self._stats[key]
            for stat_name in ("q01", "q99"):
                if stat_name not in values or values[stat_name].shape != (expected_dim,):
                    shape = None if stat_name not in values else tuple(values[stat_name].shape)
                    raise ValueError(
                        f"RobotWin statistic {key}.{stat_name} must have shape ({expected_dim},), got {shape}"
                    )

    def _normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        low = self._stats[key]["q01"]
        high = self._stats[key]["q99"]
        return (value - low) / (high - low + 1e-6) * 2.0 - 1.0

    def _unnormalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        low = self._stats[key]["q01"]
        high = self._stats[key]["q99"]
        return (value + 1.0) / 2.0 * (high - low + 1e-6) + low
