"""BasePipeline integration for LingBot-VLA v2 base-model inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

from .data import LingBotVlaV2InputProcessor, LingBotVlaV2Inputs, LingBotVlaV2Observation
from .policy import LingBotVlaV2PolicyStage
from .robot_profile import RobotWinProfile


@dataclass
class LingBotVlaV2PipelineConfig:
    """Runtime configuration for one LingBot-VLA v2 pipeline replica."""

    policy_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    robot_profile: RobotWinProfile = field(default_factory=RobotWinProfile.default)
    image_size: int = 256
    enable_metrics: bool = False


@dataclass(frozen=True)
class LingBotVlaV2CanonicalActionChunk:
    """Normalized canonical actions produced by the base checkpoint."""

    canonical_normalized_actions: torch.Tensor
    horizon: int
    action_dim: int
    checkpoint_variant: str = "base"
    policy_verified: bool = False
    verification_status: str = "unverified_official_6b_base"


class LingBotVlaV2Pipeline(BasePipeline):
    """Single-replica LingBot-VLA v2 canonical action SDK."""

    def _get_stages(self) -> list:
        return [self.policy_stage]

    def init(self, module_manager: ModuleManager, config: LingBotVlaV2PipelineConfig) -> None:
        self._model_info = module_manager.get_model_info()
        self.config = config
        policy = module_manager.fetch_module("lingbot_vla_v2")
        processor = module_manager.fetch_module("lingbot_vla_v2_processor")
        if policy is None or processor is None:
            raise RuntimeError("LingBot-VLA v2 requires policy and lingbot_vla_v2_processor modules")
        self.input_processor = LingBotVlaV2InputProcessor(
            processor,
            policy.config,
            config.robot_profile,
            image_size=config.image_size,
        )
        self.policy_stage = LingBotVlaV2PolicyStage("policy", module_manager, config.policy_config)
        if config.enable_metrics:
            self.enable_metrics()

    @torch.inference_mode()
    def predict(
        self,
        inputs: LingBotVlaV2Inputs,
        seed: int | None = None,
    ) -> LingBotVlaV2CanonicalActionChunk:
        """Run prepared tensors and return normalized canonical actions."""
        actions = self.policy_stage.process(inputs, seed=seed)
        if actions.shape[0] != 1:
            raise RuntimeError(f"LingBot-VLA v2 pipeline expects batch size 1, got {actions.shape[0]}")
        canonical_actions = actions[0]
        policy_config = self.policy_stage.policy.config
        return LingBotVlaV2CanonicalActionChunk(
            canonical_normalized_actions=canonical_actions,
            horizon=int(canonical_actions.shape[0]),
            action_dim=int(canonical_actions.shape[1]),
            checkpoint_variant=str(getattr(policy_config, "checkpoint_variant", "base")),
            policy_verified=bool(getattr(policy_config, "policy_verified", False)),
            verification_status=str(
                getattr(policy_config, "verification_status", "unverified_official_6b_base")
            ),
        )

    @torch.inference_mode()
    def __call__(
        self,
        observation: LingBotVlaV2Observation,
        seed: int | None = None,
    ) -> LingBotVlaV2CanonicalActionChunk:
        """Predict one normalized canonical action chunk."""
        return self.predict(self.input_processor.prepare(observation), seed=seed)

    def close(self) -> None:
        """Release policy device memory."""
        if hasattr(self, "policy_stage"):
            self.policy_stage.offload_models()
