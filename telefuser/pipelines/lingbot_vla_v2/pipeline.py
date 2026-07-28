"""BasePipeline integration for LingBot-VLA v2 RobotWin inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

from .data import LingBotVlaV2InputProcessor, LingBotVlaV2Inputs, LingBotVlaV2Observation
from .policy import LingBotVlaV2PolicyStage
from .robot_profile import LingBotVlaV2ActionChunk, RobotWinProfile


@dataclass
class LingBotVlaV2PipelineConfig:
    """Runtime configuration for one LingBot-VLA v2 pipeline replica."""

    policy_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    robot_profile: RobotWinProfile = field(default_factory=RobotWinProfile.default)
    include_canonical_actions: bool = False
    enable_metrics: bool = False


class LingBotVlaV2Pipeline(BasePipeline):
    """Single-replica LingBot-VLA v2 structured action SDK."""

    def _get_stages(self) -> list:
        return [self.policy_stage]

    def init(self, module_manager: ModuleManager, config: LingBotVlaV2PipelineConfig) -> None:
        self._model_info = module_manager.get_model_info()
        self.config = config
        policy = module_manager.fetch_module("lingbot_vla_v2")
        processor = module_manager.fetch_module("lingbot_vla_v2_processor")
        if policy is None or processor is None:
            raise RuntimeError("LingBot-VLA v2 requires policy and lingbot_vla_v2_processor modules")
        self.input_processor = LingBotVlaV2InputProcessor(processor, policy.config, config.robot_profile)
        self.policy_stage = LingBotVlaV2PolicyStage("policy", module_manager, config.policy_config)
        if config.enable_metrics:
            self.enable_metrics()

    @torch.inference_mode()
    def predict(self, inputs: LingBotVlaV2Inputs, seed: int | None = None) -> LingBotVlaV2ActionChunk:
        """Run prepared tensors and convert the canonical result to RobotWin fields."""
        actions = self.policy_stage.process(inputs, seed=seed)
        return self.config.robot_profile.structure_actions(
            actions,
            include_canonical=self.config.include_canonical_actions,
        )

    @torch.inference_mode()
    def __call__(
        self,
        observation: LingBotVlaV2Observation,
        seed: int | None = None,
    ) -> LingBotVlaV2ActionChunk:
        """Predict one structured RobotWin action chunk."""
        return self.predict(self.input_processor.prepare(observation), seed=seed)

    def close(self) -> None:
        """Release policy device memory."""
        if hasattr(self, "policy_stage"):
            self.policy_stage.offload_models()
