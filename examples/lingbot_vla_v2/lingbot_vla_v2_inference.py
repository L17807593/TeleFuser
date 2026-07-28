"""Run LingBot-VLA v2 with a RobotWin observation."""

from __future__ import annotations

import json

import click
import numpy as np
import torch
from transformers import AutoProcessor

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.lingbot_vla_v2_loader import load_lingbot_vla_v2
from telefuser.pipelines.lingbot_vla_v2 import (
    LingBotVlaV2Observation,
    LingBotVlaV2Pipeline,
    LingBotVlaV2PipelineConfig,
    ROBOTWIN_CAMERA_KEYS,
)


def get_pipeline(
    model_root: str,
    qwen3vl_root: str,
    device: str = "cuda",
    include_canonical_actions: bool = False,
) -> LingBotVlaV2Pipeline:
    """Load the official 6B checkpoint and Qwen3-VL processor."""
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(qwen3vl_root, local_files_only=True, padding_side="right")
    manager = ModuleManager(torch_dtype=dtype, device="cpu")
    manager.add_module(processor, "lingbot_vla_v2_processor", path=qwen3vl_root)
    load_lingbot_vla_v2(manager, model_root, qwen3vl_root, torch_dtype=dtype)
    pipeline = LingBotVlaV2Pipeline(device=device, torch_dtype=dtype)
    pipeline.init(
        manager,
        LingBotVlaV2PipelineConfig(
            policy_config=ModelRuntimeConfig(device_type=torch.device(device).type, torch_dtype=dtype),
            include_canonical_actions=include_canonical_actions,
        ),
    )
    return pipeline


@click.command()
@click.option("--model-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--qwen3vl-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--camera-high", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--camera-left-wrist", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--camera-right-wrist", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--task", required=True)
@click.option("--state-json", required=True, help="Raw 14-D RobotWin state as a JSON list")
@click.option("--output", default="action_chunk.npz", type=click.Path(dir_okay=False))
@click.option("--include-canonical-actions", is_flag=True)
@click.option("--seed", default=None, type=int)
@click.option("--device", default="cuda")
def main(
    model_root: str,
    qwen3vl_root: str,
    camera_high: str,
    camera_left_wrist: str,
    camera_right_wrist: str,
    task: str,
    state_json: str,
    output: str,
    include_canonical_actions: bool,
    seed: int | None,
    device: str,
) -> None:
    """Predict and save a structured RobotWin action chunk."""
    try:
        state = json.loads(state_json)
    except json.JSONDecodeError as error:
        raise click.BadParameter("state-json must be valid JSON") from error
    if not isinstance(state, list) or len(state) != 14:
        raise click.BadParameter("state-json must decode to a 14-element JSON list")
    observation = LingBotVlaV2Observation(
        task=task,
        state=state,
        images=dict(
            zip(
                ROBOTWIN_CAMERA_KEYS,
                (camera_high, camera_left_wrist, camera_right_wrist),
                strict=True,
            )
        ),
    )
    pipeline = get_pipeline(
        model_root,
        qwen3vl_root,
        device=device,
        include_canonical_actions=include_canonical_actions,
    )
    try:
        chunk = pipeline(observation, seed=seed)
        arrays = {
            "action": chunk.raw_actions.numpy(),
            "action_arm_position": chunk.fields["action.arm.position"].numpy(),
            "action_effector_position": chunk.fields["action.effector.position"].numpy(),
            "action_mask": chunk.action_mask.numpy(),
            "horizon": np.asarray(chunk.horizon),
            "robot_profile": np.asarray(chunk.robot_profile),
            "policy_verified": np.asarray(chunk.policy_verified),
            "verification_status": np.asarray(chunk.verification_status),
        }
        if chunk.canonical_normalized_actions is not None:
            arrays["canonical_normalized_actions"] = chunk.canonical_normalized_actions.numpy()
        np.savez(output, **arrays)
        click.echo(
            f"Saved {chunk.horizon}-step RobotWin action chunk to {output}; "
            f"policy status: {chunk.verification_status}"
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
