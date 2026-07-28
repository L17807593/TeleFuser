"""Run the LingBot-VLA v2 base checkpoint with a RobotWin observation adapter."""

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
    ROBOTWIN_CAMERA_KEYS,
    LingBotVlaV2Observation,
    LingBotVlaV2Pipeline,
    LingBotVlaV2PipelineConfig,
)


def get_pipeline(
    model_root: str,
    qwen3vl_root: str,
    device: str = "cuda",
) -> LingBotVlaV2Pipeline:
    """Load the official 6B checkpoint and Qwen3-VL processor."""
    target_device = torch.device(device)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(qwen3vl_root, local_files_only=True, padding_side="right")
    manager = ModuleManager(torch_dtype=dtype, device="cpu")
    manager.add_module(processor, "lingbot_vla_v2_processor", path=qwen3vl_root)
    load_lingbot_vla_v2(manager, model_root, qwen3vl_root, torch_dtype=dtype)
    pipeline = LingBotVlaV2Pipeline(device=device, torch_dtype=dtype)
    pipeline.init(
        manager,
        LingBotVlaV2PipelineConfig(
            policy_config=ModelRuntimeConfig(
                device_type=target_device.type,
                device_id=target_device.index or 0,
                torch_dtype=dtype,
            ),
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
@click.option("--output", default="canonical_action_chunk.npz", type=click.Path(dir_okay=False))
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
    seed: int | None,
    device: str,
) -> None:
    """Predict and save a normalized canonical action chunk."""
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
    )
    try:
        chunk = pipeline(observation, seed=seed)
        arrays = {
            "canonical_normalized_actions": chunk.canonical_normalized_actions.numpy(),
            "horizon": np.asarray(chunk.horizon),
            "action_dim": np.asarray(chunk.action_dim),
            "checkpoint_variant": np.asarray(chunk.checkpoint_variant),
            "policy_verified": np.asarray(chunk.policy_verified),
            "verification_status": np.asarray(chunk.verification_status),
        }
        np.savez(output, **arrays)
        click.echo(
            f"Saved {chunk.horizon}-step normalized canonical action chunk to {output}; "
            f"policy status: {chunk.verification_status}"
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
