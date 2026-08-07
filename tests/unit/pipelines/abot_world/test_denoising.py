from __future__ import annotations

from types import MethodType

import torch

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.abot_world_dit import ABotWorldDiT
from telefuser.pipelines.abot_world.denoising import ABotWorldDenoisingStage


def _stage_with_recording_dit() -> tuple[ABotWorldDenoisingStage, list[torch.Tensor]]:
    dit = ABotWorldDiT(
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        downscale_factor_control_adapter=2,
    )
    manager = ModuleManager(torch_dtype=torch.float32, device="cpu")
    manager.add_module(dit, "abot_world_dit")
    stage = ABotWorldDenoisingStage(
        "abot-world-test",
        manager,
        ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
    )
    stage.parallel_models()
    observed_timesteps: list[torch.Tensor] = []

    def zero_flow_prediction(model: ABotWorldDiT, **kwargs: object) -> torch.Tensor:
        del model
        observed_timesteps.append(kwargs["timestep"].detach().clone())
        return torch.zeros_like(kwargs["x"])

    dit.forward = MethodType(zero_flow_prediction, dit)
    return stage, observed_timesteps


def test_official_four_step_schedule_matches_warped_wan_training_indices() -> None:
    scheduler = ABotWorldDenoisingStage._scheduler()

    actual = ABotWorldDenoisingStage._official_denoising_timesteps(scheduler)

    torch.testing.assert_close(actual, torch.tensor([1000.0, 937.5, 833.3333, 625.0]), rtol=1e-4, atol=1e-4)


def test_x0_prediction_uses_the_scheduler_sigma_for_each_frame() -> None:
    scheduler = ABotWorldDenoisingStage._scheduler()
    timestep = ABotWorldDenoisingStage._official_denoising_timesteps(scheduler)[1].reshape(1, 1)
    latent = torch.full((1, 1, 1, 1, 1), 4.0)
    flow_prediction = torch.full_like(latent, 2.0)

    actual = ABotWorldDenoisingStage._x0_prediction(flow_prediction, latent, timestep, scheduler)

    torch.testing.assert_close(actual, torch.full_like(latent, 2.125))


def test_denoising_block_runs_four_model_updates_then_issues_context_cache_update() -> None:
    stage, observed_timesteps = _stage_with_recording_dit()
    self_cache, cross_cache = stage._new_cache(batch_size=1, height=8, width=8)
    scheduler = stage._scheduler()
    generator = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn(1, 4, 3, 8, 8, generator=generator)

    output = stage._denoise_block(
        latent=noise,
        prompt_emb=torch.randn(1, 4, 16),
        action_context=torch.randn(1, 32, 3, 16, 16),
        first_frame_latent=None,
        self_cache=self_cache,
        cross_cache=cross_cache,
        current_start=3,
        generator=generator,
        scheduler=scheduler,
    )

    expected = ABotWorldDenoisingStage._official_denoising_timesteps(scheduler)
    assert output.shape == noise.shape
    assert len(observed_timesteps) == 5
    for observed, timestep in zip(observed_timesteps[:4], expected, strict=True):
        torch.testing.assert_close(observed, torch.full((1, 3), timestep))
    assert torch.equal(observed_timesteps[-1], torch.zeros(1, 3))
