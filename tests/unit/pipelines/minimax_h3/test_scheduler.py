"""Scheduler vectors derived from SGLang test_minimax_h3_denoise_loop.py."""

import pytest
import torch

from telefuser.pipelines.minimax_h3.denoising import _minimax_h3_update_target_rows_
from telefuser.pipelines.minimax_h3.scheduler import (
    MiniMaxH3EulerAncestralEta0SchedulerAdapter,
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)


def test_inplace_target_update_matches_public_scheduler_math() -> None:
    torch.manual_seed(0)
    state = torch.randn(7, 5)
    velocity = torch.randn_like(state)
    sigma_curr = 0.75
    sigma_next = 0.25
    ratio = torch.tensor(sigma_next / sigma_curr)
    denoised = state + sigma_curr * velocity
    expected = ratio * state + (1.0 - ratio) * denoised

    actual = state.clone()
    reusable_velocity = velocity.clone()
    _minimax_h3_update_target_rows_(
        actual,
        reusable_velocity,
        sigma_t=torch.tensor(sigma_curr),
        sigma_curr=sigma_curr,
        sigma_ratio=ratio,
        one_minus_sigma_ratio=1.0 - ratio,
        denoised_scratch=torch.empty_like(actual),
    )

    torch.testing.assert_close(actual, expected)


def test_rf_velocity_conversion_and_eta0_step() -> None:
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    velocity = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    timestep = torch.tensor(0.25)
    denoised = minimax_h3_rf_v_to_x0(state, velocity, timestep)
    torch.testing.assert_close(denoised, state + 0.75 * velocity)
    actual = minimax_h3_euler_eta0_step(state, denoised, sigma_curr=0.75, sigma_next=0.25)
    expected = (0.25 / 0.75) * state + (1.0 - 0.25 / 0.75) * denoised
    torch.testing.assert_close(actual, expected)


def test_dual_modality_timesteps_use_independent_sigma_schedules() -> None:
    adapter = MiniMaxH3EulerAncestralEta0SchedulerAdapter()
    visual = torch.ones(2, 3)
    audio = torch.ones(4, 2) * 2
    result = adapter.step_denoising(
        input_visual_latent=visual,
        input_audio_latent=audio,
        timestep=torch.tensor(0.5),
        video_timestep=torch.tensor(0.25),
        audio_timestep=torch.tensor(0.75),
        noise_pred_visual=torch.ones_like(visual),
        noise_pred_audio=-torch.ones_like(audio),
        sigma_curr=0.5,
        sigma_next=0.0,
        video_sigma_curr=0.75,
        video_sigma_next=0.5,
        audio_sigma_curr=0.25,
        audio_sigma_next=0.125,
    )
    expected_visual_x0 = visual + 0.75 * torch.ones_like(visual)
    expected_audio_x0 = audio - 0.25 * torch.ones_like(audio)
    torch.testing.assert_close(
        result["output_visual_latent"], (0.5 / 0.75) * visual + (1 - 0.5 / 0.75) * expected_visual_x0
    )
    torch.testing.assert_close(result["output_audio_latent"], 0.5 * audio + 0.5 * expected_audio_x0)


def test_scheduler_rejects_timestep_sigma_mismatch() -> None:
    adapter = MiniMaxH3EulerAncestralEta0SchedulerAdapter()
    with pytest.raises(ValueError, match="video_sigma_curr"):
        adapter.step_denoising(
            input_visual_latent=torch.zeros(1),
            input_audio_latent=torch.zeros(1),
            timestep=torch.tensor(0.5),
            noise_pred_visual=torch.zeros(1),
            noise_pred_audio=torch.zeros(1),
            sigma_curr=0.4,
            sigma_next=0.0,
        )
