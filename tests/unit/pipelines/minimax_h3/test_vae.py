import numpy as np
import torch

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.minimax_h3.resolved_plan import MiniMaxH3MaterialPlanItem
from telefuser.pipelines.minimax_h3.vae import (
    MiniMaxH3AudioVAEStage,
    MiniMaxH3PreparedCondition,
    MiniMaxH3VideoVAEStage,
)


class _VideoVAEProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"latents_mean": (0.0,) * 24, "latents_std": (1.0,) * 24})()
        self.observed: np.ndarray | None = None
        self.prepared_dtype: torch.dtype | None = None

    def encode_videos(self, frames: np.ndarray, **_: object) -> list[torch.Tensor]:
        self.observed = frames
        return [torch.zeros(24, 2, 2, 2)]

    def prepare_decoder_autocast_weights(self, dtype: torch.dtype) -> None:
        self.prepared_dtype = dtype

    def decode_normalized(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :3]


def test_reference_video_uses_uint8_numpy_preprocessing_path() -> None:
    video_vae = _VideoVAEProbe()
    audio_vae = torch.nn.Linear(1, 1)
    manager = ModuleManager(device="cpu")
    manager.add_module(video_vae, "minimax_h3_video_vae")
    manager.add_module(audio_vae, "minimax_h3_audio_vae")
    runtime = ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32)
    stage = MiniMaxH3VideoVAEStage(manager, runtime)
    material = MiniMaxH3MaterialPlanItem(
        0,
        "reference",
        "video",
        "reference.mov",
        "video.reference_preserve",
    )
    condition = MiniMaxH3PreparedCondition(
        material,
        "video",
        video_frames=torch.zeros(5, 32, 32, 3, dtype=torch.uint8),
    )

    encoded = stage.encode_visual([condition])

    assert video_vae.observed is not None
    assert video_vae.observed.dtype == np.uint8
    assert video_vae.observed.shape == (5, 32, 32, 3)
    assert encoded[0].visual_rows is not None


def test_vae_stages_accept_independent_component_placement() -> None:
    manager = ModuleManager(device="cpu")
    manager.add_module(_VideoVAEProbe(), "minimax_h3_video_vae")
    manager.add_module(torch.nn.Linear(1, 1), "minimax_h3_audio_vae")
    video_config = ModelRuntimeConfig(device_type="cpu", device_id=0)
    audio_config = ModelRuntimeConfig(device_type="cpu", device_id=1)

    video_stage = MiniMaxH3VideoVAEStage(manager, video_config)
    audio_stage = MiniMaxH3AudioVAEStage(manager, audio_config)

    assert video_stage.model_runtime_config is video_config
    assert audio_stage.model_runtime_config is audio_config


def test_video_decode_keeps_cpu_path_in_fp32() -> None:
    video_vae = _VideoVAEProbe()
    manager = ModuleManager(device="cpu")
    manager.add_module(video_vae, "minimax_h3_video_vae")
    manager.add_module(torch.nn.Linear(1, 1), "minimax_h3_audio_vae")
    runtime = ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32)
    stage = MiniMaxH3VideoVAEStage(manager, runtime)

    frames = stage.decode_video(torch.zeros(1, 24, 1, 2, 2))

    assert frames.shape == (1, 1, 2, 2, 3)
    assert video_vae.prepared_dtype is None
