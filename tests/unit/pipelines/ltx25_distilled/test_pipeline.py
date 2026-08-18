"""End-to-end shape contract for the lazy public LTX-2.5 pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from telefuser.models.ltx25.checkpoint import LTX25ModelPaths
from telefuser.pipelines.ltx25_distilled.pipeline import LTX25DistilledConfig, LTX25DistilledPipeline


class _Transformer(torch.nn.Module):
    def forward(
        self, video: object, audio: object, perturbations: object
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del perturbations
        return (
            torch.zeros_like(video.latent) if video is not None else None,  # type: ignore[union-attr]
            torch.zeros_like(audio.latent) if audio is not None else None,  # type: ignore[union-attr]
        )


class _Upsampler(torch.nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)


class _VideoDecoder(torch.nn.Module):
    def decode_video(self, latent: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        del generator
        return (latent,)


class _ConvVideoDecoder(torch.nn.Module):
    def decode(self, latent: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        del generator
        return (latent[:, :3],)


class _Identity(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Statistics(torch.nn.Module):
    def un_normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class _DurationHead(torch.nn.Module):
    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        del video, audio
        return torch.tensor([1.0])


class _TrackingModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.devices: list[str] = []

    def to(self, *args: object, **kwargs: object) -> "_TrackingModule":
        del kwargs
        self.devices.append(str(args[0]))
        return self


def test_public_pipeline_runs_two_stage_t2v_contract_with_loaded_component_protocols(monkeypatch) -> None:
    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.torch_dtype = torch.bfloat16
    pipeline.paths = LTX25ModelPaths(*(Path("unused") for _ in range(7)))
    pipeline.config = LTX25DistilledConfig(model_root="unused", device="cpu")
    pipeline._encode_prompt = lambda prompt: (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25AVTransformer.from_checkpoint",
        lambda *args, **kwargs: _Transformer(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25SpatialUpsampler.from_checkpoint",
        lambda *args, **kwargs: _Upsampler(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.load_video_latent_statistics",
        lambda *args, **kwargs: _Statistics(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.DiffusionVideoDecoder.from_checkpoint",
        lambda *args, **kwargs: _VideoDecoder(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.load_ltx25_audio_decoder_and_vocoder",
        lambda *args, **kwargs: (_Identity(), _Identity()),
    )

    result = pipeline(
        "A test prompt",
        seed=7,
        height=256,
        width=384,
        num_frames=9,
        frame_rate=24.0,
    )

    assert result.video_latent.shape == (1, 128, 2, 8, 12)
    assert result.audio_latent.shape == (1, 8, 9, 16)
    assert len(result.video_chunks) == 1
    assert result.video_chunks[0] is result.video_latent
    assert result.audio.shape == result.audio_latent.squeeze(0).shape
    assert result.audio.dtype == torch.float32


def test_public_pipeline_resolves_auto_duration_after_prompt_encoding(monkeypatch) -> None:
    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.torch_dtype = torch.bfloat16
    pipeline.paths = LTX25ModelPaths(*(Path("unused") for _ in range(7)))
    pipeline.config = LTX25DistilledConfig(model_root="unused", device="cpu")
    pipeline._encode_prompt = lambda prompt: (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25DurationHead.from_checkpoint",
        lambda *args, **kwargs: _DurationHead(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25AVTransformer.from_checkpoint",
        lambda *args, **kwargs: _Transformer(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25SpatialUpsampler.from_checkpoint",
        lambda *args, **kwargs: _Upsampler(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.load_video_latent_statistics",
        lambda *args, **kwargs: _Statistics(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.DiffusionVideoDecoder.from_checkpoint",
        lambda *args, **kwargs: _VideoDecoder(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.load_ltx25_audio_decoder_and_vocoder",
        lambda *args, **kwargs: (_Identity(), _Identity()),
    )

    result = pipeline("A test prompt", seed=7, height=256, width=384)

    assert result.num_frames == 25
    assert result.video_latent.shape == (1, 128, 4, 8, 12)


def test_public_pipeline_uses_selected_conv_vae_for_bridge_and_decode(monkeypatch) -> None:
    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.torch_dtype = torch.bfloat16
    pipeline.paths = LTX25ModelPaths(
        Path("transformer"),
        Path("text"),
        Path("diff-video"),
        Path("conv-video"),
        Path("audio"),
        Path("upsampler"),
        Path("duration"),
    )
    pipeline.config = LTX25DistilledConfig(model_root="unused", device="cpu", video_vae="conv")
    pipeline._encode_prompt = lambda prompt: (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25AVTransformer.from_checkpoint",
        lambda *args, **kwargs: _Transformer(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25SpatialUpsampler.from_checkpoint",
        lambda *args, **kwargs: _Upsampler(),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.load_video_latent_statistics",
        lambda path: _Statistics() if path == Path("conv-video") else pytest.fail("wrong VAE statistics path"),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.DiffusionVideoDecoder.from_checkpoint",
        lambda *args, **kwargs: pytest.fail("DiffVAE decoder must not load for video_vae='conv'"),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.LTX25ConvVideoVAE.from_checkpoint",
        lambda path, **kwargs: _ConvVideoDecoder() if path == Path("conv-video") else pytest.fail("wrong ConvVAE path"),
    )
    monkeypatch.setattr(
        "telefuser.pipelines.ltx25_distilled.pipeline.load_ltx25_audio_decoder_and_vocoder",
        lambda *args, **kwargs: (_Identity(), _Identity()),
    )

    result = pipeline("A test prompt", seed=7, height=256, width=384, num_frames=9)

    assert result.video_chunks[0].shape == (2, 8, 12, 3)
    assert torch.all((result.video_chunks[0] >= 0) & (result.video_chunks[0] <= 1))


def test_pipeline_offload_policy_controls_phase_release() -> None:
    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.config = LTX25DistilledConfig(model_root="unused", device="cpu", offload="cpu")
    cpu_module = _TrackingModule()
    pipeline._release(cpu_module)
    assert cpu_module.devices == ["cpu"]

    pipeline.config = LTX25DistilledConfig(model_root="unused", device="cpu", offload="none")
    resident_module = _TrackingModule()
    pipeline._release(resident_module)
    assert resident_module.devices == []
