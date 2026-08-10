import threading
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from telefuser.models.swiftvr_reae import MemBlock, TPool
from telefuser.pipelines.swiftvr import pipeline as pipeline_module
from telefuser.pipelines.swiftvr.chunk import ChunkType, build_chunk_specs
from telefuser.pipelines.swiftvr.pipeline import SwiftVRStagedStreamSession, SwiftVRStreamSession, aligned_pad
from telefuser.pipelines.swiftvr.streaming_tae import apply_parallel_with_boundary


def test_aligned_pad() -> None:
    assert aligned_pad(32) == 0
    assert aligned_pad(33) == 31
    assert aligned_pad(1080) == 8
    assert aligned_pad(1440) == 0


def test_chunk_specs_preserve_frame_count_and_tail() -> None:
    specs = build_chunk_specs(53, 24)

    assert [spec.ctype for spec in specs] == [ChunkType.FIRST, ChunkType.MIDDLE, ChunkType.LAST]
    assert sum(spec.frame_count for spec in specs) == 53
    assert specs[-1].frame_count == 1
    assert specs[-1].b == 0
    assert specs[0].is_first_decode is True


def test_memblock_boundary_matches_whole_sequence() -> None:
    torch.manual_seed(7)
    model = nn.Sequential(MemBlock(2, 2)).eval()
    inputs = torch.randn(1, 5, 2, 4, 4)

    whole, _ = apply_parallel_with_boundary(model, inputs)
    first, state = apply_parallel_with_boundary(model, inputs[:, :2])
    second, _ = apply_parallel_with_boundary(model, inputs[:, 2:], state)

    torch.testing.assert_close(torch.cat([first, second], dim=1), whole)


def test_temporal_pool_boundary_carries_non_aligned_tail() -> None:
    torch.manual_seed(11)
    model = nn.Sequential(TPool(2, 2)).eval()
    inputs = torch.randn(1, 6, 2, 3, 3)

    whole, _ = apply_parallel_with_boundary(model, inputs)
    first, state = apply_parallel_with_boundary(model, inputs[:, :3])
    second, state = apply_parallel_with_boundary(model, inputs[:, 3:], state)

    assert state["tpool_0"] is None
    torch.testing.assert_close(torch.cat([first, second], dim=1), whole)


def test_stream_sessions_do_not_share_boundary_or_rope_state() -> None:
    pipeline = SimpleNamespace(
        reae=object(),
        transformer=object(),
        _execution_lock=threading.RLock(),
    )
    first = SwiftVRStreamSession(
        pipeline,
        clip_len=24,
        resolution=None,
        upscale=4,
        dit_overlap=1,
    )
    second = SwiftVRStreamSession(
        pipeline,
        clip_len=24,
        resolution=None,
        upscale=4,
        dit_overlap=1,
    )

    first._tae._enc_st = {"mem_1": torch.ones(1)}
    first._dit._g_off = 9

    assert second._tae._enc_st is None
    assert second._dit._g_off == 0
    first.close()
    assert first._tae._enc_st is None
    assert first._dit._g_off == 0
    assert second._closed is False


def test_interleaved_stream_sessions_do_not_exchange_frames(monkeypatch) -> None:
    class FakeTAE:
        def __init__(self, _model: object) -> None:
            self.calls = 0

        def encode_chunk(self, frames: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return frames + self.calls / 100

        def decode_chunk(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

        def flush_encoder(self) -> None:
            return None

        def reset(self) -> None:
            self.calls = 0

    class FakeDiT:
        def __init__(self, _model: object, overlap: int) -> None:
            self.calls = 0
            self.overlap = overlap
            self._cond_cache = None
            self._cond_cache_key = None

        def denoise(self, latents: torch.Tensor, _prompt: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return latents + self.calls / 10

        def reset(self) -> None:
            self.calls = 0

    class FakePipeline:
        reae = object()
        transformer = object()
        prompt_emb = torch.empty(0)
        device = torch.device("cpu")
        torch_dtype = torch.float32
        upscale_mode = "nearest"
        _execution_lock = threading.RLock()

        @staticmethod
        def _target_size(
            lq_h: int,
            lq_w: int,
            _resolution: tuple[int, int] | None,
            _upscale: int,
        ) -> tuple[int, int, int, int]:
            return lq_h, lq_w, 0, 0

    monkeypatch.setattr(pipeline_module, "StreamingTAE", FakeTAE)
    monkeypatch.setattr(pipeline_module, "StreamingDiT", FakeDiT)
    pipeline = FakePipeline()
    first = SwiftVRStreamSession(pipeline, clip_len=24, resolution=None, upscale=1, dit_overlap=1)
    second = SwiftVRStreamSession(pipeline, clip_len=24, resolution=None, upscale=1, dit_overlap=1)
    control = SwiftVRStreamSession(pipeline, clip_len=24, resolution=None, upscale=1, dit_overlap=1)
    first_frames = torch.full((4, 2, 2, 3), 17, dtype=torch.uint8)
    second_frames = torch.full((4, 2, 2, 3), 193, dtype=torch.uint8)

    first.step(first_frames)
    second_first = second.step(second_frames)
    control_first = control.step(second_frames)
    first.step(first_frames)
    first.close()
    second_second = second.step(second_frames)
    control_second = control.step(second_frames)

    assert [np.asarray(frame).tolist() for frame in second_first] == [
        np.asarray(frame).tolist() for frame in control_first
    ]
    assert [np.asarray(frame).tolist() for frame in second_second] == [
        np.asarray(frame).tolist() for frame in control_second
    ]
    assert second._closed is False


def test_stage_restore_chunks_converts_decoded_tensors_to_pil(monkeypatch) -> None:
    class FakeWorker:
        def __init__(self, role: str) -> None:
            self.role = role
            self.calls = 0

        def process(self, value: object, *args: object, **kwargs: object):
            self.calls += 1
            if self.role == "encode":
                result = torch.tensor(float(self.calls))
            elif self.role == "dit":
                result = value
            else:
                result = torch.full((1, 1, 3, 2, 2), float(self.calls) / 10)
            return lambda: result

        def flush_encoder(self, **kwargs: object):
            return lambda: None

        def reset_session(self, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(pipeline_module, "ParallelWorker", FakeWorker)
    pipeline = SimpleNamespace(
        encode_stage=FakeWorker("encode"),
        dit_stage=FakeWorker("dit"),
        decode_stage=FakeWorker("decode"),
        upscale_mode="nearest",
        _target_size=lambda height, width, resolution, upscale: (height, width, 0, 0),
        _stage_parallel_active_session=True,
    )
    session = SwiftVRStagedStreamSession(
        pipeline,
        clip_len=24,
        resolution=None,
        upscale=1,
    )

    output = session.restore_chunks(torch.zeros((48, 2, 2, 3), dtype=torch.uint8), clip_len=24)

    assert len(output) == 2
    assert all(isinstance(frame, Image.Image) for frame in output)
    assert [np.asarray(frame)[0, 0, 0] for frame in output] == [25, 51]
