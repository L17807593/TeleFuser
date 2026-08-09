from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from examples.swiftvr.swiftvr_restore_h100 import _warmup_frame_count, main, run


class FakePipeline:
    def __init__(self) -> None:
        self.frames: torch.Tensor | None = None
        self.options: dict[str, object] = {}
        self.step_sizes: list[int] = []
        self.closed = False

    def stream(self, **options: object) -> "FakePipeline":
        self.options = options
        return self

    def step(self, frames: torch.Tensor) -> torch.Tensor:
        self.frames = frames
        self.step_sizes.append(len(frames))
        return frames.permute(0, 3, 1, 2).unsqueeze(0).float() / 255

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_run_uses_flashvsr_style_pil_interface() -> None:
    pipeline = FakePipeline()
    inputs = [Image.fromarray(np.full((10, 18, 3), value, dtype=np.uint8)) for value in (17, 193)]

    outputs = run(
        pipeline,
        inputs,
        scale=2,
    )

    assert pipeline.frames is not None
    assert pipeline.frames.shape == (2, 8, 16, 3)
    assert pipeline.frames.dtype == torch.uint8
    assert pipeline.options == {"upscale": 2}
    assert pipeline.step_sizes == [2]
    assert pipeline.closed is True
    assert [image.size for image in outputs] == [(16, 8), (16, 8)]
    assert np.asarray(outputs[0])[0, 0].tolist() == [17, 17, 17]
    assert np.asarray(outputs[1])[0, 0].tolist() == [193, 193, 193]


def test_run_rejects_empty_video() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        run(FakePipeline(), [])


def test_run_processes_24_frame_streaming_chunks() -> None:
    pipeline = FakePipeline()
    inputs = [Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)) for value in range(25)]

    outputs = run(pipeline, inputs, scale=3)

    assert pipeline.step_sizes == [24, 1]
    assert pipeline.options == {"upscale": 3}
    assert len(outputs) == 25


def test_warmup_covers_full_and_tail_shapes() -> None:
    assert _warmup_frame_count(81) == 57
    assert _warmup_frame_count(72) == 48
    assert _warmup_frame_count(25) == 25


def test_default_input_is_flashvsr_example_video() -> None:
    input_option = next(parameter for parameter in main.params if parameter.name == "input_video")
    input_video = Path(input_option.default)

    assert input_video.name == "dag.mp4"
    assert input_video.parent.name == "data"
    assert input_video.is_file()


def test_cli_uses_flashvsr_style_options() -> None:
    assert {parameter.name for parameter in main.params} == {
        "input_video",
        "scale",
        "height",
        "width",
        "gpu_num",
        "model_root",
        "output",
    }
