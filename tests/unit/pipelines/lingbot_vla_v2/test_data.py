from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from telefuser.pipelines.lingbot_vla_v2.data import LingBotVlaV2InputProcessor, LingBotVlaV2Observation
from telefuser.pipelines.lingbot_vla_v2.robot_profile import ROBOTWIN_CAMERA_KEYS, RobotWinProfile


class _ImageProcessor:
    def __init__(self) -> None:
        self.values: list[int] = []

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        assert image.dtype == torch.uint8
        assert image.shape == (3, 8, 8)
        value = int(image[0, 0, 0])
        self.values.append(value)
        return {
            "pixel_values": torch.full((4, 6), float(value)),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


class _Tokenizer:
    def __init__(self) -> None:
        self.rendered_task: str | None = None
        self.padding_side: str | None = None

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        self.rendered_task = messages[0]["content"]
        return f"chat:{self.rendered_task}"

    def __call__(self, prompts, **kwargs) -> dict[str, torch.Tensor]:
        assert prompts == [f"chat:{self.rendered_task}"]
        self.padding_side = kwargs["padding_side"]
        length = kwargs["max_length"]
        return {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length),
        }


def _processor() -> tuple[LingBotVlaV2InputProcessor, _ImageProcessor, _Tokenizer]:
    image_processor = _ImageProcessor()
    tokenizer = _Tokenizer()
    processor = SimpleNamespace(image_processor=image_processor, tokenizer=tokenizer)
    config = SimpleNamespace(max_state_dim=55, tokenizer_max_length=6)
    return (
        LingBotVlaV2InputProcessor(processor, config, RobotWinProfile.default(), image_size=8),
        image_processor,
        tokenizer,
    )


def _observation() -> LingBotVlaV2Observation:
    images = {
        ROBOTWIN_CAMERA_KEYS[0]: np.full((8, 8, 3), 10, dtype=np.uint8),
        ROBOTWIN_CAMERA_KEYS[1]: np.full((8, 8, 3), 20, dtype=np.uint8),
        ROBOTWIN_CAMERA_KEYS[2]: np.full((8, 8, 3), 30, dtype=np.uint8),
    }
    return LingBotVlaV2Observation(task="pick up the block", state=[0.0] * 14, images=images)


def test_prepare_preserves_robotwin_camera_order_and_tensor_contract() -> None:
    processor, image_processor, tokenizer = _processor()
    observation = _observation()

    inputs = processor.prepare(observation)

    assert image_processor.values == [10, 20, 30]
    assert tokenizer.rendered_task == observation.task
    assert tokenizer.padding_side == "right"
    assert inputs.images.shape == (1, 3, 4, 6)
    assert inputs.img_masks.tolist() == [[True, True, True]]
    assert inputs.image_grid_thw.shape == (1, 3, 3)
    assert inputs.lang_tokens.shape == (1, 6)
    assert inputs.lang_masks.dtype == torch.bool
    assert torch.equal(inputs.state, processor.robot_profile.normalize_state(observation.state).unsqueeze(0))


def test_prepare_rejects_a_missing_robotwin_camera() -> None:
    processor, _, _ = _processor()
    observation = _observation()
    images = dict(observation.images)
    del images[ROBOTWIN_CAMERA_KEYS[1]]

    with pytest.raises(ValueError, match="missing camera keys"):
        processor.prepare(LingBotVlaV2Observation(observation.task, observation.state, images))


def test_prepare_scales_unit_float_images_to_uint8() -> None:
    processor, image_processor, _ = _processor()
    observation = _observation()
    images = dict(observation.images)
    images[ROBOTWIN_CAMERA_KEYS[0]] = np.full((3, 8, 8), 0.5, dtype=np.float32)

    processor.prepare(LingBotVlaV2Observation(observation.task, observation.state, images))

    assert image_processor.values[0] == 128


def test_prepare_resizes_each_camera_before_qwen_processing() -> None:
    processor, image_processor, _ = _processor()
    observation = _observation()
    images = {
        key: np.full((12, 16, 3), value, dtype=np.uint8)
        for key, value in zip(ROBOTWIN_CAMERA_KEYS, (10, 20, 30), strict=True)
    }

    processor.prepare(LingBotVlaV2Observation(observation.task, observation.state, images))

    assert image_processor.values == [10, 20, 30]
