"""Regression tests for pre-quantized Qwen-Image FP8 checkpoints."""

import torch

from telefuser.models.qwen_image_dit import QwenImageDiT, QwenImageDitStateDictConverter
from telefuser.ops.quantized_linear import LinearFP8
from telefuser.utils.model_weight import init_weights_on_device


def test_converter_marks_prequantized_fp8_checkpoint() -> None:
    converter = QwenImageDitStateDictConverter()
    state_dict = {
        "transformer_blocks.0.img_mod.1.weight": torch.empty(4, 4, dtype=torch.float8_e4m3fn),
        "transformer_blocks.0.img_mod.1.weight_scale": torch.ones(4, 1, dtype=torch.bfloat16),
    }

    converted = converter.from_official(state_dict)

    assert isinstance(converted, tuple)
    assert converted[0] is state_dict
    assert converted[1] == {"prequantized_fp8": True}


def test_qwen_dit_builds_scale_aware_fp8_block_modules_on_meta() -> None:
    with init_weights_on_device("meta"):
        model = QwenImageDiT(num_layers=1, prequantized_fp8=True)

    layer = model.transformer_blocks[0].img_mod[1]
    assert isinstance(layer, LinearFP8)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (18432, 1)
