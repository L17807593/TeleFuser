# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from telefuser.utils.lora_loader import LoRALoader, LoRATarget


def test_loader_preserves_existing_mapping_and_scaling_behavior():
    model = nn.Linear(3, 2, bias=False)
    before = model.weight.detach().clone()
    lora_weights = {
        "weight.lora_up.weight": torch.ones(2, 1),
        "weight.lora_down.weight": torch.ones(1, 3),
    }

    applied = LoRALoader().apply_lora(
        {"weight.weight": model.weight},
        lora_weights,
        strength=0.5,
    )

    assert applied == 1
    torch.testing.assert_close(model.weight, before + 0.5)


def test_loader_streams_safetensors_and_merges_multiple_slices(tmp_path):
    parameter = nn.Parameter(torch.zeros(6, 3))
    checkpoint_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "branch_a.lora_A.default.weight": torch.full((2, 3), 1.0),
            "branch_a.lora_B.default.weight": torch.full((2, 2), 1.0),
            "branch_b.lora_A.default.weight": torch.full((2, 3), 1.0),
            "branch_b.lora_B.default.weight": torch.full((2, 2), 1.0),
        },
        checkpoint_path,
        metadata={"alpha": "4"},
    )

    def resolve_target(model_key: str, weights: Mapping[str, torch.Tensor]) -> LoRATarget | None:
        offsets = {"branch_a.weight": 0, "branch_b.weight": 2}
        if model_key not in offsets:
            return None
        offset = offsets[model_key]
        return LoRATarget("qkv.weight", weights["qkv.weight"][offset : offset + 2])

    loader = LoRALoader(
        target_resolver=resolve_target,
        strict=True,
        default_alpha=8.0,
        stream_safetensors=True,
        merge_dtype=torch.float32,
    )
    applied = loader.apply_lora(
        {"qkv.weight": parameter},
        checkpoint_path,
        strength=0.5,
    )

    assert applied == 2
    torch.testing.assert_close(parameter[:4], torch.full((4, 3), 2.0))
    torch.testing.assert_close(parameter[4:], torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("lora_weights", "message"),
    [
        ({"weight.lora_up.weight": torch.ones(2, 1)}, "incomplete LoRA pairs"),
        (
            {
                "missing.lora_up.weight": torch.ones(2, 1),
                "missing.lora_down.weight": torch.ones(1, 3),
            },
            "Model key not found",
        ),
    ],
)
def test_loader_strict_mode_rejects_invalid_checkpoint(
    lora_weights: dict[str, torch.Tensor],
    message: str,
):
    model = nn.Linear(3, 2, bias=False)

    with pytest.raises(ValueError, match=message):
        LoRALoader(strict=True).apply_lora({"weight.weight": model.weight}, lora_weights)
