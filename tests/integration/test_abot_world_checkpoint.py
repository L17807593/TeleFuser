from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

try:
    from safetensors import safe_open
except ImportError:
    pytest.skip("safetensors is required for ABot checkpoint contract tests", allow_module_level=True)

from telefuser.models.abot_world_dit import ABotWorldDiT


@pytest.mark.filesystem
def test_official_abot_checkpoint_has_the_exact_native_dit_contract() -> None:
    model_root = os.environ.get("ABOT_WORLD_MODEL_ROOT")
    if not model_root:
        pytest.skip("set ABOT_WORLD_MODEL_ROOT to run the ABot checkpoint contract test")
    checkpoint = Path(model_root) / "diffusion_pytorch_model.safetensors"
    if not checkpoint.is_file():
        pytest.skip("ABot DiT checkpoint is unavailable")

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        official = {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}
    with torch.device("meta"):
        model = ABotWorldDiT()
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}

    assert set(official) == set(expected)
    mismatches = {name: (official[name], expected[name]) for name in expected if official[name] != expected[name]}
    assert not mismatches
