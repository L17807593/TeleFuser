"""Tests for quantized linear layers."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

# Keep the module import optional for CPU-only test environments.
try:
    from telefuser.ops.quantized_linear import (
        LinearFP8,
        convert_params_to_buffers,
        replace_linear_layers,
    )

    HAS_QUANTIZE = True
except ImportError:
    HAS_QUANTIZE = False


pytestmark = [
    pytest.mark.skipif(not HAS_QUANTIZE, reason="Quantize module not available"),
    pytest.mark.quant,
]


class TestLinearFP8Initialization:
    """Test LinearFP8 initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        original = nn.Linear(64, 128)
        layer = LinearFP8(original, torch.float8_e4m3fn)

        assert layer.weight.shape == (128, 64)
        assert layer.bias.shape == (128,)
        assert layer.weight_scale.shape == (128, 1)

    def test_weight_scale_initialized_to_zero(self):
        """Test that weight_scale is initialized to zeros."""
        original = nn.Linear(32, 64)
        layer = LinearFP8(original, torch.float8_e4m3fn)

        assert torch.allclose(layer.weight_scale, torch.zeros(64, 1))

    def test_forward_requires_tf_kernel(self):
        """Test that missing tf-kernel produces an actionable error."""
        original = nn.Linear(32, 64)
        layer = LinearFP8(original, torch.float8_e4m3fn)

        with patch("telefuser.ops.quantized_linear.tf_kernel", None):
            with pytest.raises(RuntimeError, match="install tf-kernel"):
                layer(torch.randn(2, 32))


class TestConvertParamsToBuffers:
    """Test convert_params_to_buffers function."""

    def test_converts_params_to_buffers(self):
        """Test that parameters are converted to buffers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 30),
        )

        # Check initial state - should have parameters
        initial_params = list(model.parameters())
        assert len(initial_params) > 0

        # Convert
        convert_params_to_buffers(model)

        # Check that parameters are now buffers
        # The function converts all params except FP8e4m3fn type
        buffers = list(model.buffers())
        assert len(buffers) >= len(initial_params)

    def test_preserves_fp8_params(self):
        """Test that FP8e4m3fn params are preserved."""

        class FP8Model(nn.Module):
            def __init__(self):
                super().__init__()
                # Use float32 instead of float8 for compatibility
                self.normal_param = nn.Parameter(torch.randn(10, 10))

        model = FP8Model()

        # Convert
        convert_params_to_buffers(model, ignore_dtype=torch.float32)

        # With ignore_dtype=torch.float32, normal_param should still be a parameter
        assert isinstance(model.normal_param, nn.Parameter)


class TestReplaceLinearLayers:
    """Test replace_linear_layers function."""

    def test_replaces_linear_layers(self):
        """Test that Linear layers are replaced with LinearFP8."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
        )

        # Don't use mock, let it actually replace
        replace_linear_layers(model, torch.float8_e4m3fn)

        # Check that Linear layers are replaced
        assert isinstance(model[0], LinearFP8)
        assert isinstance(model[2], LinearFP8)

    def test_handles_nested_modules(self):
        """Test handling of nested modules."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.block = nn.Sequential(
                    nn.Linear(20, 30),
                    nn.ReLU(),
                )

        model = NestedModel()

        replace_linear_layers(model, torch.float8_e4m3fn)

        # Check that Linear layers are replaced (3 total)
        assert isinstance(model.layer1, LinearFP8)
        assert isinstance(model.block[0], LinearFP8)
        assert isinstance(model.block[1], nn.ReLU)

    def test_preserves_non_linear_layers(self):
        """Test that non-Linear layers are preserved."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
        )

        replace_linear_layers(model, torch.float8_e4m3fn)

        # ReLU and BatchNorm should not be touched
        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.BatchNorm1d)
