"""Tests for activation functions."""

import pytest
import torch
import torch.nn as nn

from telefuser.ops.activations import (
    ACT2CLS,
    GEGLU,
    GELU,
    ApproximateGELU,
    FP32SiLU,
    LinearActivation,
    SwiGLU,
    gelu_and_mul,
    get_activation,
    silu_and_mul,
    silu_and_mul_reuse_input,
)
from tests.conftest import requires_cuda


class TestGetActivation:
    """Test get_activation helper function."""

    @pytest.mark.parametrize(
        "name,expected_type",
        [
            ("swish", nn.SiLU),
            ("silu", nn.SiLU),
            ("mish", nn.Mish),
            ("gelu", nn.GELU),
            ("relu", nn.ReLU),
        ],
    )
    def test_get_activation(self, name, expected_type):
        """Test getting various activations by name."""
        act = get_activation(name)
        assert isinstance(act, expected_type)

    @pytest.mark.parametrize("name", ["SiLU", "silu", "SILU"])
    def test_case_insensitive(self, name):
        """Test that activation names are case insensitive."""
        act = get_activation(name)
        assert isinstance(act, nn.SiLU)

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="activation function invalid not found"):
            get_activation("invalid")


def test_fused_gated_activations_fall_back_for_cpu_tensors() -> None:
    x = torch.randn(2, 8)
    gate, value = x.chunk(2, dim=-1)

    torch.testing.assert_close(silu_and_mul(x), torch.nn.functional.silu(gate) * value)
    torch.testing.assert_close(gelu_and_mul(x), torch.nn.functional.gelu(gate) * value)


def test_reuse_input_falls_back_without_mutating_cpu_input() -> None:
    value = torch.randn(3, 8)
    original = value.clone()

    torch.testing.assert_close(silu_and_mul_reuse_input(value), silu_and_mul(original))
    torch.testing.assert_close(value, original)


@requires_cuda
def test_reuse_input_uses_rounded_inplace_cuda_path() -> None:
    value = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
    original = value.clone()
    gate, up = original.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate).to(torch.bfloat16) * up

    output = silu_and_mul_reuse_input(value)

    assert output.data_ptr() == value.data_ptr()
    assert torch.equal(output, expected)


class TestFP32SiLU:
    """Test FP32SiLU activation."""

    def test_forward_dtype_preservation(self):
        """Test output dtype matches input dtype."""
        act = FP32SiLU()
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 10, 64, dtype=dtype)
            output = act(x)
            assert output.shape == x.shape
            assert output.dtype == dtype

    def test_forward_computation(self):
        """Test forward produces correct SiLU values."""
        act = FP32SiLU()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
        output = act(x)
        expected = x * torch.sigmoid(x)  # SiLU(x) = x * sigmoid(x)
        assert torch.allclose(output, expected, atol=1e-5)


class TestGELU:
    """Test GELU activation module."""

    def test_forward_shape_and_computation(self):
        """Test forward pass shape and computation."""
        gelu = GELU(dim_in=64, dim_out=128)
        x = torch.randn(2, 10, 64)

        output = gelu(x)

        assert output.shape == (2, 10, 128)
        # Verify GELU was applied (values should change)
        linear_output = gelu.proj(x)
        assert not torch.allclose(output, linear_output)

    def test_approximate_modes(self):
        """Test different GELU approximation modes."""
        for approximate in ["none", "tanh"]:
            gelu = GELU(dim_in=32, dim_out=32, approximate=approximate)
            x = torch.randn(2, 10, 32)
            output = gelu(x)
            assert output.shape == (2, 10, 32)


class TestGEGLU:
    """Test GEGLU activation module."""

    @requires_cuda
    def test_forward_computation(self):
        """Test forward produces correct output shape."""
        geglu = GEGLU(dim_in=64, dim_out=128).cuda()
        x = torch.randn(2, 10, 64, device="cuda")

        output = geglu(x)

        assert output.shape == (2, 10, 128)
        # Output should be non-zero for random input
        assert not torch.allclose(output, torch.zeros_like(output))


class TestSwiGLU:
    """Test SwiGLU activation module."""

    @requires_cuda
    def test_forward_computation(self):
        """Test forward produces correct output."""
        swiglu = SwiGLU(dim_in=64, dim_out=128).cuda()
        x = torch.randn(2, 10, 64, device="cuda")

        output = swiglu(x)

        assert output.shape == (2, 10, 128)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestApproximateGELU:
    """Test ApproximateGELU activation module."""

    def test_approximation_formula(self):
        """Test that approximation formula is correct."""
        gelu = ApproximateGELU(dim_in=5, dim_out=5)
        gelu.proj.weight.data = torch.eye(5)
        if gelu.proj.bias is not None:
            gelu.proj.bias.data.fill_(0.0)

        x = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]])
        output = gelu(x)

        # Approximate GELU: x * sigmoid(1.702 * x)
        expected = x * torch.sigmoid(1.702 * x)
        assert torch.allclose(output, expected, atol=1e-5)


class TestLinearActivation:
    """Test LinearActivation module."""

    @pytest.mark.parametrize("activation_name", ["silu", "gelu", "relu"])
    def test_various_activations(self, activation_name):
        """Test LinearActivation with different activation functions."""
        layer = LinearActivation(dim_in=64, dim_out=128, activation=activation_name)
        x = torch.randn(2, 10, 64)
        output = layer(x)
        assert output.shape == (2, 10, 128)

    def test_applies_activation(self):
        """Test that activation is actually applied."""
        layer = LinearActivation(dim_in=64, dim_out=64)
        layer.proj.weight.data = torch.eye(64) * 0.5
        if layer.proj.bias is not None:
            layer.proj.bias.data.fill_(0.0)

        x = torch.ones(1, 1, 64)
        output = layer(x)

        # SiLU should reduce positive values
        assert (output < x).all()


class TestACT2CLS:
    """Test ACT2CLS dictionary."""

    def test_all_activations_instantiable(self):
        """Test that all activations can be instantiated."""
        for name, cls in ACT2CLS.items():
            act = cls()
            assert isinstance(act, nn.Module)
            # Verify swish and silu are the same
            if name in ["swish", "silu"]:
                assert cls == nn.SiLU
