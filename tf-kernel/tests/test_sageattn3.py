import pytest
import torch
import torch.nn.functional as F

from tf_kernel import FP4_AVAILABLE


def torch_sageattn_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
):
    """
    Reference implementation of self-attention using PyTorch.
    Input layout: HND (batch_size, num_heads, seq_len, head_dim)
    """
    # q, k, v: [batch_size, num_heads, seq_len, head_dim]
    b, h, seq_len, head_dim = q.shape

    # Compute attention scores: [b, h, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1))

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    scores = scores * sm_scale

    if is_causal:
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values: [b, h, seq_len, head_dim]
    output = torch.matmul(attn_weights, v)

    return output


def _create_self_attention_tensors(batch_size, num_heads, seq_len, head_dim, dtype):
    """Create test tensors for self-attention (q, k, v from same source)."""
    # HND layout: [batch_size, num_heads, seq_len, head_dim]
    x = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype
    )
    return x, x.clone(), x.clone()


SAGEATTN3_AVAILABLE = (
    FP4_AVAILABLE
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (10, 0)
    and hasattr(torch.ops.tf_kernel, "sageattn3_fp4_attn")
)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.sm100,
    pytest.mark.skipif(
        not SAGEATTN3_AVAILABLE,
        reason="SageAttention3 requires a Blackwell wheel with FP4 operators",
    ),
]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn3_blackwell_self_attention(
    batch_size, num_heads, seq_len, head_dim, is_causal, dtype
):
    """Test sageattn3_blackwell for self-attention."""
    from tf_kernel import sageattn3_blackwell

    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype
    )

    # Run the kernel implementation
    o_kernel = sageattn3_blackwell(
        q,
        k,
        v,
        is_causal=is_causal,
        per_block_mean=True,
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        is_causal=is_causal,
    )

    # Compare results with relaxed tolerance due to FP4 quantization
    # FP4 quantization has lower precision than int8/fp8
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.2, atol=0.2)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn3_blackwell_per_block_mean(
    batch_size, num_heads, seq_len, head_dim, dtype
):
    """Test sageattn3_blackwell with per_block_mean=False."""
    from tf_kernel import sageattn3_blackwell

    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype
    )

    # Run with per_block_mean=False
    o_kernel = sageattn3_blackwell(
        q,
        k,
        v,
        is_causal=False,
        per_block_mean=False,
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        is_causal=False,
    )

    # Compare results
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.2, atol=0.2)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn3_blackwell_output_shape(
    batch_size, num_heads, seq_len, head_dim, dtype
):
    """Test sageattn3_blackwell output shape is correct."""
    from tf_kernel import sageattn3_blackwell

    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype
    )

    # Run the kernel implementation
    o_kernel = sageattn3_blackwell(
        q,
        k,
        v,
        is_causal=False,
        per_block_mean=True,
    )

    # Check output shape matches input shape
    assert o_kernel.shape == q.shape
    assert o_kernel.dtype == q.dtype
    assert o_kernel.device == q.device


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len_q", [128, 256])
@pytest.mark.parametrize("seq_len_kv", [128, 256])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn3_blackwell_cross_attention(
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim, dtype
):
    """Test sageattn3_blackwell for cross-attention (different seq_len for q and kv)."""
    from tf_kernel import sageattn3_blackwell

    # Create q with different sequence length than k and v
    q = torch.randn(
        batch_size, num_heads, seq_len_q, head_dim, device="cuda", dtype=dtype
    )
    k = torch.randn(
        batch_size, num_heads, seq_len_kv, head_dim, device="cuda", dtype=dtype
    )
    v = torch.randn(
        batch_size, num_heads, seq_len_kv, head_dim, device="cuda", dtype=dtype
    )

    # Run the kernel implementation
    o_kernel = sageattn3_blackwell(
        q,
        k,
        v,
        is_causal=False,
        per_block_mean=True,
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        is_causal=False,
    )

    # Compare results
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.2, atol=0.2)


if __name__ == "__main__":
    pytest.main([__file__])
