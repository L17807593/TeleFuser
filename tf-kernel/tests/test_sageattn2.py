import pytest
import torch
import torch.nn.functional as F

from tf_kernel import (sageattn_qk_int8_pv_fp8_cuda,
                       sageattn_qk_int8_pv_fp8_cuda_sm90,
                       sageattn_qk_int8_pv_fp16_cuda,
                       sageattn_qk_int8_pv_fp16_triton)


def torch_sageattn_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float = None,
):
    """
    Reference implementation of self-attention using PyTorch.
    """
    if tensor_layout == "HND":
        # q: [batch_size, num_qo_heads, qo_len, head_dim]
        # k: [batch_size, num_kv_heads, kv_len, head_dim]
        # v: [batch_size, num_kv_heads, kv_len, head_dim]
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        # Compute attention scores: [b, h_qo, qo_len, kv_len]
        scores = torch.matmul(q, k.transpose(-2, -1))
    elif tensor_layout == "NHD":
        # q: [batch_size, qo_len, num_qo_heads, head_dim]
        # k: [batch_size, kv_len, num_kv_heads, head_dim]
        # v: [batch_size, kv_len, num_kv_heads, head_dim]
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        # Compute attention scores: [b, h_qo, qo_len, kv_len]
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    scores = scores * sm_scale

    if is_causal:
        # Create causal mask
        if tensor_layout == "HND":
            causal_mask = torch.triu(
                torch.ones(qo_len, kv_len, device=q.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))
        else:
            causal_mask = torch.triu(
                torch.ones(qo_len, kv_len, device=q.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    if tensor_layout == "HND":
        output = torch.matmul(attn_weights, v)
    else:
        output = torch.matmul(attn_weights, v.transpose(1, 2)).transpose(1, 2)

    return output


def _create_self_attention_tensors(
    batch_size, num_heads, seq_len, head_dim, dtype, tensor_layout
):
    """Create test tensors for self-attention (q, k, v from same source)."""
    if tensor_layout == "HND":
        # [batch_size, num_heads, seq_len, head_dim]
        x = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype
        )
    elif tensor_layout == "NHD":
        # [batch_size, seq_len, num_heads, head_dim]
        x = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype
        )
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # For self-attention, q, k, v are all from the same input
    return x, x, x


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn_qk_int8_pv_fp16_cuda_self_attention(
    batch_size, num_heads, seq_len, head_dim, tensor_layout, is_causal, dtype
):
    """Test sageattn_qk_int8_pv_fp16_cuda for self-attention."""
    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype, tensor_layout
    )

    # Run the kernel implementation
    o_kernel = sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        pv_accum_dtype="fp32",
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
    )

    # Compare results with relaxed tolerance due to int8 quantization
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn_qk_int8_pv_fp16_triton_self_attention(
    batch_size, num_heads, seq_len, head_dim, tensor_layout, is_causal, dtype
):
    """Test sageattn_qk_int8_pv_fp16_triton for self-attention."""
    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype, tensor_layout
    )

    # Run the kernel implementation
    o_kernel = sageattn_qk_int8_pv_fp16_triton(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
    )

    # Compare results with relaxed tolerance due to int8 quantization
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn_qk_int8_pv_fp8_cuda_self_attention(
    batch_size, num_heads, seq_len, head_dim, tensor_layout, is_causal, dtype
):
    """Test sageattn_qk_int8_pv_fp8_cuda for self-attention."""
    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype, tensor_layout
    )

    # Run the kernel implementation
    o_kernel = sageattn_qk_int8_pv_fp8_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        pv_accum_dtype="fp32",
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
    )

    # Compare results with relaxed tolerance due to int8/fp8 quantization
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.15, atol=0.15)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (9, 0),
    reason="sageattn_qk_int8_pv_fp8_cuda_sm90 is only supported on sm90 (Hopper)",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn_qk_int8_pv_fp8_cuda_sm90_self_attention(
    batch_size, num_heads, seq_len, head_dim, tensor_layout, is_causal, dtype
):
    """Test sageattn_qk_int8_pv_fp8_cuda_sm90 for self-attention."""
    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype, tensor_layout
    )

    # Run the kernel implementation
    o_kernel = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        pv_accum_dtype="fp32+fp32",
    )
    torch.cuda.synchronize(q.device)

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
    )

    # Compare results with relaxed tolerance due to int8/fp8 quantization
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.15, atol=0.15)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sageattn_qk_int8_pv_fp16_cuda_with_return_lse(
    batch_size, num_heads, seq_len, head_dim, tensor_layout, dtype
):
    """Test sageattn_qk_int8_pv_fp16_cuda with return_lse=True for self-attention."""
    q, k, v = _create_self_attention_tensors(
        batch_size, num_heads, seq_len, head_dim, dtype, tensor_layout
    )

    # Run the kernel implementation with return_lse
    o_kernel, lse = sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=False,
        pv_accum_dtype="fp32",
        return_lse=True,
    )

    # Run reference implementation
    o_ref = torch_sageattn_self_attention(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=False,
    )

    # Check output shape and LSE shape
    assert o_kernel.shape == o_ref.shape
    if tensor_layout == "HND":
        # LSE shape: [batch_size, num_heads, seq_len]
        assert lse.shape == (batch_size, num_heads, seq_len)
    else:
        # LSE shape: [batch_size, num_heads, seq_len]
        assert lse.shape == (batch_size, num_heads, seq_len)

    # Compare results with relaxed tolerance due to int8 quantization
    torch.testing.assert_close(o_kernel, o_ref, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
