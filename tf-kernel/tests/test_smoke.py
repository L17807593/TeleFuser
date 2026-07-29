import pytest
import torch

import tf_kernel

pytestmark = [pytest.mark.gpu, pytest.mark.smoke]


def test_rmsnorm_smoke():
    x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
    weight = torch.ones(1024, device="cuda", dtype=torch.float16)

    output = tf_kernel.rmsnorm(x, weight)
    torch.cuda.synchronize(x.device)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.sm90
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="H100 auto-dispatch smoke test requires SM90",
)
def test_sageattn_auto_dispatch_sm90_is_synchronized():
    q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = tf_kernel.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    torch.cuda.synchronize(q.device)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(output, reference, rtol=0.15, atol=0.15)

@pytest.mark.sm90
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="H100 SM90 LSE smoke test requires SM90",
)
def test_sageattn_sm90_return_lse_is_synchronized():
    q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output, lse = tf_kernel.sageattn_qk_int8_pv_fp8_cuda_sm90(
        q, k, v, tensor_layout="HND", is_causal=False, return_lse=True
    )
    torch.cuda.synchronize(q.device)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert lse.shape == (1, 8, 128)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(output, reference, rtol=0.15, atol=0.15)
