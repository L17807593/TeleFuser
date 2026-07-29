"""H100 integration coverage for tf-kernel through the public attention op."""

import pytest
import torch
import torch.nn.functional as F

try:
    import tf_kernel
except ImportError as error:
    pytest.skip(f"tf-kernel is unavailable: {error}", allow_module_level=True)

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.ops.attention.attention_impl import attention
from telefuser.ops.attention.backends import sageattention


@pytest.mark.gpu
def test_tf_kernel_sm90_through_public_attention_op() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 GPU required")

    torch.manual_seed(42)
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90, is_causal=True)

    output, lse = attention(q, k, v, attention_config=config, return_lse=True)
    torch.cuda.synchronize(q.device)

    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)

    assert sageattention is tf_kernel
    assert output.shape == q.shape
    assert lse.shape == (q.shape[0], q.shape[2], q.shape[1])
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(output, reference, atol=0.08, rtol=0.08)
