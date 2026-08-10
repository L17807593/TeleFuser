import logging

import torch

from tf_kernel.load_utils import _load_architecture_specific_ops

logger = logging.getLogger(__name__)

# Initialize the ops library based on current GPU
common_ops = _load_architecture_specific_ops()

# Check if FP4 is available (compiled with ENABLE_NVFP4)
try:
    # Try to access FP4 operations to check if they are compiled
    _ = torch.ops.tf_kernel.cutlass_scaled_fp4_mm
    _ = torch.ops.tf_kernel.scaled_fp4_quant
    FP4_AVAILABLE = True
except AttributeError as error:
    FP4_AVAILABLE = False
    logger.debug("FP4 operators are unavailable: %s", error)

from tf_kernel.elementwise import (FusedSetKVBufferArg,
                                   apply_rope_with_cos_sin_cache_inplace,
                                   copy_to_gpu_no_ce, downcast_fp8,
                                   fused_add_rmsnorm, gelu_and_mul,
                                   gelu_tanh_and_mul, gemma_rmsnorm, rmsnorm,
                                   rotary_embedding, silu_and_mul)
from tf_kernel.gemm import (bmm_fp8, fp8_blockwise_scaled_mm, fp8_scaled_mm,
                            int8_scaled_mm, tf_per_tensor_quant_fp8,
                            tf_per_token_group_quant_8bit,
                            tf_per_token_group_quant_fp8,
                            tf_per_token_group_quant_int8,
                            tf_per_token_quant_fp8)

# Import FP4 functions only if available
if FP4_AVAILABLE:
    from tf_kernel.gemm import (
        cutlass_scaled_fp4_mm,
        scaled_fp4_experts_quant,
        scaled_fp4_grouped_quant,
        scaled_fp4_quant,
        silu_and_mul_scaled_fp4_grouped_quant,
    )

from tf_kernel.memory import set_kv_buffer_kernel, weak_ref_tensor
from tf_kernel.ulysses import AsyncUlyssesHandle, CudaIpcUlyssesGroup
from tf_kernel.sageattn2 import (sageattn, sageattn_qk_int8_pv_fp8_cuda,
                                 sageattn_qk_int8_pv_fp8_cuda_sm90,
                                 sageattn_qk_int8_pv_fp16_cuda,
                                 sageattn_qk_int8_pv_fp16_triton,
                                 sageattn_varlen)

# Import SageAttention3 (FP4 Blackwell) if available
if FP4_AVAILABLE:
    from tf_kernel.sageattn3 import (
        blockscaled_fp4_attn,
        sageattn3_blackwell,
        scale_and_quant_fp4,
        scale_and_quant_fp4_permute,
        scale_and_quant_fp4_transpose,
    )

from tf_kernel.block_sparse_attn import (block_sparse_attn_func,
                                         block_streaming_attn_func,
                                         token_streaming_attn_func)

try:
    from importlib.metadata import version
    __version__ = version("tf-kernel")
except ImportError:
    __version__ = "unknown"
