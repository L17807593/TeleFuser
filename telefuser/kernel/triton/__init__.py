"""Triton kernels for TeleFuser.

This module provides optimized Triton kernels for:
- Normalization: LayerNorm, RMSNorm, fused add + RMSNorm, tiled RMSNorm
- Position Encoding: Rotary Position Embedding (RoPE)
- Element-wise Operations: Fused scale and shift
- Quantization: FP8 per-token quantization
- Attention: Merge attention states for Ring Attention

Note: All functions in this module require triton to be installed.
"""

from .indexed_modulation import indexed_gate_bf16_, indexed_scale_shift_bf16_
from .merge_attn_states import fused_merge_attn_states
from .norm import (
    fused_add_rms_norm,
    layer_norm_fn,
    norm_infer,
    triton_one_pass_rms_norm,
)
from .qknorm_rope import qknorm_rope_neox_bf16_
from .quant import per_token_dequant_fp8, per_token_quant_fp8
from .rotary import apply_rotary_embedding
from .scale_shift import (
    fused_add_layernorm_scale_shift,
    fused_layernorm_scale_shift,
    fused_layernorm_scale_shift_gate_select01,
    fused_residual_layernorm_scale_shift_gate_select01,
    fused_scale_shift,
    fused_scale_shift_gate_select,
)
from .silu_mul import silu_and_mul_bf16_input_inplace_
from .ulysses_relayout import merge_ulysses_heads, pack_qkv_destination_major

__all__ = [
    "layer_norm_fn",
    "norm_infer",
    "triton_one_pass_rms_norm",
    "fused_add_rms_norm",
    "apply_rotary_embedding",
    "apply_rotary_embedding_inplace",
    "fused_scale_shift",
    "fused_layernorm_scale_shift",
    "fused_add_layernorm_scale_shift",
    "fused_scale_shift_gate_select",
    "fused_layernorm_scale_shift_gate_select01",
    "fused_residual_layernorm_scale_shift_gate_select01",
    "fused_merge_attn_states",
    "indexed_gate_bf16_",
    "indexed_scale_shift_bf16_",
    "qknorm_rope_neox_bf16_",
    "per_token_quant_fp8",
    "per_token_dequant_fp8",
    "pack_qkv_destination_major",
    "merge_ulysses_heads",
    "silu_and_mul_bf16_input_inplace_",
]
