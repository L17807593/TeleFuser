# SPDX-License-Identifier: Apache-2.0
"""Fused in-place Q/K RMSNorm and partial NeoX RoPE."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16_to_fp32(value):
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _qknorm_rope_neox_bf16_kernel(
    query_ptr,
    key_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cache_ptr,
    heads,
    head_dim,
    rope_dim,
    eps,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_cache_row,
    BLOCK_N: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // heads
    head = program - row * heads
    columns = tl.arange(0, BLOCK_N)
    mask = columns < head_dim
    q_base = query_ptr + row * stride_q_row + head * stride_q_head
    k_base = key_ptr + row * stride_k_row + head * stride_k_head
    query = tl.load(q_base + columns, mask=mask, other=0.0).to(tl.float32)
    key = tl.load(k_base + columns, mask=mask, other=0.0).to(tl.float32)
    q_weight = tl.load(q_weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    q_inv_rms = tl.rsqrt(tl.sum(query * query, axis=0) / head_dim + eps)
    k_inv_rms = tl.rsqrt(tl.sum(key * key, axis=0) / head_dim + eps)
    query = _round_bf16_to_fp32(query * q_inv_rms * q_weight)
    key = _round_bf16_to_fp32(key * k_inv_rms * k_weight)

    rope_half = rope_dim // 2
    rotary_mask = columns < rope_dim
    partner_columns = tl.where(columns < rope_half, columns + rope_half, columns - rope_half)
    q_partner = tl.load(q_base + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    k_partner = tl.load(k_base + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    q_partner_weight = tl.load(q_weight_ptr + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    k_partner_weight = tl.load(k_weight_ptr + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    q_partner = _round_bf16_to_fp32(q_partner * q_inv_rms * q_partner_weight)
    k_partner = _round_bf16_to_fp32(k_partner * k_inv_rms * k_partner_weight)
    frequency_column = columns % rope_half
    cosine = tl.load(cache_ptr + row * stride_cache_row + frequency_column, mask=rotary_mask, other=1.0).to(tl.float32)
    sine = tl.load(cache_ptr + row * stride_cache_row + rope_half + frequency_column, mask=rotary_mask, other=0.0).to(
        tl.float32
    )
    q_rotated = tl.where(
        columns < rope_half,
        query * cosine - q_partner * sine,
        query * cosine + q_partner * sine,
    )
    k_rotated = tl.where(
        columns < rope_half,
        key * cosine - k_partner * sine,
        key * cosine + k_partner * sine,
    )
    tl.store(q_base + columns, tl.where(rotary_mask, q_rotated, query), mask=mask)
    tl.store(k_base + columns, tl.where(rotary_mask, k_rotated, key), mask=mask)


def qknorm_rope_neox_bf16_(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate head-contiguous BF16 Q/K tensors in place."""
    rows, heads, head_dim = query.shape
    rope_dim = cos_sin_cache.shape[-1]
    block_n = triton.next_power_of_2(head_dim)
    _qknorm_rope_neox_bf16_kernel[(rows * heads,)](
        query,
        key,
        q_weight,
        k_weight,
        cos_sin_cache,
        heads,
        head_dim,
        rope_dim,
        eps,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        cos_sin_cache.stride(0),
        BLOCK_N=block_n,
        num_warps=1,
    )
    return query, key
