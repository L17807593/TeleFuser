# SPDX-License-Identifier: Apache-2.0
"""In-place BF16 SiLU-gated multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_bf16_input_inplace_kernel(
    input_ptr,
    total_vectors,
    hidden_vectors,
    input_width_vectors,
    VECTOR_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vector_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channels = tl.arange(0, VECTOR_SIZE)
    mask = vector_ids < total_vectors
    row = vector_ids // hidden_vectors
    column_vector = vector_ids - row * hidden_vectors
    gate_offsets = (row * input_width_vectors + column_vector) * VECTOR_SIZE + channels[:, None]
    gate = tl.load(input_ptr + gate_offsets, mask=mask[None, :]).to(tl.float32)
    up = tl.load(input_ptr + gate_offsets + hidden_vectors * VECTOR_SIZE, mask=mask[None, :]).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(input_ptr + gate_offsets, activated * up, mask=mask[None, :])


def silu_and_mul_bf16_input_inplace_(input_tensor: torch.Tensor) -> torch.Tensor:
    """Write rounded ``silu(gate) * up`` into the gate half and return that view."""
    hidden_size = input_tensor.shape[-1] // 2
    output_elements = input_tensor.numel() // 2
    if output_elements:
        vector_size = 8
        hidden_vectors = hidden_size // vector_size
        total_vectors = output_elements // vector_size
        block_size = 256
        _silu_and_mul_bf16_input_inplace_kernel[(triton.cdiv(total_vectors, block_size),)](
            input_tensor,
            total_vectors,
            hidden_vectors,
            input_tensor.shape[-1] // vector_size,
            VECTOR_SIZE=vector_size,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return input_tensor[..., :hidden_size]
