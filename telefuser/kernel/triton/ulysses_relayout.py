# SPDX-License-Identifier: Apache-2.0
"""Destination-major Ulysses input and output relayout kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_destination_major_kernel(
    output_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    total_elements,
    rows,
    local_heads,
    head_dim,
    stride_query_row,
    stride_query_head,
    stride_key_row,
    stride_key_head,
    stride_value_row,
    stride_value_head,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    channel = offsets % head_dim
    head_slot = offsets // head_dim
    local_head = head_slot % local_heads
    row_slot = head_slot // local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * local_heads + local_head

    query = tl.load(
        query_ptr + row * stride_query_row + global_head * stride_query_head + channel,
        mask=mask,
    )
    key = tl.load(
        key_ptr + row * stride_key_row + global_head * stride_key_head + channel,
        mask=mask,
    )
    value = tl.load(
        value_ptr + row * stride_value_row + global_head * stride_value_head + channel,
        mask=mask,
    )
    output_base = head_slot * (3 * head_dim) + channel
    tl.store(output_ptr + output_base, query, mask=mask)
    tl.store(output_ptr + output_base + head_dim, key, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_dim, value, mask=mask)


@triton.jit
def _merge_ulysses_heads_kernel(
    output_ptr,
    input_ptr,
    total_vectors,
    batch,
    sequence,
    world_size: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vector_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channels = tl.arange(0, VECTOR_SIZE)
    mask = vector_ids < total_vectors
    channel_vector = vector_ids % (head_dim // VECTOR_SIZE)
    rest = vector_ids // (head_dim // VECTOR_SIZE)
    local_head = rest % local_heads
    rest //= local_heads
    destination = rest % world_size
    rest //= world_size
    row = rest % sequence
    batch_index = rest // sequence
    source = (
        (((destination * sequence + row) * batch + batch_index) * local_heads + local_head) * head_dim
    ) + channel_vector * VECTOR_SIZE
    values = tl.load(input_ptr + source[:, None] + channels[None, :], mask=mask[:, None])
    tl.store(
        output_ptr + vector_ids[:, None] * VECTOR_SIZE + channels[None, :],
        values,
        mask=mask[:, None],
    )


def pack_qkv_destination_major(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Pack 3D Q/K/V directly as ``[destination, row, local_head, 3 * dim]``."""
    rows, global_heads, head_dim = query.shape
    local_heads = global_heads // world_size
    output = torch.empty(
        world_size,
        rows,
        local_heads,
        3 * head_dim,
        dtype=query.dtype,
        device=query.device,
    )
    total_elements = rows * global_heads * head_dim
    if total_elements == 0:
        return output
    block_size = 1024
    _pack_qkv_destination_major_kernel[(triton.cdiv(total_elements, block_size),)](
        output,
        query,
        key,
        value,
        total_elements,
        rows,
        local_heads,
        head_dim,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def merge_ulysses_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Relayout ``[world, sequence, batch, local_head, dim]`` to destination-last order."""
    world_size, sequence, batch, local_heads, head_dim = tensor.shape
    output = torch.empty(
        batch,
        sequence,
        world_size,
        local_heads,
        head_dim,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if tensor.numel() == 0:
        return output
    vector_size = 8 if head_dim % 8 == 0 else 1
    total_vectors = tensor.numel() // vector_size
    block_size = 256
    _merge_ulysses_heads_kernel[(triton.cdiv(total_vectors, block_size),)](
        output,
        tensor,
        total_vectors,
        batch,
        sequence,
        world_size=world_size,
        local_heads=local_heads,
        head_dim=head_dim,
        VECTOR_SIZE=vector_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output
