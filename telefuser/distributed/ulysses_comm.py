"""Ulysses All-to-All communication primitives for sequence parallelism.

Ulysses requires an equal head partition: ``num_heads`` must be divisible by
the process-group world size.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc

from telefuser.distributed.ulysses_backend import UlyssesCommunicator

logger = logging.getLogger(__name__)


def _get_distributed_info(process_group: dist.ProcessGroup) -> tuple[int, int]:
    """Return the process-group rank and world size."""
    return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)


def _local_head_count(num_heads: int, world_size: int) -> int:
    """Return the equal Ulysses head partition or reject an invalid topology."""
    if num_heads % world_size:
        raise ValueError(
            f"Ulysses sequence parallelism requires num_heads ({num_heads}) to be divisible "
            f"by world_size ({world_size})"
        )
    return num_heads // world_size


def _wait_async_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Resolve an asynchronous functional collective tensor."""
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()
    return tensor


def ulysses_scatter_heads(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    async_comm: bool = True,
    tag: str | None = None,
    barrier: bool = True,
    communicator: UlyssesCommunicator | None = None,
) -> Callable[[], torch.Tensor]:
    """Scatter global heads and gather sequence across Ulysses ranks.

    A tagged call with ``barrier=False`` starts a same-host Copy Engine group.
    Tagged calls share completion until the final ``barrier=True`` call; standalone
    calls remain on NCCL.
    """
    _, world_size = _get_distributed_info(process_group)
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    local_heads = _local_head_count(num_heads, world_size)

    use_optimized_backend = (
        async_comm and tag is not None and communicator is not None and (not barrier or communicator.has_pending_group)
    )
    if use_optimized_backend:
        wait = communicator.submit(tensor, tag=tag, barrier=barrier)
        if wait is not None:
            return wait

    tensor = tensor.reshape(batch, local_seq_len, world_size, local_heads, head_dim)
    tensor = tensor.permute(2, 1, 0, 3, 4).contiguous()
    comm_buffer_shape = tensor.shape

    if async_comm:
        submitted = fc.all_to_all_single(tensor.flatten(), None, None, process_group)

        def wait() -> torch.Tensor:
            result = _wait_async_tensor(submitted).reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 0, 2, 3)

    else:
        submitted = tensor.flatten()
        output = torch.empty_like(submitted)
        dist.all_to_all_single(output, submitted, None, None, group=process_group, async_op=False)

        def wait() -> torch.Tensor:
            result = output.reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 0, 2, 3)

    return wait


def ulysses_gather_heads(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    num_heads: int,
    async_comm: bool = True,
) -> Callable[[], torch.Tensor]:
    """Gather global heads and scatter sequence across Ulysses ranks."""
    _, world_size = _get_distributed_info(process_group)
    batch, global_seq_len, local_heads, head_dim = tensor.shape
    if global_seq_len % world_size:
        raise ValueError(f"Ulysses sequence length ({global_seq_len}) must be divisible by world_size ({world_size})")
    expected_local_heads = _local_head_count(num_heads, world_size)
    if local_heads != expected_local_heads:
        raise ValueError(f"Ulysses local head count must be {expected_local_heads}, got {local_heads}")

    if _can_use_destination_major_kernel(tensor):
        return _gather_heads_destination_major(
            tensor,
            process_group,
            world_size=world_size,
            num_heads=num_heads,
            async_comm=async_comm,
        )

    local_seq_len = global_seq_len // world_size

    tensor = tensor.reshape(batch, world_size, local_seq_len, local_heads, head_dim)
    tensor = tensor.permute(1, 3, 0, 2, 4).contiguous()
    comm_buffer_shape = tensor.shape

    if async_comm:
        submitted = fc.all_to_all_single(tensor.flatten(), None, None, process_group)

        def wait() -> torch.Tensor:
            result = _wait_async_tensor(submitted).reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 2, 0, 3)

    else:
        submitted = tensor.flatten()
        output = torch.empty_like(submitted)
        dist.all_to_all_single(output, submitted, None, None, group=process_group, async_op=False)

        def wait() -> torch.Tensor:
            result = output.reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 2, 0, 3)

    return wait


def _can_use_destination_major_kernel(*tensors: torch.Tensor) -> bool:
    return (
        bool(tensors)
        and all(tensor.is_cuda for tensor in tensors)
        and all(tensor.dtype in (torch.float16, torch.bfloat16) for tensor in tensors)
        and all(tensor.dtype == tensors[0].dtype for tensor in tensors)
        and all(tensor.stride(-1) == 1 for tensor in tensors)
        and not torch.compiler.is_compiling()
    )


def ulysses_scatter_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: dist.ProcessGroup,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Scatter Q/K/V with one destination-major collective and no intermediate concat."""
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("Ulysses Q/K/V scatter requires matching 4D tensors")
    _, world_size = _get_distributed_info(process_group)
    batch, local_seq_len, num_heads, head_dim = query.shape
    local_heads = _local_head_count(num_heads, world_size)
    if batch != 1 or not _can_use_destination_major_kernel(query, key, value):
        combined_wait = ulysses_scatter_heads(torch.cat((query, key, value), dim=-1), process_group, tag="qkv")

        def fallback_wait() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return combined_wait().chunk(3, dim=-1)

        return fallback_wait

    from telefuser.kernel.triton.ulysses_relayout import pack_qkv_destination_major

    packed = pack_qkv_destination_major(query[0], key[0], value[0], world_size)
    output = torch.empty_like(packed.flatten())
    dist.all_to_all_single(output, packed.flatten(), group=process_group)

    def wait() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = output.reshape(world_size * local_seq_len, local_heads, 3 * head_dim).unsqueeze(0)
        return result.chunk(3, dim=-1)

    return wait


def ulysses_gather_heads_destination_major(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    num_heads: int,
) -> Callable[[], torch.Tensor]:
    """Gather heads using sequence-major NCCL input and one fused output relayout."""
    if tensor.ndim != 4:
        raise ValueError("Ulysses head gather requires a 4D tensor")
    _, world_size = _get_distributed_info(process_group)
    batch, global_seq_len, local_heads, head_dim = tensor.shape
    if global_seq_len % world_size:
        raise ValueError(f"Ulysses sequence length ({global_seq_len}) must be divisible by world_size ({world_size})")
    expected_local_heads = _local_head_count(num_heads, world_size)
    if local_heads != expected_local_heads:
        raise ValueError(f"Ulysses local head count must be {expected_local_heads}, got {local_heads}")
    if not _can_use_destination_major_kernel(tensor):
        return ulysses_gather_heads(tensor, process_group, num_heads=num_heads)

    return _gather_heads_destination_major(
        tensor,
        process_group,
        world_size=world_size,
        num_heads=num_heads,
        async_comm=True,
    )


def _gather_heads_destination_major(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    world_size: int,
    num_heads: int,
    async_comm: bool,
) -> Callable[[], torch.Tensor]:
    """Submit sequence-major All-to-All and fuse the received head relayout."""

    from telefuser.kernel.triton.ulysses_relayout import merge_ulysses_heads

    batch, global_seq_len, local_heads, head_dim = tensor.shape
    local_seq_len = global_seq_len // world_size
    packed = tensor.permute(1, 0, 2, 3).contiguous()
    output = torch.empty_like(packed.flatten())
    work = dist.all_to_all_single(
        output,
        packed.flatten(),
        group=process_group,
        async_op=async_comm,
    )

    def wait() -> torch.Tensor:
        if work is not None:
            work.wait()
        received = output.reshape(world_size, local_seq_len, batch, local_heads, head_dim)
        return merge_ulysses_heads(received).flatten(2, 3).reshape(batch, local_seq_len, num_heads, head_dim)

    return wait
