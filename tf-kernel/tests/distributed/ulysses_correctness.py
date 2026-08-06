"""Run with: torchrun --standalone --nproc-per-node=4 tests/distributed/ulysses_correctness.py."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from tf_kernel.ulysses import CudaIpcUlyssesGroup


def _input(rank: int, shape: tuple[int, ...], dtype: torch.dtype, offset: int = 0) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(count, device="cuda", dtype=torch.float32).reshape(shape)
    return (values.remainder(32) + rank * 64 + offset).to(dtype)


def _expected_scatter(
    rank: int,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    offset: int = 0,
) -> torch.Tensor:
    world_size = dist.get_world_size()
    local_heads = shape[2] // world_size
    sources = [_input(peer, shape, dtype, offset) for peer in range(world_size)]
    pieces = [source[:, :, rank * local_heads : (rank + 1) * local_heads] for source in sources]
    return torch.cat(pieces, dim=1)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    shape = (2, 12, 8, 16)

    group = CudaIpcUlyssesGroup(dist.group.WORLD)
    for dtype in (torch.float16, torch.bfloat16):
        input = _input(rank, shape, dtype)
        packed_qkv = torch.empty(
            shape[0],
            shape[1],
            3,
            shape[2],
            shape[3],
            dtype=dtype,
            device="cuda",
        )
        packed_qkv[:, :, 2].copy_(input)
        strided_value = packed_qkv[:, :, 2]
        assert not strided_value.is_contiguous()
        strided = group.all_to_all_single_4d_async(strided_value, mode=0, tag=f"strided-{dtype}").wait()
        torch.testing.assert_close(strided, _expected_scatter(rank, shape, dtype), rtol=0, atol=0)

        scattered = group.all_to_all_single_4d_async(input, mode=0, tag=f"scatter-{dtype}").wait()
        torch.testing.assert_close(scattered, _expected_scatter(rank, shape, dtype), rtol=0, atol=0)

        restored = group.all_to_all_single_4d_async(scattered, mode=1, tag=f"gather-{dtype}").wait()
        torch.testing.assert_close(restored, input, rtol=0, atol=0)

        handles = [
            group.all_to_all_single_4d_async(input + 16, mode=0, tag=f"q-{dtype}", barrier=False),
            group.all_to_all_single_4d_async(input + 32, mode=0, tag=f"k-{dtype}", barrier=False),
            group.all_to_all_single_4d_async(input + 48, mode=0, tag=f"v-{dtype}"),
        ]
        outputs = [handle.wait() for handle in handles]
        for output, offset in zip(outputs, (16, 32, 48), strict=True):
            torch.testing.assert_close(output, _expected_scatter(rank, shape, dtype, offset), rtol=0, atol=0)

        pointers = [output.data_ptr() for output in outputs]
        handles = [
            group.all_to_all_single_4d_async(input + 80, mode=0, tag=f"q-{dtype}", barrier=False),
            group.all_to_all_single_4d_async(input + 96, mode=0, tag=f"k-{dtype}", barrier=False),
            group.all_to_all_single_4d_async(input + 112, mode=0, tag=f"v-{dtype}"),
        ]
        outputs = [handle.wait() for handle in handles]
        assert [output.data_ptr() for output in outputs] == pointers
        for output, offset in zip(outputs, (80, 96, 112), strict=True):
            torch.testing.assert_close(output, _expected_scatter(rank, shape, dtype, offset), rtol=0, atol=0)

    torch.cuda.synchronize()
    dist.barrier()
    group.close()
    if rank == 0:
        print("CUDA IPC Ulysses correctness: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
