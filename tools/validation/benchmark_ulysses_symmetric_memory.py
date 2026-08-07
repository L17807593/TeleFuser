# SPDX-License-Identifier: Apache-2.0
"""Compare PyTorch Symmetric Memory and CUDA IPC Ulysses scatter paths.

Run from the repository root with a source-built SM90 ``tf-kernel`` wheel::

    PYTHONPATH=/tmp/tf-kernel-ulysses torchrun --standalone --nproc-per-node=4 \
        tools/validation/benchmark_ulysses_symmetric_memory.py \
        --group-size 2 --profile minimax_h3_tp2_u2 \
        --output /tmp/ulysses-symm-u2.json

The benchmark reports collective-only and complete scatter timings separately.
Complete scatter timings include any layout packing required by the backend.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclass(frozen=True)
class Profile:
    batch: int
    local_sequence: int
    local_heads: int
    head_dim: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.local_sequence, self.local_heads, self.head_dim


PROFILES = {
    # MiniMax H3 production dimensions: 56 global heads and 18,944 global tokens.
    "minimax_h3_tp2_u2": Profile(1, 9472, 28, 128),
    "minimax_h3_u4": Profile(1, 4736, 56, 128),
    # One 832x480 latent-frame token slab with the production 5,120-wide DiT.
    "lingbot_world_v2_u4": Profile(1, 1560, 40, 128),
}


def _nccl_options() -> dist.ProcessGroupNCCL.Options:
    options = dist.ProcessGroupNCCL.Options()
    options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    return options


def _create_benchmark_group(group_size: int) -> dist.ProcessGroup:
    world_size = dist.get_world_size()
    if world_size % group_size:
        raise ValueError(f"world size {world_size} is not divisible by group size {group_size}")
    if group_size == world_size:
        return dist.group.WORLD

    rank = dist.get_rank()
    selected: dist.ProcessGroup | None = None
    for first_rank in range(0, world_size, group_size):
        ranks = list(range(first_rank, first_rank + group_size))
        group = dist.new_group(ranks, pg_options=_nccl_options())
        if rank in ranks:
            selected = group
    if selected is None:
        raise RuntimeError(f"rank {rank} was not assigned to a benchmark group")
    return selected


def _make_input(profile: Profile, rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    elements = profile.batch * profile.local_sequence * profile.local_heads * profile.head_dim
    values = torch.arange(elements, device=device, dtype=torch.int64).reshape(profile.shape)
    contiguous = (values.remainder(251) + rank * 257).to(dtype)
    fused_qkv = torch.empty(
        profile.batch,
        profile.local_sequence,
        3,
        profile.local_heads,
        profile.head_dim,
        dtype=dtype,
        device=device,
    )
    fused_qkv[:, :, 2].copy_(contiguous)
    value = fused_qkv[:, :, 2]
    if value.is_contiguous():
        raise RuntimeError("the validation input must retain the fused-QKV stride")
    return value


def _expected_scatter(input: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_heads = input.shape[2] // world_size
    reference_input = input.contiguous()
    sources = [torch.empty_like(reference_input) for _ in range(world_size)]
    dist.all_gather(sources, reference_input, group=group)
    return torch.cat(
        [source[:, :, rank * local_heads : (rank + 1) * local_heads] for source in sources],
        dim=1,
    ).contiguous()


def _output_shape(input: torch.Tensor, world_size: int) -> tuple[int, int, int, int]:
    batch, sequence, heads, head_dim = input.shape
    if heads % world_size:
        raise ValueError(f"head count {heads} is not divisible by group size {world_size}")
    return batch, sequence * world_size, heads // world_size, head_dim


def _pack_heads(input: torch.Tensor, output: torch.Tensor, world_size: int) -> None:
    batch, sequence, heads, head_dim = input.shape
    local_heads = heads // world_size
    source = input.reshape(batch, sequence, world_size, local_heads, head_dim).permute(2, 1, 0, 3, 4)
    output.reshape(world_size, sequence, batch, local_heads, head_dim).copy_(source)


def _received_view(output: torch.Tensor, input: torch.Tensor, world_size: int) -> torch.Tensor:
    batch, sequence, heads, head_dim = input.shape
    local_heads = heads // world_size
    return output.reshape(world_size, sequence, batch, local_heads, head_dim).flatten(0, 1).permute(1, 0, 2, 3)


class NcclScatter:
    def __init__(
        self,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        *,
        symmetric: bool,
    ) -> None:
        self.input = input
        self.group = group
        self.world_size = dist.get_world_size(group)
        elements = input.numel()
        allocator = symm_mem.empty if symmetric else torch.empty
        self.send = allocator(elements, dtype=input.dtype, device=input.device)
        self.output = allocator(elements, dtype=input.dtype, device=input.device)
        self._handles: list[Any] = []
        if symmetric:
            self._handles.append(symm_mem.rendezvous(self.send, group))
            self._handles.append(symm_mem.rendezvous(self.output, group))
        _pack_heads(input, self.send, self.world_size)

    def collective_only(self) -> torch.Tensor:
        work = dist.all_to_all_single(self.output, self.send, group=self.group, async_op=True)
        work.wait()
        return _received_view(self.output, self.input, self.world_size)

    def full_scatter(self) -> torch.Tensor:
        _pack_heads(self.input, self.send, self.world_size)
        return self.collective_only()


class SymmetricRemoteScatter:
    def __init__(
        self,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        *,
        use_copy_engine_primitive: bool,
        use_handle_barrier: bool,
    ) -> None:
        self.input = input
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.output = symm_mem.empty(
            _output_shape(input, self.world_size),
            dtype=input.dtype,
            device=input.device,
        )
        self.handle = symm_mem.rendezvous(self.output, group)
        self.peer_outputs = [
            self.handle.get_buffer(peer, self.output.shape, self.output.dtype) for peer in range(self.world_size)
        ]
        _, greatest_priority = torch.cuda.Stream.priority_range()
        self.stream = torch.cuda.Stream(device=input.device, priority=greatest_priority)
        self.use_copy_engine_primitive = use_copy_engine_primitive
        self.use_handle_barrier = use_handle_barrier

    def full_scatter(self) -> torch.Tensor:
        caller_stream = torch.cuda.current_stream(self.input.device)
        self.stream.wait_stream(caller_stream)
        local_heads = self.input.shape[2] // self.world_size
        local_sequence = self.input.shape[1]
        with torch.cuda.stream(self.stream):
            for peer, peer_output in enumerate(self.peer_outputs):
                if self.use_copy_engine_primitive:
                    torch.ops.tf_kernel.ulysses_all_to_all_ce(
                        self.input,
                        peer_output.data_ptr(),
                        self.rank,
                        self.world_size,
                        0,
                        peer,
                    )
                else:
                    source = self.input[:, :, peer * local_heads : (peer + 1) * local_heads]
                    target = peer_output[:, self.rank * local_sequence : (self.rank + 1) * local_sequence]
                    target.copy_(source)
            if self.use_handle_barrier:
                self.handle.barrier(channel=0)
            else:
                work = dist.barrier(group=self.group, async_op=True)
                work.wait()
        self.input.record_stream(self.stream)
        caller_stream.wait_stream(self.stream)
        return self.output


class CudaIpcScatter:
    def __init__(self, input: torch.Tensor, group: dist.ProcessGroup) -> None:
        from tf_kernel.ulysses import CudaIpcUlyssesGroup

        self.input = input
        self.backend = CudaIpcUlyssesGroup(group, input.device)

    def full_scatter(self) -> torch.Tensor:
        return self.backend.all_to_all_single_4d_async(self.input, mode=0, tag="benchmark").wait()

    def close(self) -> None:
        self.backend.close()


class ModelOwnedScatter:
    def __init__(self, input: torch.Tensor, group: dist.ProcessGroup) -> None:
        from telefuser.distributed.ulysses_backend import UlyssesCommunicator

        self.input = input
        self.communicator = UlyssesCommunicator(group)

    def full_scatter(self) -> torch.Tensor:
        wait = self.communicator.submit(self.input, tag="benchmark", barrier=True)
        if wait is None:
            raise RuntimeError("no optimized model-owned Ulysses backend is available")
        return wait()

    def close(self) -> None:
        self.communicator.close()


class MiniMaxH3PreAttention:
    """Hold model-owned inputs while switching only the Ulysses scatter backend."""

    def __init__(
        self,
        profile: Profile,
        group: dist.ProcessGroup,
        cuda_ipc_backend: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        from telefuser.core.config import AttentionConfig, AttnImplType
        from telefuser.models.minimax_h3_dit import MiniMaxH3Attention, MiniMaxH3DiTConfig

        config = MiniMaxH3DiTConfig(
            hidden_size=5376,
            num_layers=1,
            token_refiner_num_layers=1,
            num_attention_heads=profile.local_heads,
            attention_head_dim=profile.head_dim,
            ffn_hidden_size=64,
            latents_dim=2,
            audio_latents_dim=2,
            patch_size=(1, 2, 2),
            text_dim=16,
            timestep_input_dim=8,
            time_embed_hidden_size=32,
            time_embed_dim=16,
            rope_inv_freq_len=16,
        )
        torch.manual_seed(17)
        self.module = MiniMaxH3Attention(config).eval().to(device=device, dtype=dtype)
        self.module.set_ulysses_group(group)
        torch.manual_seed(31 + dist.get_rank(group))
        self.hidden = torch.randn(profile.local_sequence, config.hidden_size, device=device, dtype=dtype)
        angles = torch.randn(profile.local_sequence, config.rope_inv_freq_len, device=device, dtype=torch.float32)
        self.rope = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
        self.sequence_lengths = [profile.local_sequence * dist.get_world_size(group)]
        self.attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        self.group = group
        self.cuda_ipc_backend = cuda_ipc_backend
        from telefuser.distributed.ulysses_backend import UlyssesCommunicator

        self.symmetric_communicator = UlyssesCommunicator(group)
        self.cuda_ipc_communicator = UlyssesCommunicator(group)
        self.cuda_ipc_communicator._backend = cuda_ipc_backend
        self.cuda_ipc_communicator._backend_name = "CUDA IPC"

    def _run(self, communicator: Any | None) -> torch.Tensor:
        from telefuser.models import minimax_h3_dit

        self.module.set_ulysses_group(self.group, communicator)
        original_attention = minimax_h3_dit.attention
        minimax_h3_dit.attention = lambda query, *_args, **_kwargs: query
        try:
            return self.module(
                self.hidden,
                sequence_lengths=self.sequence_lengths,
                rope_cos_sin_cache=self.rope,
                attention_config=self.attention_config,
            )
        finally:
            minimax_h3_dit.attention = original_attention

    def pytorch_nccl(self) -> torch.Tensor:
        return self._run(None)

    def symmetric_memory(self) -> torch.Tensor:
        return self._run(self.symmetric_communicator)

    def cuda_ipc(self) -> torch.Tensor:
        return self._run(self.cuda_ipc_communicator)

    def close(self) -> None:
        self.module.set_ulysses_group(None)
        self.symmetric_communicator.close()
        self.cuda_ipc_communicator.close()


def _maximum_group_value(value: float, group: dist.ProcessGroup, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _benchmark(
    function: Callable[[], torch.Tensor],
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        milliseconds = start.elapsed_time(end) / iterations
        samples.append(_maximum_group_value(milliseconds, group, device))
        dist.barrier(group=group)

    return {
        "min_us": min(samples) * 1000,
        "median_us": statistics.median(samples) * 1000,
        "max_us": max(samples) * 1000,
    }


def _check(name: str, function: Callable[[], torch.Tensor], expected: torch.Tensor, device: torch.device) -> None:
    actual = function()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda message: f"{name}: {message}")


def _profile_paths(
    paths: dict[str, Callable[[], torch.Tensor]],
    output: Path,
    rank: int,
    device: torch.device,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        ) as profiler:
            for name, function in paths.items():
                with torch.profiler.record_function(name):
                    function()
                    torch.cuda.synchronize(device)
        profiler.export_chrome_trace(str(output / "ulysses_symmetric_memory_rank0.json.gz"))
    else:
        for function in paths.values():
            function()
            torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--group-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--symmetric-backend", choices=("NCCL", "CUDA"), default="NCCL")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model-level", action="store_true")
    parser.add_argument("--model-warmup", type=int, default=3)
    parser.add_argument("--model-iterations", type=int, default=10)
    parser.add_argument("--model-repeats", type=int, default=5)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", pg_options=_nccl_options(), device_id=device)
    rank = dist.get_rank()
    group = _create_benchmark_group(args.group_size)
    group_rank = dist.get_rank(group)
    dtype = getattr(torch, args.dtype)
    profile = PROFILES[args.profile]
    if profile.local_heads % args.group_size:
        raise ValueError(f"profile {args.profile} is incompatible with group size {args.group_size}")

    if args.symmetric_backend == "NCCL":
        symm_mem.set_backend("NCCL")
    elif symm_mem.get_backend(device) != "CUDA":
        raise RuntimeError("the PyTorch build does not provide the default CUDA Symmetric Memory backend")
    input = _make_input(profile, group_rank, dtype, device)
    expected = _expected_scatter(input, group)

    setup_started = time.perf_counter()
    nccl = NcclScatter(input, group, symmetric=False)
    symmetric_nccl = NcclScatter(input, group, symmetric=True)
    use_handle_barrier = args.symmetric_backend == "CUDA"
    symmetric_remote = SymmetricRemoteScatter(
        input,
        group,
        use_copy_engine_primitive=False,
        use_handle_barrier=use_handle_barrier,
    )
    symmetric_hybrid = SymmetricRemoteScatter(
        input,
        group,
        use_copy_engine_primitive=True,
        use_handle_barrier=use_handle_barrier,
    )
    cuda_ipc = CudaIpcScatter(input, group)
    model_owned = ModelOwnedScatter(input, group)
    model_benchmark = None
    if args.model_level:
        if not args.profile.startswith("minimax_h3"):
            raise ValueError("--model-level currently supports only MiniMax H3 profiles")
        model_benchmark = MiniMaxH3PreAttention(profile, group, cuda_ipc.backend, dtype, device)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started

    symmetric_collective = (
        "pytorch_symm_nccl_zero_cta" if args.symmetric_backend == "NCCL" else "pytorch_symm_cuda_nccl"
    )
    paths = {
        "pytorch_nccl_collective": nccl.collective_only,
        "pytorch_nccl_full_scatter": nccl.full_scatter,
        f"{symmetric_collective}_collective": symmetric_nccl.collective_only,
        f"{symmetric_collective}_full_scatter": symmetric_nccl.full_scatter,
        "pytorch_symm_remote_copy_full_scatter": symmetric_remote.full_scatter,
        "pytorch_symm_ce_hybrid_full_scatter": symmetric_hybrid.full_scatter,
        "cuda_ipc_ce_full_scatter": cuda_ipc.full_scatter,
        "telefuser_model_owned_full_scatter": model_owned.full_scatter,
    }

    try:
        for name, function in paths.items():
            _check(name, function, expected, device)
        dist.barrier(group=group)

        results = {
            name: _benchmark(
                function,
                group,
                device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            for name, function in paths.items()
        }

        model_results = None
        if model_benchmark is not None:
            model_reference = model_benchmark.pytorch_nccl().clone()
            _check("minimax_h3_symmetric_memory", model_benchmark.symmetric_memory, model_reference, device)
            _check("minimax_h3_cuda_ipc", model_benchmark.cuda_ipc, model_reference, device)
            model_results = {
                "pytorch_symmetric_memory_v_first": _benchmark(
                    model_benchmark.symmetric_memory,
                    group,
                    device,
                    warmup=args.model_warmup,
                    iterations=args.model_iterations,
                    repeats=args.model_repeats,
                ),
                "pytorch_nccl_v_first": _benchmark(
                    model_benchmark.pytorch_nccl,
                    group,
                    device,
                    warmup=args.model_warmup,
                    iterations=args.model_iterations,
                    repeats=args.model_repeats,
                ),
                "cuda_ipc_v_first": _benchmark(
                    model_benchmark.cuda_ipc,
                    group,
                    device,
                    warmup=args.model_warmup,
                    iterations=args.model_iterations,
                    repeats=args.model_repeats,
                ),
            }

        if args.trace_dir is not None:
            dist.barrier(group=group)
            _profile_paths(paths, args.trace_dir, rank, device)
            dist.barrier(group=group)

        if rank == 0:
            logical_bytes = input.numel() * input.element_size()
            payload = {
                "environment": {
                    "hostname": platform.node(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": dist.get_world_size(),
                    "group_size": args.group_size,
                    "symmetric_memory_backend": symm_mem.get_backend(device),
                    "telefuser_backend": model_owned.communicator.backend_name,
                    "telefuser_synchronization": "stream-memory barrier",
                    "symmetric_remote_synchronization": (
                        "handle.barrier" if use_handle_barrier else "ProcessGroupNCCL barrier"
                    ),
                    "cta_policy": dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO,
                },
                "profile": args.profile,
                "shape": asdict(profile),
                "dtype": args.dtype,
                "logical_bytes_per_rank": logical_bytes,
                "input_stride": list(input.stride()),
                "setup_seconds": setup_seconds,
                "measurement": {
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "repeats": args.repeats,
                    "aggregation": "maximum rank latency per repeat",
                },
                "correctness": "exact",
                "results": results,
                "model_results": model_results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, sort_keys=True))
    finally:
        if model_benchmark is not None:
            model_benchmark.close()
        model_owned.close()
        cuda_ipc.close()
        del symmetric_hybrid, symmetric_remote, symmetric_nccl, nccl
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
