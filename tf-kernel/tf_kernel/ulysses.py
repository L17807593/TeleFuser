"""Single-node Ulysses all-to-all over CUDA IPC and copy engines."""

from __future__ import annotations

import socket
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed as dist

_MAX_CACHED_TARGETS = 12


@dataclass
class _Completion:
    event: torch.cuda.Event | None = None


class AsyncUlyssesHandle:
    """GPU-side completion handle for an asynchronous Ulysses all-to-all."""

    def __init__(self, output: torch.Tensor, completion: _Completion, source: torch.Tensor) -> None:
        self._output = output
        self._completion = completion
        self._source: torch.Tensor | None = source

    def wait(self) -> torch.Tensor:
        if self._completion.event is None:
            raise RuntimeError("a deferred Ulysses handle was waited before a barrier-carrying call")
        torch.cuda.current_stream().wait_event(self._completion.event)
        self._source = None
        return self._output


@dataclass
class _Target:
    output: torch.Tensor
    peer_outputs: list[int]
    remote_pointers: list[int]


class CudaIpcUlyssesGroup:
    """A same-host process group that writes directly into peer target buffers.

    Target buffers are registered collectively and cached by tag, shape, and dtype.
    Transfers run on one high-priority copy-engine stream. Several calls may defer
    their stream-memory handshake so Q/K/V share one handshake without being
    packed into one tensor.
    """

    def __init__(self, process_group: dist.ProcessGroup, device: torch.device | None = None) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before CUDA IPC Ulysses")
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        if self.world_size <= 1:
            raise ValueError("CUDA IPC Ulysses requires at least two ranks")

        self.device = torch.device(device or torch.device("cuda", torch.cuda.current_device()))
        if self.device.type != "cuda":
            raise ValueError(f"CUDA IPC Ulysses requires a CUDA device, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        hosts: list[str | None] = [None] * self.world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=process_group)
        if len(set(hosts)) != 1:
            raise NotImplementedError("CUDA IPC Ulysses only supports process groups on one host")

        _, greatest_priority = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=self.device, priority=greatest_priority)
        self._targets: OrderedDict[tuple[object, ...], _Target] = OrderedDict()
        self._barrier = torch.zeros(self.world_size, dtype=torch.int64, device=self.device)
        self._peer_barriers, self._barrier_remote_pointers = self._open_peer_handles(self._barrier)
        self._barrier_epoch = 0
        self._pending_completion: _Completion | None = None
        self._closed = False

    @staticmethod
    def is_available() -> bool:
        try:
            _ = torch.ops.tf_kernel.cuda_ipc_get_mem_handle
            _ = torch.ops.tf_kernel.ulysses_all_to_all_ce
            _ = torch.ops.tf_kernel.ulysses_stream_barrier
        except AttributeError:
            return False
        return torch.cuda.is_available()

    @property
    def has_pending_group(self) -> bool:
        """Whether deferred transfers are waiting for the final grouped handshake."""
        return self._pending_completion is not None

    def _all_gather_handle(self, handle: torch.Tensor) -> torch.Tensor:
        local = handle.to(device=self.device, non_blocking=False)
        gathered = torch.empty(self.world_size * handle.numel(), dtype=torch.uint8, device=self.device)
        dist.all_gather_into_tensor(gathered, local, group=self.process_group)
        return gathered.reshape(self.world_size, handle.numel()).cpu()

    def _open_peer_handles(self, tensor: torch.Tensor) -> tuple[list[int], list[int]]:
        handles = self._all_gather_handle(torch.ops.tf_kernel.cuda_ipc_get_mem_handle(tensor))
        pointers: list[int] = []
        remote_pointers: list[int] = []
        error: Exception | None = None
        for peer in range(self.world_size):
            if peer == self.rank:
                pointers.append(tensor.data_ptr())
                continue
            try:
                pointer = int(torch.ops.tf_kernel.cuda_ipc_open_mem_handle(handles[peer]))
                pointers.append(pointer)
                remote_pointers.append(pointer)
            except Exception as caught:  # all ranks must reach the status collective
                error = caught
                pointers.append(0)

        status = torch.tensor(0 if error else 1, dtype=torch.int32, device=self.device)
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self.process_group)
        if not status.item():
            for pointer in remote_pointers:
                torch.ops.tf_kernel.cuda_ipc_close_mem_handle(pointer)
            detail = f": {error}" if error is not None else " on another rank"
            raise RuntimeError(f"failed to open CUDA IPC peer memory{detail}")
        return pointers, remote_pointers

    def _output_shape(self, input: torch.Tensor, mode: int) -> tuple[int, int, int, int]:
        batch, sequence, heads, head_dim = input.shape
        if mode == 0:
            if heads % self.world_size:
                raise ValueError(f"head count {heads} is not divisible by world size {self.world_size}")
            return batch, sequence * self.world_size, heads // self.world_size, head_dim
        if mode == 1:
            if sequence % self.world_size:
                raise ValueError(f"sequence length {sequence} is not divisible by world size {self.world_size}")
            return batch, sequence // self.world_size, heads * self.world_size, head_dim
        raise ValueError(f"Ulysses mode must be 0 or 1, got {mode}")

    def _target(self, input: torch.Tensor, mode: int, tag: str) -> _Target:
        output_shape = self._output_shape(input, mode)
        key = (tag, mode, output_shape, input.dtype, input.device)
        target = self._targets.get(key)
        if target is not None:
            self._targets.move_to_end(key)
            return target

        if len(self._targets) >= _MAX_CACHED_TARGETS:
            self._evict_oldest_target()
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        peer_outputs, remote_pointers = self._open_peer_handles(output)
        target = _Target(output=output, peer_outputs=peer_outputs, remote_pointers=remote_pointers)
        self._targets[key] = target
        return target

    @staticmethod
    def _close_target(target: _Target) -> None:
        for pointer in target.remote_pointers:
            torch.ops.tf_kernel.cuda_ipc_close_mem_handle(pointer)

    def _evict_oldest_target(self) -> None:
        # Peer ranks may still be writing while the caller stream consumes an older target.
        # A cache miss is infrequent, so prefer a device-wide synchronization over unsafe reuse.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.process_group)
        _, target = self._targets.popitem(last=False)
        self._close_target(target)

    def all_to_all_single_4d_async(
        self,
        input: torch.Tensor,
        *,
        mode: int,
        tag: str,
        barrier: bool = True,
    ) -> AsyncUlyssesHandle:
        if self._closed:
            raise RuntimeError("CUDA IPC Ulysses group is closed")
        if input.device != self.device:
            raise ValueError(f"input is on {input.device}, expected {self.device}")
        if input.ndim != 4:
            raise ValueError(f"Ulysses input must be 4D, got {input.ndim}D")

        target = self._target(input, mode, tag)
        caller_stream = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(caller_stream)
        self._comm_stream.wait_event(ready)

        completion = self._pending_completion
        if completion is None:
            completion = _Completion()
            self._pending_completion = completion

        with torch.cuda.stream(self._comm_stream):
            for peer in range(self.world_size):
                torch.ops.tf_kernel.ulysses_all_to_all_ce(
                    input, target.peer_outputs[peer], self.rank, self.world_size, mode, peer
                )
        input.record_stream(self._comm_stream)

        if barrier:
            self._barrier_epoch += 1
            with torch.cuda.stream(self._comm_stream):
                torch.ops.tf_kernel.ulysses_stream_barrier(
                    self._peer_barriers,
                    self._barrier,
                    self.rank,
                    self.world_size,
                    self._barrier_epoch,
                )
                completion.event = torch.cuda.Event()
                completion.event.record(self._comm_stream)
                self._pending_completion = None
        return AsyncUlyssesHandle(target.output, completion, input)

    def close(self) -> None:
        if self._closed:
            return
        self._comm_stream.synchronize()
        if dist.is_initialized():
            dist.barrier(group=self.process_group)
        for target in self._targets.values():
            self._close_target(target)
        for pointer in self._barrier_remote_pointers:
            torch.ops.tf_kernel.cuda_ipc_close_mem_handle(pointer)
        self._targets.clear()
        self._closed = True

    def __enter__(self) -> CudaIpcUlyssesGroup:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
