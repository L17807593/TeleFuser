"""Optional model-owned backends for grouped Ulysses scatter."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_MAX_CACHED_TARGETS = 12


@dataclass
class _Completion:
    event: torch.cuda.Event | None = None


class _AsyncUlyssesHandle:
    def __init__(self, output: torch.Tensor, completion: _Completion, source: torch.Tensor) -> None:
        self._output = output
        self._completion = completion
        self._source: torch.Tensor | None = source

    def wait(self) -> torch.Tensor:
        if self._completion.event is None:
            raise RuntimeError("a deferred Ulysses handle was waited before a barrier-carrying call")
        torch.cuda.current_stream(self._output.device).wait_event(self._completion.event)
        self._source = None
        return self._output


@dataclass
class _SymmetricTarget:
    output: torch.Tensor
    handle: Any
    peer_outputs: list[torch.Tensor]


class _SymmetricMemoryUlyssesGroup:
    """PyTorch-owned peer mappings with Copy Engine Ulysses transfers."""

    def __init__(self, process_group: dist.ProcessGroup, device: torch.device) -> None:
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.device = torch.device(device)
        self._symmetric_memory = import_module("torch.distributed._symmetric_memory")
        _, greatest_priority = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=self.device, priority=greatest_priority)
        self._barrier = self._symmetric_memory.empty(self.world_size, dtype=torch.int64, device=self.device)
        self._barrier_handle = self._symmetric_memory.rendezvous(self._barrier, self.process_group)
        self._peer_barriers = [
            self._barrier_handle.get_buffer(peer, self._barrier.shape, self._barrier.dtype).data_ptr()
            for peer in range(self.world_size)
        ]
        self._barrier_epoch = 0
        self._targets: OrderedDict[tuple[object, ...], _SymmetricTarget] = OrderedDict()
        self._pending_completion: _Completion | None = None
        self._closed = False

    @staticmethod
    def is_available(device: torch.device | None = None) -> bool:
        try:
            major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
        except (TypeError, ValueError):
            return False
        if (major, minor) < (2, 11):
            return False
        if not torch.cuda.is_available():
            return False
        try:
            symmetric_memory = import_module("torch.distributed._symmetric_memory")
            if device is not None and symmetric_memory.get_backend(device) != "CUDA":
                return False
            _ = symmetric_memory.empty
            _ = symmetric_memory.rendezvous
            _ = torch.ops.tf_kernel.ulysses_all_to_all_ce
            _ = torch.ops.tf_kernel.ulysses_stream_barrier
        except (AttributeError, ImportError):
            return False
        return True

    @property
    def has_pending_group(self) -> bool:
        return self._pending_completion is not None

    def _output_shape(self, input: torch.Tensor, mode: int) -> tuple[int, int, int, int]:
        batch, sequence, heads, head_dim = input.shape
        if mode != 0:
            raise NotImplementedError("PyTorch Symmetric Memory is currently used only for Ulysses scatter")
        if heads % self.world_size:
            raise ValueError(f"head count {heads} is not divisible by world size {self.world_size}")
        return batch, sequence * self.world_size, heads // self.world_size, head_dim

    def _target(self, input: torch.Tensor, mode: int, tag: str) -> _SymmetricTarget:
        output_shape = self._output_shape(input, mode)
        key = (tag, mode, output_shape, input.dtype, input.device)
        target = self._targets.get(key)
        if target is not None:
            self._targets.move_to_end(key)
            return target

        if len(self._targets) >= _MAX_CACHED_TARGETS:
            self._comm_stream.synchronize()
            dist.barrier(group=self.process_group)
            self._targets.popitem(last=False)
        output = self._symmetric_memory.empty(output_shape, dtype=input.dtype, device=input.device)
        handle = self._symmetric_memory.rendezvous(output, self.process_group)
        peer_outputs = [handle.get_buffer(peer, output_shape, input.dtype) for peer in range(self.world_size)]
        target = _SymmetricTarget(output=output, handle=handle, peer_outputs=peer_outputs)
        self._targets[key] = target
        return target

    def all_to_all_single_4d_async(
        self,
        input: torch.Tensor,
        *,
        mode: int,
        tag: str,
        barrier: bool = True,
    ) -> _AsyncUlyssesHandle:
        if self._closed:
            raise RuntimeError("PyTorch Symmetric Memory Ulysses group is closed")
        if input.device != self.device:
            raise ValueError(f"input is on {input.device}, expected {self.device}")
        if input.ndim != 4:
            raise ValueError(f"Ulysses input must be 4D, got {input.ndim}D")

        target = self._target(input, mode, tag)
        caller_stream = torch.cuda.current_stream(self.device)
        self._comm_stream.wait_stream(caller_stream)
        completion = self._pending_completion
        if completion is None:
            completion = _Completion()
            self._pending_completion = completion

        with torch.cuda.stream(self._comm_stream):
            for peer, peer_output in enumerate(target.peer_outputs):
                torch.ops.tf_kernel.ulysses_all_to_all_ce(
                    input,
                    peer_output.data_ptr(),
                    self.rank,
                    self.world_size,
                    mode,
                    peer,
                )
            if barrier:
                self._barrier_epoch += 1
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
        input.record_stream(self._comm_stream)
        return _AsyncUlyssesHandle(target.output, completion, input)

    def close(self) -> None:
        if self._closed:
            return
        self._comm_stream.synchronize()
        self._targets.clear()
        self._peer_barriers.clear()
        self._barrier_handle = None
        self._pending_completion = None
        self._closed = True


class UlyssesCommunicator:
    """Model-owned dispatcher for optional grouped Ulysses scatter backends."""

    def __init__(self, process_group: dist.ProcessGroup) -> None:
        self.process_group = process_group
        self._backend: Any | None = None
        self._backend_name: str | None = None
        self._disabled_backends: set[str] = set()

    @property
    def has_pending_group(self) -> bool:
        return self._backend is not None and self._backend.has_pending_group

    @property
    def backend_name(self) -> str | None:
        return self._backend_name

    def _create_backend(self, tensor: torch.Tensor) -> Any | None:
        if not tensor.is_cuda or torch.compiler.is_compiling():
            return None

        if "PyTorch Symmetric Memory" not in self._disabled_backends:
            try:
                if _SymmetricMemoryUlyssesGroup.is_available(tensor.device):
                    self._backend = _SymmetricMemoryUlyssesGroup(self.process_group, tensor.device)
                    self._backend_name = "PyTorch Symmetric Memory"
                    logger.info("Using PyTorch Symmetric Memory for grouped Ulysses scatter")
                    return self._backend
                self._disabled_backends.add("PyTorch Symmetric Memory")
            except (ImportError, NotImplementedError, RuntimeError) as error:
                logger.debug("PyTorch Symmetric Memory Ulysses is unavailable: %s", error)
                self._disabled_backends.add("PyTorch Symmetric Memory")

        if "CUDA IPC" not in self._disabled_backends:
            try:
                backend_class = import_module("tf_kernel.ulysses").CudaIpcUlyssesGroup
                if backend_class.is_available():
                    self._backend = backend_class(self.process_group, tensor.device)
                    self._backend_name = "CUDA IPC"
                    logger.info("Using CUDA IPC for grouped Ulysses scatter")
                    return self._backend
                self._disabled_backends.add("CUDA IPC")
            except (ImportError, NotImplementedError, RuntimeError) as error:
                logger.debug("CUDA IPC Ulysses is unavailable: %s", error)
                self._disabled_backends.add("CUDA IPC")
        return None

    def submit(
        self,
        tensor: torch.Tensor,
        *,
        tag: str,
        barrier: bool,
    ) -> Callable[[], torch.Tensor] | None:
        while True:
            backend = self._backend or self._create_backend(tensor)
            if backend is None:
                return None
            group_was_pending = backend.has_pending_group
            try:
                handle = backend.all_to_all_single_4d_async(tensor, mode=0, tag=tag, barrier=barrier)
                return handle.wait
            except RuntimeError as error:
                if group_was_pending or backend.has_pending_group:
                    raise RuntimeError(
                        f"{self._backend_name} Ulysses failed after a grouped transfer started; "
                        "the request cannot safely fall back"
                    ) from error
                failed_name = self._backend_name
                backend.close()
                self._backend = None
                self._backend_name = None
                if failed_name is not None:
                    self._disabled_backends.add(failed_name)
                    logger.warning("%s Ulysses failed; trying the next backend: %s", failed_name, error)

    def close(self) -> None:
        """Release backend resources without depending on process-group teardown."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None
            self._backend_name = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
