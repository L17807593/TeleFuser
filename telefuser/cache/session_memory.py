"""Hardware-aware retained-session memory planning and slot management."""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch

GIB = 1024**3
DEFAULT_UNTRACKED_SESSION_BYTES = 0


@dataclass(frozen=True)
class SessionCapacityPolicy:
    """Safety policy applied when sizing retained-session pools."""

    safety_margin_percent: int = 5
    minimum_safety_margin_bytes: int = 2 * GIB
    max_capacity: int = 64

    def __post_init__(self) -> None:
        if not 0 <= self.safety_margin_percent < 100:
            raise ValueError("safety_margin_percent must be in [0, 100)")
        if self.minimum_safety_margin_bytes < 0:
            raise ValueError("minimum_safety_margin_bytes must be non-negative")
        if self.max_capacity < 1:
            raise ValueError("max_capacity must be positive")


DEFAULT_SESSION_CAPACITY_POLICY = SessionCapacityPolicy()


@dataclass(frozen=True)
class SessionMemoryBudget:
    """Per-process retained and warmup-replacement bytes for one stage role."""

    persistent_bytes_per_session: int
    replaced_warmup_transient_bytes: int = 0

    def __post_init__(self) -> None:
        if self.persistent_bytes_per_session < 0:
            raise ValueError("persistent_bytes_per_session must be non-negative")
        if self.replaced_warmup_transient_bytes < 0:
            raise ValueError("replaced_warmup_transient_bytes must be non-negative")


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """Raw allocator and hardware facts captured inside a CUDA stage worker."""

    rank: int
    process_id: int
    device_index: int
    device_name: str
    total_bytes: int
    free_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def capture_device_memory_snapshot(device: torch.device) -> DeviceMemorySnapshot:
    """Capture raw facts for one CUDA allocator and its globally visible device."""
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return DeviceMemorySnapshot(
        rank=rank,
        process_id=os.getpid(),
        device_index=device_index,
        device_name=properties.name,
        total_bytes=int(total_bytes),
        free_bytes=int(free_bytes),
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


@dataclass(frozen=True)
class DeviceSessionCapacity:
    """Capacity facts for one physical device visible to a pipeline."""

    device_index: int
    device_name: str
    total_bytes: int
    free_bytes: int
    reusable_reserved_bytes: int
    raw_transient_headroom_bytes: int
    transient_headroom_bytes: int
    safety_margin_bytes: int
    persistent_bytes_per_session: int
    persistent_bytes_by_role: dict[str, int]
    untracked_bytes_per_session: int
    computed_capacity: int


@dataclass(frozen=True)
class SessionCapacityPlan:
    """Result of combining hardware facts with role-specific memory budgets."""

    computed_capacity: int
    effective_capacity: int
    configured_limit: int | None
    limiting_device: int | None
    devices: tuple[DeviceSessionCapacity, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "computed_capacity": self.computed_capacity,
            "effective_capacity": self.effective_capacity,
            "configured_limit": self.configured_limit,
            "limiting_device": self.limiting_device,
            "devices": [asdict(device) for device in self.devices],
        }


def _transient_bytes(
    snapshot: Mapping[str, int | str],
    budget: SessionMemoryBudget | None,
) -> tuple[int, int]:
    raw_bytes = max(
        int(snapshot["peak_allocated_bytes"]) - int(snapshot["allocated_bytes"]),
        0,
    )
    replaced_bytes = budget.replaced_warmup_transient_bytes if budget is not None else 0
    return raw_bytes, max(raw_bytes - replaced_bytes, 0)


def calculate_session_capacity(
    snapshots: Sequence[Mapping[str, int | str]],
    *,
    role_budgets: Mapping[str, SessionMemoryBudget],
    configured_limit: int | None,
    untracked_bytes_per_session: int = DEFAULT_UNTRACKED_SESSION_BYTES,
    policy: SessionCapacityPolicy = DEFAULT_SESSION_CAPACITY_POLICY,
) -> SessionCapacityPlan:
    """Calculate a safe retained-session count from worker-local CUDA facts."""
    if configured_limit is not None and configured_limit < 1:
        raise ValueError("configured_limit must be positive when provided")
    if untracked_bytes_per_session < 0:
        raise ValueError("untracked_bytes_per_session must be non-negative")
    if any(not role for role in role_budgets):
        raise ValueError("role budget names must not be empty")

    by_device: dict[int, list[Mapping[str, int | str]]] = {}
    for snapshot in snapshots:
        device_index = int(snapshot["device_index"])
        by_device.setdefault(device_index, []).append(snapshot)

    device_plans: list[DeviceSessionCapacity] = []
    for device_index, device_snapshots in sorted(by_device.items()):
        total_bytes = min(int(snapshot["total_bytes"]) for snapshot in device_snapshots)
        free_bytes = min(int(snapshot["free_bytes"]) for snapshot in device_snapshots)
        reusable_reserved = sum(
            max(int(snapshot["reserved_bytes"]) - int(snapshot["allocated_bytes"]), 0) for snapshot in device_snapshots
        )
        raw_transient_headroom = 0
        transient_headroom = 0
        role_process_counts: dict[str, int] = {}
        for snapshot in device_snapshots:
            role = str(snapshot.get("role", ""))
            budget = role_budgets.get(role)
            raw_bytes, steady_bytes = _transient_bytes(snapshot, budget)
            raw_transient_headroom += raw_bytes
            transient_headroom += steady_bytes
            if budget is not None:
                role_process_counts[role] = role_process_counts.get(role, 0) + 1

        safety_margin = max(
            total_bytes * policy.safety_margin_percent // 100,
            policy.minimum_safety_margin_bytes,
        )
        role_bytes = {
            role: role_budgets[role].persistent_bytes_per_session * process_count
            for role, process_count in role_process_counts.items()
        }
        persistent_per_session = sum(role_bytes.values())
        computed_capacity = 0
        for candidate in range(1, policy.max_capacity + 1):
            physical_growth = candidate * untracked_bytes_per_session
            for snapshot in device_snapshots:
                role = str(snapshot.get("role", ""))
                budget = role_budgets.get(role)
                persistent_bytes = budget.persistent_bytes_per_session if budget is not None else 0
                cached_bytes = max(
                    int(snapshot["reserved_bytes"]) - int(snapshot["allocated_bytes"]),
                    0,
                )
                _, transient_bytes = _transient_bytes(snapshot, budget)
                physical_growth += max(candidate * persistent_bytes + transient_bytes - cached_bytes, 0)
            if physical_growth + safety_margin > free_bytes:
                break
            computed_capacity = candidate

        device_plans.append(
            DeviceSessionCapacity(
                device_index=device_index,
                device_name=str(device_snapshots[0]["device_name"]),
                total_bytes=total_bytes,
                free_bytes=free_bytes,
                reusable_reserved_bytes=reusable_reserved,
                raw_transient_headroom_bytes=raw_transient_headroom,
                transient_headroom_bytes=transient_headroom,
                safety_margin_bytes=safety_margin,
                persistent_bytes_per_session=persistent_per_session,
                persistent_bytes_by_role=role_bytes,
                untracked_bytes_per_session=untracked_bytes_per_session,
                computed_capacity=computed_capacity,
            )
        )

    if not device_plans:
        fallback = min(configured_limit or 1, policy.max_capacity)
        return SessionCapacityPlan(
            computed_capacity=fallback,
            effective_capacity=fallback,
            configured_limit=configured_limit,
            limiting_device=None,
            devices=(),
        )

    limiting = min(device_plans, key=lambda item: item.computed_capacity)
    if limiting.computed_capacity < 1:
        raise RuntimeError(
            "Insufficient GPU memory for one retained session: "
            f"device={limiting.device_index} free={limiting.free_bytes} "
            f"reusable_reserved={limiting.reusable_reserved_bytes} "
            f"transient_headroom={limiting.transient_headroom_bytes} "
            f"safety_margin={limiting.safety_margin_bytes} "
            f"bytes_per_session="
            f"{limiting.persistent_bytes_per_session + limiting.untracked_bytes_per_session}"
        )
    effective = limiting.computed_capacity
    if configured_limit is not None:
        effective = min(effective, configured_limit)
    return SessionCapacityPlan(
        computed_capacity=limiting.computed_capacity,
        effective_capacity=effective,
        configured_limit=configured_limit,
        limiting_device=limiting.device_index,
        devices=tuple(device_plans),
    )


class SessionSlotPool:
    """Thread-safe fixed-capacity slot ownership without storage assumptions."""

    def __init__(self, capacity: int, *, name: str = "Session cache") -> None:
        if capacity < 1:
            raise ValueError(f"slot pool capacity must be positive, got {capacity}")
        if not name:
            raise ValueError("slot pool name must not be empty")
        self.capacity = capacity
        self.name = name
        self._lock = threading.Lock()
        self._free_slots = list(reversed(range(capacity)))

    def acquire(self) -> int:
        slot = self.try_acquire()
        if slot is None:
            raise RuntimeError(f"{self.name} pool is full (capacity={self.capacity})")
        return slot

    def try_acquire(self) -> int | None:
        """Acquire a slot, or return None when the pool is exhausted."""
        with self._lock:
            if not self._free_slots:
                return None
            return self._free_slots.pop()

    def release(self, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise ValueError(f"{self.name} slot {slot} is outside capacity {self.capacity}")
        with self._lock:
            if slot in self._free_slots:
                raise RuntimeError(f"{self.name} slot {slot} was released twice")
            self._free_slots.append(slot)
