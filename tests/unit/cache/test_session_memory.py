from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from telefuser.cache import (
    SessionCapacityPolicy,
    SessionMemoryBudget,
    SessionSlotPool,
    calculate_session_capacity,
    capture_device_memory_snapshot,
)
from telefuser.cache.session_memory import GIB


def _snapshot(
    device: int,
    *,
    role: str,
    free_gib: int,
    reserved_gib: int,
    peak_reserved_gib: int,
    allocated_gib: int | None = None,
    peak_allocated_gib: int | None = None,
) -> dict[str, int | str]:
    if allocated_gib is None:
        allocated_gib = reserved_gib
    if peak_allocated_gib is None:
        peak_allocated_gib = peak_reserved_gib
    return {
        "rank": device,
        "process_id": device,
        "role": role,
        "device_index": device,
        "device_name": f"gpu-{device}",
        "total_bytes": 80 * GIB,
        "free_bytes": free_gib * GIB,
        "allocated_bytes": allocated_gib * GIB,
        "reserved_bytes": reserved_gib * GIB,
        "peak_allocated_bytes": peak_allocated_gib * GIB,
        "peak_reserved_bytes": peak_reserved_gib * GIB,
    }


def test_memory_snapshot_resolves_an_unspecified_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20, 80))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="test-gpu"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 12)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 15)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 16)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    snapshot = capture_device_memory_snapshot(torch.device("cuda"))

    assert snapshot.device_index == 3
    assert snapshot.device_name == "test-gpu"
    assert snapshot.free_bytes == 20
    assert snapshot.total_bytes == 80


def test_capacity_uses_the_most_constrained_device_and_operator_ceiling() -> None:
    snapshots = [
        _snapshot(0, role="denoise", free_gib=60, reserved_gib=10, peak_reserved_gib=12),
        _snapshot(0, role="vae_decode", free_gib=60, reserved_gib=4, peak_reserved_gib=5),
        _snapshot(1, role="denoise", free_gib=50, reserved_gib=10, peak_reserved_gib=12),
    ]

    plan = calculate_session_capacity(
        snapshots,
        role_budgets={"denoise": SessionMemoryBudget(4 * GIB)},
        configured_limit=6,
    )

    assert [device.computed_capacity for device in plan.devices] == [13, 11]
    assert plan.computed_capacity == 11
    assert plan.effective_capacity == 6
    assert plan.limiting_device == 1


def test_capacity_without_cuda_facts_falls_back_safely() -> None:
    automatic = calculate_session_capacity([], role_budgets={}, configured_limit=None)
    capped = calculate_session_capacity([], role_budgets={}, configured_limit=3)

    assert automatic.effective_capacity == 1
    assert capped.effective_capacity == 3
    assert automatic.devices == ()


def test_capacity_includes_a_device_without_a_budgeted_role() -> None:
    snapshots = [
        _snapshot(0, role="denoise", free_gib=60, reserved_gib=10, peak_reserved_gib=12),
        _snapshot(2, role="vae_decode", free_gib=20, reserved_gib=4, peak_reserved_gib=4),
    ]

    plan = calculate_session_capacity(
        snapshots,
        role_budgets={"denoise": SessionMemoryBudget(2 * GIB)},
        configured_limit=None,
        untracked_bytes_per_session=4 * GIB,
    )

    assert plan.limiting_device == 2
    assert plan.computed_capacity == 4
    assert plan.devices[1].persistent_bytes_per_session == 0
    assert plan.devices[1].persistent_bytes_by_role == {}


def test_capacity_accounts_for_role_specific_session_memory() -> None:
    snapshots = [
        _snapshot(0, role="denoise", free_gib=60, reserved_gib=10, peak_reserved_gib=12),
        _snapshot(0, role="vae_encode", free_gib=60, reserved_gib=1, peak_reserved_gib=1),
        _snapshot(1, role="vae_decode", free_gib=50, reserved_gib=1, peak_reserved_gib=1),
    ]

    plan = calculate_session_capacity(
        snapshots,
        role_budgets={
            "denoise": SessionMemoryBudget(4 * GIB),
            "vae_encode": SessionMemoryBudget(2 * GIB),
            "vae_decode": SessionMemoryBudget(3 * GIB),
        },
        configured_limit=None,
    )

    assert plan.devices[0].persistent_bytes_per_session == 6 * GIB
    assert plan.devices[0].persistent_bytes_by_role == {"denoise": 4 * GIB, "vae_encode": 2 * GIB}
    assert plan.devices[1].persistent_bytes_per_session == 3 * GIB
    assert plan.devices[1].persistent_bytes_by_role == {"vae_decode": 3 * GIB}
    assert plan.computed_capacity == 9
    assert plan.limiting_device == 0


def test_capacity_rejects_a_device_without_one_session_of_headroom() -> None:
    snapshots = [_snapshot(0, role="denoise", free_gib=10, reserved_gib=1, peak_reserved_gib=5)]

    with pytest.raises(RuntimeError, match="Insufficient GPU memory"):
        calculate_session_capacity(
            snapshots,
            role_budgets={"denoise": SessionMemoryBudget(4 * GIB)},
            configured_limit=None,
        )


def test_capacity_reuses_process_local_allocator_reservations() -> None:
    snapshots = [
        _snapshot(
            0,
            role="denoise",
            free_gib=24,
            reserved_gib=10,
            allocated_gib=6,
            peak_allocated_gib=8,
            peak_reserved_gib=12,
        )
    ]

    plan = calculate_session_capacity(
        snapshots,
        role_budgets={"denoise": SessionMemoryBudget(8 * GIB)},
        configured_limit=None,
    )

    assert plan.computed_capacity == 2
    assert plan.devices[0].reusable_reserved_bytes == 4 * GIB


def test_capacity_excludes_warmup_memory_replaced_by_fixed_storage() -> None:
    snapshots = [
        _snapshot(
            0,
            role="denoise",
            free_gib=25,
            reserved_gib=10,
            allocated_gib=10,
            peak_allocated_gib=18,
            peak_reserved_gib=18,
        )
    ]

    plan = calculate_session_capacity(
        snapshots,
        role_budgets={
            "denoise": SessionMemoryBudget(
                persistent_bytes_per_session=8 * GIB,
                replaced_warmup_transient_bytes=8 * GIB,
            )
        },
        configured_limit=None,
    )

    assert plan.computed_capacity == 2
    assert plan.devices[0].raw_transient_headroom_bytes == 8 * GIB
    assert plan.devices[0].transient_headroom_bytes == 0


def test_capacity_policy_controls_margin_and_upper_bound() -> None:
    snapshots = [_snapshot(0, role="worker", free_gib=60, reserved_gib=1, peak_reserved_gib=1)]
    policy = SessionCapacityPolicy(
        safety_margin_percent=10,
        minimum_safety_margin_bytes=0,
        max_capacity=2,
    )

    plan = calculate_session_capacity(
        snapshots,
        role_budgets={"worker": SessionMemoryBudget(GIB)},
        configured_limit=None,
        policy=policy,
    )

    assert plan.computed_capacity == 2
    assert plan.devices[0].safety_margin_bytes == 8 * GIB


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"safety_margin_percent": -1}, "safety_margin_percent"),
        ({"safety_margin_percent": 100}, "safety_margin_percent"),
        ({"minimum_safety_margin_bytes": -1}, "minimum_safety_margin_bytes"),
        ({"max_capacity": 0}, "max_capacity"),
    ],
)
def test_capacity_policy_rejects_invalid_values(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SessionCapacityPolicy(**kwargs)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((-1,), "persistent_bytes_per_session"),
        ((0, -1), "replaced_warmup_transient_bytes"),
    ],
)
def test_memory_budget_rejects_invalid_values(args: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SessionMemoryBudget(*args)


def test_slot_pool_exhaustion_release_and_reuse() -> None:
    pool = SessionSlotPool(2, name="Test cache")

    assert pool.acquire() == 0
    assert pool.acquire() == 1
    assert pool.try_acquire() is None
    with pytest.raises(RuntimeError, match=r"Test cache pool is full \(capacity=2\)"):
        pool.acquire()

    pool.release(0)
    assert pool.acquire() == 0


def test_slot_pool_rejects_invalid_and_duplicate_releases() -> None:
    pool = SessionSlotPool(2)
    slot = pool.acquire()

    pool.release(slot)
    with pytest.raises(RuntimeError, match="released twice"):
        pool.release(slot)
    with pytest.raises(ValueError, match="outside capacity"):
        pool.release(-1)
    with pytest.raises(ValueError, match="outside capacity"):
        pool.release(2)


def test_slot_pool_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        SessionSlotPool(0)
    with pytest.raises(ValueError, match="name must not be empty"):
        SessionSlotPool(1, name="")
