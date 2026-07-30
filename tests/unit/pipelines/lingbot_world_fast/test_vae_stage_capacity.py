from __future__ import annotations

import pytest
import torch

from telefuser.pipelines.lingbot_world_fast import vae_stage
from telefuser.pipelines.lingbot_world_fast.vae_stage import _VAECachePool, _cache_tensor_bytes


def test_cache_tensor_bytes_ignores_non_tensor_entries() -> None:
    cache = [
        torch.empty((2, 3), dtype=torch.float32),
        None,
        "Rep",
        torch.empty((4,), dtype=torch.bfloat16),
    ]

    assert _cache_tensor_bytes(cache) == 32


def test_vae_cache_pool_stabilizes_and_reuses_fixed_slots() -> None:
    pool = _VAECachePool(
        capacity=2,
        layout={
            0: (torch.float32, 8),
            2: (torch.bfloat16, 6),
        },
        device=torch.device("cpu"),
    )
    first_slot = pool.acquire()
    second_slot = pool.acquire()

    assert pool.try_acquire() is None
    with pytest.raises(RuntimeError, match="pool is full"):
        pool.acquire()

    cache: list[object] = [
        torch.arange(6, dtype=torch.float32).view(2, 3),
        "Rep",
        torch.arange(4, dtype=torch.bfloat16),
    ]
    original = cache[0].clone()
    pool.stabilize(cache, first_slot)

    assert torch.equal(cache[0], original)
    assert cache[0].untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert pool.bytes_per_session == 44

    pool.release(first_slot)
    assert pool.acquire() == first_slot
    pool.release(first_slot)
    pool.release(second_slot)


def test_vae_encode_stage_reports_capacity_without_raising() -> None:
    stage = vae_stage.LingBotWorldFastVAEEncodeStage.__new__(vae_stage.LingBotWorldFastVAEEncodeStage)
    stage._cache_registry = {}
    stage._cache_pool = _VAECachePool(capacity=1, layout={}, device=torch.device("cpu"))
    image = torch.zeros(3, 4, 4)

    assert stage.initialize_cache(1, image) is True
    assert stage.initialize_cache(2, image) is False
    assert stage.release_cache(1) is True
    assert stage.initialize_cache(2, image) is True
    assert stage.release_cache(2) is True


def test_vae_decode_stage_reports_capacity_without_raising() -> None:
    stage = vae_stage.LingBotWorldFastVAEDecodeStage.__new__(vae_stage.LingBotWorldFastVAEDecodeStage)
    stage._cache_registry = {}
    stage._cache_pool = _VAECachePool(capacity=1, layout={}, device=torch.device("cpu"))

    assert stage.initialize_cache(1) is True
    assert stage.initialize_cache(2) is False
    assert stage.release_cache(1) is True
    assert stage.initialize_cache(2) is True
    assert stage.release_cache(2) is True
