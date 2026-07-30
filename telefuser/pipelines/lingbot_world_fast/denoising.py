from __future__ import annotations

from dataclasses import dataclass

import torch

from telefuser.cache.session_memory import SessionSlotPool
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_ulysses_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.models.lingbot_world_fast_dit import LingBotWorldFastDiT
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.logging import logger


def _select_timesteps(
    scheduler: FlowUniPCMultistepScheduler,
    indices: tuple[int, ...],
    shift: float,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    if not indices:
        raise ValueError("timestep indices must not be empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError(f"timestep indices must be integers, got {indices!r}")
    if any(index < 0 or index >= num_train_timesteps for index in indices):
        raise ValueError(f"timestep indices must be in [0, {num_train_timesteps}), got {indices!r}")
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError(f"timestep indices must be strictly increasing, got {indices!r}")

    scheduler.set_timesteps(num_train_timesteps, shift=shift)
    if max(indices) >= len(scheduler.timesteps):
        raise ValueError(f"timestep index exceeds scheduler output: {indices!r}")
    return scheduler.timesteps[list(indices)].clone()


@dataclass
class _DenoisingCacheState:
    scheduler: FlowUniPCMultistepScheduler
    timesteps: torch.Tensor
    self_kv_cache: list[dict[str, torch.Tensor | int]]
    crossattn_cache: list[dict[str, torch.Tensor | bool | int]]
    generator: torch.Generator
    noise_generator: torch.Generator
    noise_shape: tuple[int, int, int, int, int]
    pool_slot: int | None = None


@dataclass(frozen=True)
class DenoisingCachePoolProfile:
    """Fixed cache-pool dimensions and storage size for one worker rank."""

    capacity: int
    batch_size: int
    kv_size: int
    max_sequence_length: int
    bytes_per_session: int
    allocated_bytes: int


class _DenoisingCachePool:
    """Preallocated rank-local KV storage split into reusable session slots."""

    def __init__(
        self,
        *,
        capacity: int,
        num_layers: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
        num_heads: int,
        local_num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.kv_size = kv_size
        self.max_sequence_length = max_sequence_length
        self._slots = SessionSlotPool(capacity, name="LingBot KV cache")

        self.self_k = torch.empty(
            (capacity, num_layers, batch_size, kv_size, local_num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.self_v = torch.empty_like(self.self_k)
        self.cross_k = torch.empty(
            (capacity, num_layers, batch_size, max_sequence_length, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.cross_v = torch.empty_like(self.cross_k)
        self.cursors = torch.zeros((capacity, num_layers, 2), dtype=torch.int64, device=device)

    @property
    def allocated_bytes(self) -> int:
        tensors = (self.self_k, self.self_v, self.cross_k, self.cross_v, self.cursors)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def bytes_per_session(self) -> int:
        return self.allocated_bytes // self.capacity

    def acquire(
        self,
    ) -> tuple[int, list[dict[str, torch.Tensor | int]], list[dict[str, torch.Tensor | bool | int]]]:
        acquired = self.try_acquire()
        if acquired is None:
            raise RuntimeError(f"LingBot KV cache pool is full (capacity={self.capacity})")
        return acquired

    def try_acquire(
        self,
    ) -> tuple[int, list[dict[str, torch.Tensor | int]], list[dict[str, torch.Tensor | bool | int]]] | None:
        """Acquire reusable KV views without raising on capacity exhaustion."""
        slot = self._slots.try_acquire()
        if slot is None:
            return None
        self.cursors[slot].zero_()
        self_cache = [
            {
                "k": self.self_k[slot, layer],
                "v": self.self_v[slot, layer],
                "global_end_index": self.cursors[slot, layer, 0],
                "local_end_index": self.cursors[slot, layer, 1],
            }
            for layer in range(self.num_layers)
        ]
        cross_cache = [
            {
                "k": self.cross_k[slot, layer],
                "v": self.cross_v[slot, layer],
                "is_init": False,
                "sequence_length": 0,
            }
            for layer in range(self.num_layers)
        ]
        return slot, self_cache, cross_cache

    def release(self, slot: int) -> None:
        self._slots.release(slot)


class LingBotWorldFastDenoisingStage(BaseStage):
    """Chunk-level denoising stage with worker-local persistent KV caches."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.dit: LingBotWorldFastDiT = module_manager.fetch_module("lingbot_world_fast_dit")
        if self.dit is None:
            raise ValueError("LingBot denoising stage requires a loaded lingbot_world_fast_dit module")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.model_names = ["dit"]
        self._cache_registry: dict[int, _DenoisingCacheState] = {}
        self._cache_pool: _DenoisingCachePool | None = None
        if model_runtime_config.parallel_config.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info(f"Enabling torch.compile for {self.name}")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    def parallel_models(self) -> None:
        """Configure Ulysses SP and optional FSDP inside a ParallelWorker."""
        parallel_config = self.model_runtime_config.parallel_config
        self.dit.device_mesh = create_device_mesh_from_config(parallel_config)
        self.dit.set_attention_config(self.model_runtime_config.attention_config)
        if parallel_config.sp_ulysses_degree > 1:
            self.dit.enable_usp(self.dit.device_mesh)
        if parallel_config.enable_fsdp:
            logger.info(f"Enabling FSDP for {self.name}")
            self.dit = shard_model(
                module=self.dit,
                device_id=self.device,
                wrap_module_names=self.dit.get_fsdp_module_names(),
                param_dtype=self.torch_dtype,
                reduce_dtype=self.torch_dtype,
                buffer_dtype=self.torch_dtype,
            )
            self.onload_models_flag = True
        if self.model_runtime_config.compile_config.enabled:
            logger.info(f"Enabling torch.compile for {self.name}")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())

    def _init_self_kv_cache(
        self,
        batch_size: int,
        kv_size: int,
    ) -> list[dict[str, torch.Tensor | int]]:
        head_dim = self.dit.dim // self.dit.num_heads
        device_mesh = getattr(self.dit, "device_mesh", None)
        ulysses_world_size = get_ulysses_world_size(device_mesh)
        num_heads = self.dit.num_heads
        if ulysses_world_size > 1:
            if num_heads % ulysses_world_size:
                raise ValueError(
                    f"LingBot Ulysses SP requires {num_heads} attention heads to be divisible "
                    f"by degree {ulysses_world_size}"
                )
            num_heads //= ulysses_world_size

        shape = (batch_size, kv_size, num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                # These cursors cross actor/FSDP call boundaries.  Keep their
                # storage inside the cache so shallow argument copies retain
                # the updates made by the attention block.
                "global_end_index": torch.zeros((), dtype=torch.int64, device=self.device),
                "local_end_index": torch.zeros((), dtype=torch.int64, device=self.device),
            }
            for _ in range(self.dit.num_layers)
        ]

    def _init_crossattn_cache(
        self,
        batch_size: int,
        max_sequence_length: int,
    ) -> list[dict[str, torch.Tensor | bool | int]]:
        head_dim = self.dit.dim // self.dit.num_heads
        shape = (batch_size, max_sequence_length, self.dit.num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "is_init": False,
                "sequence_length": 0,
            }
            for _ in range(self.dit.num_layers)
        ]

    def _cache_dimensions(self) -> tuple[int, int, int]:
        head_dim = self.dit.dim // self.dit.num_heads
        device_mesh = getattr(self.dit, "device_mesh", None)
        ulysses_world_size = get_ulysses_world_size(device_mesh)
        if self.dit.num_heads % ulysses_world_size:
            raise ValueError(
                f"LingBot attention heads {self.dit.num_heads} are not divisible by SP degree {ulysses_world_size}"
            )
        return head_dim, self.dit.num_heads, self.dit.num_heads // ulysses_world_size

    def estimate_session_cache_bytes(self, batch_size: int, kv_size: int, max_sequence_length: int) -> int:
        """Return exact persistent DiT KV bytes for one session on this rank."""
        head_dim, num_heads, local_num_heads = self._cache_dimensions()
        element_size = torch.empty((), dtype=self.torch_dtype).element_size()
        self_kv = 2 * self.dit.num_layers * batch_size * kv_size * local_num_heads * head_dim * element_size
        cross_kv = 2 * self.dit.num_layers * batch_size * max_sequence_length * num_heads * head_dim * element_size
        cursors = self.dit.num_layers * 2 * torch.empty((), dtype=torch.int64).element_size()
        return self_kv + cross_kv + cursors

    def configure_cache_pool(
        self,
        capacity: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
    ) -> DenoisingCachePoolProfile:
        """Allocate all persistent DiT KV slots before accepting sessions."""
        if capacity < 1:
            raise ValueError(f"cache pool capacity must be positive, got {capacity}")
        if self._cache_registry:
            raise RuntimeError("cannot configure the LingBot KV cache pool while sessions are active")
        existing = getattr(self, "_cache_pool", None)
        if existing is not None:
            profile = DenoisingCachePoolProfile(
                capacity=existing.capacity,
                batch_size=existing.batch_size,
                kv_size=existing.kv_size,
                max_sequence_length=existing.max_sequence_length,
                bytes_per_session=existing.bytes_per_session,
                allocated_bytes=existing.allocated_bytes,
            )
            requested = (capacity, batch_size, kv_size, max_sequence_length)
            current = (profile.capacity, profile.batch_size, profile.kv_size, profile.max_sequence_length)
            if requested != current:
                raise RuntimeError(f"LingBot KV cache pool is already configured as {current}, requested {requested}")
            return profile

        head_dim, num_heads, local_num_heads = self._cache_dimensions()
        pool = _DenoisingCachePool(
            capacity=capacity,
            num_layers=self.dit.num_layers,
            batch_size=batch_size,
            kv_size=kv_size,
            max_sequence_length=max_sequence_length,
            num_heads=num_heads,
            local_num_heads=local_num_heads,
            head_dim=head_dim,
            dtype=self.torch_dtype,
            device=self.device,
        )
        self._cache_pool = pool
        return DenoisingCachePoolProfile(
            capacity=capacity,
            batch_size=batch_size,
            kv_size=kv_size,
            max_sequence_length=max_sequence_length,
            bytes_per_session=pool.bytes_per_session,
            allocated_bytes=pool.allocated_bytes,
        )

    @with_model_offload(["dit"])
    def initialize_cache(
        self,
        cache_handle: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
        sample_shift: float,
        generator_state: list[int],
        noise_generator_state: list[int],
        noise_shape: tuple[int, int, int, int, int],
        timestep_indices: tuple[int, ...] = (0, 179, 358, 679),
    ) -> bool:
        """Atomically register session-scoped KV, scheduler, and RNG state."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"Cache handle {cache_handle} is already registered")

        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        timesteps = _select_timesteps(scheduler, tuple(timestep_indices), sample_shift)
        generator = torch.Generator(device=self.device)
        generator.set_state(torch.tensor(generator_state, dtype=torch.uint8))
        noise_generator = torch.Generator(device=self.device)
        noise_generator.set_state(torch.tensor(noise_generator_state, dtype=torch.uint8))
        pool = getattr(self, "_cache_pool", None)
        pool_slot = None
        try:
            if pool is None:
                self_kv_cache = self._init_self_kv_cache(batch_size, kv_size)
                crossattn_cache = self._init_crossattn_cache(batch_size, max_sequence_length)
            else:
                if (
                    batch_size != pool.batch_size
                    or kv_size > pool.kv_size
                    or max_sequence_length > pool.max_sequence_length
                ):
                    raise ValueError(
                        "Session cache dimensions exceed the fixed LingBot cache profile: "
                        f"requested={(batch_size, kv_size, max_sequence_length)}, "
                        f"configured={(pool.batch_size, pool.kv_size, pool.max_sequence_length)}"
                    )
                acquired = pool.try_acquire()
                if acquired is None:
                    return False
                pool_slot, self_kv_cache, crossattn_cache = acquired
            state = _DenoisingCacheState(
                scheduler=scheduler,
                timesteps=timesteps,
                self_kv_cache=self_kv_cache,
                crossattn_cache=crossattn_cache,
                generator=generator,
                noise_generator=noise_generator,
                noise_shape=noise_shape,
                pool_slot=pool_slot,
            )
        except Exception:
            if pool is not None and pool_slot is not None:
                pool.release(pool_slot)
            raise
        self._cache_registry[cache_handle] = state
        return True

    @staticmethod
    def _convert_flow_pred_to_x0(
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep.double()).abs())
        sigma_t = sigmas[timestep_id].reshape(-1)
        while sigma_t.ndim < xt.ndim:
            sigma_t = sigma_t.unsqueeze(-1)
        x0 = xt - sigma_t * flow_pred
        return x0.to(original_dtype)

    def denoise_chunk(
        self,
        latent_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        prompt_emb: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
        control_chunk: torch.Tensor | None,
        self_kv_cache: list[dict[str, torch.Tensor | int]],
        crossattn_cache: list[dict[str, torch.Tensor | bool | int]],
        current_start: int,
        max_attention_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        current_latent = latent_chunk
        for timestep_idx in range(len(timesteps)):
            schedule_timestep = timesteps[timestep_idx].view(1).to(device=current_latent.device)
            model_timestep = schedule_timestep.to(dtype=torch.float32)
            with torch.amp.autocast(
                current_latent.device.type,
                dtype=self.torch_dtype,
                enabled=current_latent.device.type == "cuda",
            ):
                noise_pred = self.dit(
                    x=current_latent.to(dtype=self.torch_dtype),
                    timestep=model_timestep,
                    context=prompt_emb,
                    y=condition_chunk,
                    control_tensor=control_chunk,
                    kv_cache=self_kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    max_attention_size=max_attention_size,
                )
            x0 = self._convert_flow_pred_to_x0(noise_pred, current_latent, schedule_timestep[0], scheduler)
            if timestep_idx < len(timesteps) - 1:
                next_timestep = timesteps[timestep_idx + 1].view(1).to(device=x0.device)
                noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                current_latent = scheduler.add_noise(x0, noise, next_timestep)
            else:
                current_latent = x0

        logger.debug("LingBotWorldFast chunk denoised")
        return current_latent

    def _next_noise_chunk(self, state: _DenoisingCacheState) -> torch.Tensor:
        """Generate the replicated pre-Ulysses input noise for one causal chunk."""
        return torch.randn(
            state.noise_shape,
            generator=state.noise_generator,
            device=self.device,
            dtype=torch.float32,
        )

    @with_model_offload(["dit"])
    def denoise_and_update_cache(
        self,
        cache_handle: int,
        condition_chunk: torch.Tensor,
        prompt_emb: torch.Tensor,
        control_chunk: torch.Tensor | None,
        current_start: int,
        max_attention_size: int,
    ) -> torch.Tensor:
        """Denoise a chunk and commit its clean KV state inside each worker."""
        try:
            state = self._cache_registry[cache_handle]
        except KeyError as exc:
            raise KeyError(f"Unknown cache handle {cache_handle}") from exc
        denoised = self.denoise_chunk(
            latent_chunk=self._next_noise_chunk(state),
            condition_chunk=condition_chunk,
            prompt_emb=prompt_emb,
            timesteps=state.timesteps,
            scheduler=state.scheduler,
            control_chunk=control_chunk,
            self_kv_cache=state.self_kv_cache,
            crossattn_cache=state.crossattn_cache,
            current_start=current_start,
            max_attention_size=max_attention_size,
            generator=state.generator,
        )
        with torch.amp.autocast(
            self.device.type,
            dtype=self.torch_dtype,
            enabled=self.device.type == "cuda",
        ):
            self.dit(
                x=denoised.to(dtype=self.torch_dtype),
                timestep=torch.zeros((1,), dtype=torch.float32, device=self.device),
                context=prompt_emb,
                y=condition_chunk,
                control_tensor=control_chunk,
                kv_cache=state.self_kv_cache,
                crossattn_cache=state.crossattn_cache,
                current_start=current_start,
                max_attention_size=max_attention_size,
            )
        return denoised

    def advance_noise(self, cache_handle: int) -> bool:
        """Advance the actor-owned noise RNG for a decode-only cache hit."""
        try:
            state = self._cache_registry[cache_handle]
        except KeyError as exc:
            raise KeyError(f"Unknown cache handle {cache_handle}") from exc
        self._next_noise_chunk(state)
        return True

    def has_cache(self, cache_handle: int) -> bool:
        """Return whether this worker owns the requested cache handle."""
        return cache_handle in self._cache_registry

    def list_cache_handles(self) -> tuple[int, ...]:
        """Return registered cache handles for diagnostics and tests."""
        return tuple(sorted(self._cache_registry))

    def release_cache(self, cache_handle: int) -> bool:
        """Idempotently release worker-local state for one generation session."""
        state = self._cache_registry.pop(cache_handle, None)
        if state is None:
            return False
        pool = getattr(self, "_cache_pool", None)
        if pool is not None and state.pool_slot is not None:
            pool.release(state.pool_slot)
        return True
