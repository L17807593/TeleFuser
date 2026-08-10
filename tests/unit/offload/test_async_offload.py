"""Tests for async offload functionality."""

import time
from unittest.mock import Mock, patch

import pytest
import torch

from telefuser.offload.async_offload import AsyncOffloadManager

# All tests in this file require GPU
pytestmark = pytest.mark.gpu


class TestLayer(torch.nn.Module):
    """Simple test layer with parameters and buffers for testing."""

    def __init__(self, input_size=128, output_size=128):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.randn(output_size))
        self.register_buffer("running_mean", torch.zeros(output_size))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class PerformanceLayer(torch.nn.Module):
    """Performance testing layer with realistic computation."""

    def __init__(self, input_size=512, output_size=512):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MixedDtypeLayer(torch.nn.Module):
    """Layer with the same structure but multiple weight and buffer dtypes."""

    def __init__(self, size=16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(size, size, dtype=torch.bfloat16))
        self.register_buffer("scale", torch.ones(size, dtype=torch.float32))


@pytest.fixture
def test_layers():
    """Create test layers with reproducible initialization."""
    layers = torch.nn.ModuleList([TestLayer(128, 128) for _ in range(3)])
    for i, layer in enumerate(layers):
        torch.manual_seed(i)
        layer.weight.data.normal_(0, 0.02)
        layer.bias.data.zero_()
        layer.running_mean.data.zero_()
    return layers


class TestAsyncOffloadInitialization:
    """Test AsyncOffloadManager initialization."""

    def test_basic_initialization(self, test_layers):
        """Test that AsyncOffloadManager initializes correctly."""
        manager = AsyncOffloadManager(test_layers)

        assert manager.enabled
        assert manager.num_layers == 3
        assert manager.pin_cpu_memory
        assert manager.device.type == "cuda"
        assert manager.copy_stream is not None
        assert 0 in manager._gpu_layers  # Initial non-resident window loaded

    def test_initialization_with_device(self, test_layers):
        """Test initialization with explicit device."""
        device = torch.device("cuda")
        manager = AsyncOffloadManager(test_layers, device=device)
        assert manager.device == device

    def test_disabled_mode(self, test_layers):
        """Test that disabled mode works correctly."""
        manager = AsyncOffloadManager(test_layers, enabled=False)
        assert not manager.enabled

    @pytest.mark.parametrize("lazy_cache", [True, False])
    def test_lazy_gpu_cache_initialization(self, test_layers, lazy_cache):
        """Test lazy_gpu_cache initialization option."""
        manager = AsyncOffloadManager(test_layers, lazy_gpu_cache=lazy_cache)
        assert manager.lazy_gpu_cache == lazy_cache
        assert bool(manager._gpu_buffer_pool) is not lazy_cache

    def test_uniform_resident_layer_selection(self):
        """Resident blocks should be spread across the execution sequence."""
        layers = torch.nn.ModuleList([TestLayer(16, 16) for _ in range(8)])
        manager = AsyncOffloadManager(layers, offload_ratio=0.5, prefetch_size=2)

        assert manager.num_resident_layers == 4
        assert manager._resident_layers == [0, 2, 5, 7]
        assert manager._resident_layer_set.issubset(manager._gpu_layers)

    def test_mixed_dtype_weights_share_packed_cpu_allocations(self):
        """Each dtype should use one CPU allocation across all blocks."""
        layers = torch.nn.ModuleList([MixedDtypeLayer() for _ in range(4)])
        manager = AsyncOffloadManager(layers, offload_ratio=1.0)

        assert set(manager._packed_cpu_weights) == {torch.bfloat16, torch.float32}
        for dtype in (torch.bfloat16, torch.float32):
            packed_ptr = manager._packed_cpu_weights[dtype].untyped_storage().data_ptr()
            layer_ptrs = {
                manager._consolidated_cpu_weights[layer_idx][dtype].untyped_storage().data_ptr()
                for layer_idx in range(4)
            }
            assert layer_ptrs == {packed_ptr}


class TestAsyncOffloadPrefetchRelease:
    """Test prefetch and release functionality."""

    def test_prefetch_and_release_basic(self, test_layers):
        """Test basic prefetch and release functionality."""
        manager = AsyncOffloadManager(test_layers)

        # The first non-resident prefetch window is ready for execution.
        assert manager._gpu_layers == {0}

        # Prefetch layer 1 (non-resident layer)
        manager.prefetch_layer(1)
        assert 0 in manager._gpu_layers
        assert 1 in manager._gpu_layers

        # Release layer 1
        manager.release_layer(1)
        assert manager._gpu_layers == {0}

        # With offload_ratio=1.0 no layer is permanently resident.
        manager.release_layer(0)
        assert manager._gpu_layers == set()

    def test_invalid_layer_handling(self, test_layers):
        """Test handling of invalid layer indices."""
        manager = AsyncOffloadManager(test_layers)

        # Invalid layer indices should be handled gracefully
        manager.prefetch_layer(-1)  # Should do nothing
        manager.prefetch_layer(10)  # Should do nothing
        manager.release_layer(-1)  # Should do nothing
        manager.release_layer(10)  # Should do nothing

        # Valid layers should work normally
        manager.prefetch_layer(2)
        assert 2 in manager._gpu_layers

    def test_resident_layers_concept(self, test_layers):
        """Test resident layers functionality with custom offload ratio."""
        manager = AsyncOffloadManager(test_layers, offload_ratio=0.5)
        expected_resident_layers = min(max(1, int(3 * (1 - 0.5))), 3)

        assert manager.num_resident_layers == expected_resident_layers
        assert set(range(expected_resident_layers)).issubset(manager._gpu_layers)

    def test_prefetch_events(self, test_layers):
        """Test that prefetch events are recorded and retained for reuse."""
        manager = AsyncOffloadManager(test_layers)

        assert 0 in manager._prefetch_events

        manager.prefetch_layer(1)
        assert 1 in manager._prefetch_events
        assert isinstance(manager._prefetch_events[1], torch.cuda.Event)

        manager.release_layer(1)
        assert 1 in manager._prefetch_events

    def test_reused_buffer_waits_for_its_compute_stream(self, test_layers):
        """Test that a reused H2D buffer depends on its prior compute work."""
        manager = AsyncOffloadManager(test_layers)
        dtype = torch.float32

        manager.prefetch_layer(1, non_blocking=False)
        buffer = manager._layer_to_gpu_buffer[1][dtype]
        manager._gpu_buffer_pool[dtype].clear()
        manager._gpu_buffer_sizes[dtype].clear()

        buffer.add_(1)
        manager.release_layer(1)
        assert id(buffer) in manager._gpu_buffer_ready_events

        manager.prefetch_layer(2, non_blocking=True)
        assert id(buffer) not in manager._gpu_buffer_ready_events

        torch.cuda.current_stream().wait_event(manager._prefetch_events[2])

    def test_prepare_for_next_req(self, test_layers):
        """Test prepare_for_next_req functionality."""
        manager = AsyncOffloadManager(test_layers)

        manager.release_all()
        assert manager._gpu_layers == set()

        manager.prepare_for_next_req(non_blocking=False)
        assert 0 in manager._gpu_layers


class TestAsyncOffloadOperations:
    """Test async operations and synchronization."""

    def test_synchronous_operations(self, test_layers):
        """Test that computation and offload operations run synchronously."""
        manager = AsyncOffloadManager(test_layers, prefetch_size=2)
        x = torch.randn(4, 128).cuda()

        manager.prefetch_layer(1, non_blocking=False)
        result = test_layers[0](x)
        manager.release_layer(1)

        assert result.shape == (4, 128)

    def test_asynchronous_operations(self, test_layers):
        """Test asynchronous prefetch operations."""
        manager = AsyncOffloadManager(test_layers)

        manager.prefetch_layer(1, non_blocking=True)
        x = torch.randn(2, 128).cuda()
        result = test_layers[0](x)

        if manager.device.type == "cuda":
            torch.cuda.current_stream().wait_stream(manager.copy_stream)

        assert result.shape == (2, 128)
        assert 1 in manager._gpu_layers


class TestAsyncOffloadMemory:
    """Test memory management functionality."""

    def test_gpu_buffer_pool_management(self, test_layers):
        """Test GPU buffer pool management functionality."""
        manager = AsyncOffloadManager(test_layers, prefetch_size=2)

        manager.prefetch_layer(1, non_blocking=False)
        manager.prefetch_layer(2, non_blocking=False)

        assert 1 in manager._layer_to_gpu_buffer
        assert 2 in manager._layer_to_gpu_buffer

        manager.release_layer(1)
        manager.release_layer(2)

        assert 1 not in manager._layer_to_gpu_buffer
        assert 2 not in manager._layer_to_gpu_buffer

    def test_memory_efficiency(self):
        """Test memory efficiency of the offloading system."""
        large_layers = torch.nn.ModuleList([torch.nn.Linear(1024, 1024) for _ in range(5)])
        manager = AsyncOffloadManager(large_layers)

        assert len(manager._gpu_layers) == 1

        manager.load_all_layers()
        assert len(manager._gpu_layers) == 5

        manager.release_all()
        assert manager._gpu_layers == set()

    def test_cleanup_and_reallocate_cycle(self, test_layers):
        """Test cleanup and reallocate cycle for memory management."""
        manager = AsyncOffloadManager(test_layers, lazy_gpu_cache=False)

        initial_buffer_count = sum(len(buffers) for buffers in manager._gpu_buffer_pool.values())

        manager.cleanup_gpu_cache()
        assert len(manager._gpu_buffer_pool) == 0

        manager.allocate_gpu_cache()
        assert len(manager._gpu_buffer_pool) > 0

        final_buffer_count = sum(len(buffers) for buffers in manager._gpu_buffer_pool.values())
        assert final_buffer_count == initial_buffer_count + 1

    def test_gpu_buffer_pool_is_bounded_by_nonresident_layers(self):
        """Ring slots should be capped by the number of offloaded blocks."""
        layers = torch.nn.ModuleList([TestLayer(16, 16) for _ in range(8)])
        manager = AsyncOffloadManager(layers, offload_ratio=0.5, prefetch_size=3)

        expected_slots = min(2 * manager.prefetch_size, len(manager._nonresident_layers))
        allocated_slots = len(manager._gpu_buffer_pool[torch.float32]) + sum(
            torch.float32 in buffers for buffers in manager._layer_to_gpu_buffer.values()
        )
        assert allocated_slots == expected_slots

    def test_auto_initialization_on_prefetch(self, test_layers):
        """Test that buffer pool is auto-initialized when prefetching with lazy_gpu_cache=True."""
        manager = AsyncOffloadManager(test_layers, lazy_gpu_cache=True)

        manager.cleanup_gpu_cache()
        assert len(manager._gpu_buffer_pool) == 0

        manager.prefetch_layer(1, non_blocking=False)
        assert len(manager._gpu_buffer_pool) > 0
        assert 1 in manager._gpu_layers


class TestAsyncOffloadStateManagement:
    """Test enable/disable and state management."""

    def test_enable_disable_offload(self, test_layers):
        """Test enable/disable offload functionality."""
        manager = AsyncOffloadManager(test_layers)

        assert manager.enabled
        assert len(manager._forward_hooks) > 0

        manager.disable_offload()
        assert len(manager._forward_hooks) == 0
        assert len(manager._gpu_layers) == 3

        manager.enable_offload()
        assert len(manager._forward_hooks) > 0
        assert manager._gpu_layers == {0}

        hook_count = len(manager._forward_hooks)
        manager.enable_offload()
        assert len(manager._forward_hooks) == hook_count

    def test_sync_layer_to_cpu(self, test_layers):
        """Test sync_layer_to_cpu functionality."""
        manager = AsyncOffloadManager(test_layers)

        manager.prefetch_layer(1, non_blocking=False)
        original_weight = test_layers[1].weight.data.clone()
        test_layers[1].weight.data.fill_(1.0)

        manager.sync_layer_to_cpu(1)

        cpu_buffer = manager._consolidated_cpu_weights[1][torch.float32]
        meta = manager._weight_metadata[1]["layers.1.weight"]
        cpu_weight = cpu_buffer[meta["offset"] : meta["offset"] + meta["numel"]].view(meta["shape"])
        assert torch.allclose(cpu_weight, torch.ones_like(cpu_weight))

        test_layers[1].weight.data.copy_(original_weight)

    def test_sync_all_layers_to_cpu(self, test_layers):
        """Test sync_all_layers_to_cpu functionality."""
        manager = AsyncOffloadManager(test_layers)

        manager.load_all_layers()
        for layer in test_layers:
            layer.weight.data.fill_(2.0)

        manager.sync_all_layers_to_cpu()

        for layer_idx in range(1, 3):
            cpu_buffer = manager._consolidated_cpu_weights[layer_idx][torch.float32]
            meta = manager._weight_metadata[layer_idx][f"layers.{layer_idx}.weight"]
            cpu_weight = cpu_buffer[meta["offset"] : meta["offset"] + meta["numel"]].view(meta["shape"])
            assert torch.allclose(cpu_weight, torch.full_like(cpu_weight, 2.0))


class TestAsyncOffloadIntegration:
    """Integration tests with actual computation."""

    def test_forward_with_hooks(self, test_layers):
        """Test that forward hooks work with actual computation."""
        manager = AsyncOffloadManager(test_layers)

        class TestModel(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = TestModel(test_layers)
        x = torch.randn(2, 128).cuda()

        with torch.cuda.stream(manager.copy_stream):
            output = model(x)

        assert output.shape == (2, 128)


class TestAsyncOffloadPerformance:
    """Performance comparison tests."""

    @pytest.mark.slow
    def test_performance_comparison(self):
        """Test performance comparison between serial, parallel, and preloaded modes."""
        import statistics

        perf_layers = torch.nn.ModuleList([PerformanceLayer(512, 512) for _ in range(10)])
        manager = AsyncOffloadManager(perf_layers)

        num_runs = 5

        results = {
            "serial": {"total": []},
            "parallel": {"total": [], "overlap": []},
            "preloaded": {"total": []},
        }

        torch.cuda.synchronize()

        for _ in range(num_runs):
            # Serial mode
            manager.release_all()
            manager.prepare_for_next_req()
            start = time.time()
            manager.prefetch_layer(1, non_blocking=False)
            results["serial"]["total"].append(time.time() - start)

            # Parallel mode
            manager.release_all()
            start = time.time()
            onload_start = time.time()
            manager.prefetch_layer(1, non_blocking=True)
            compute_start = time.time()
            compute_time = time.time() - compute_start

            if manager.device.type == "cuda":
                torch.cuda.current_stream().wait_stream(manager.copy_stream)
            onload_time = time.time() - onload_start
            total_time = time.time() - start

            overlap = max(
                0, min(onload_start + onload_time, compute_start + compute_time) - max(onload_start, compute_start)
            )
            results["parallel"]["total"].append(total_time)
            results["parallel"]["overlap"].append(overlap)

            # Preloaded mode
            manager.load_all_layers()
            start = time.time()
            results["preloaded"]["total"].append(time.time() - start)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        serial_mean = statistics.mean(results["serial"]["total"])
        parallel_mean = statistics.mean(results["parallel"]["total"])
        preloaded_mean = statistics.mean(results["preloaded"]["total"])

        assert preloaded_mean <= parallel_mean <= serial_mean, (
            f"Performance validation failed: preloaded({preloaded_mean:.4f}) <= "
            f"parallel({parallel_mean:.4f}) <= serial({serial_mean:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
