"""Tests for device mesh module."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from telefuser.core.config import ParallelConfig
from telefuser.distributed import device_mesh
from telefuser.distributed.device_mesh import (
    _validate_parallel_config,
    create_device_mesh_from_config,
    get_group,
    get_rank,
    get_ranks,
    get_world_size,
)

# Skip if distributed not available
try:
    import torch.distributed as dist

    HAS_DISTRIBUTED = dist.is_available()
except ImportError:
    HAS_DISTRIBUTED = False

pytestmark = [
    pytest.mark.skipif(not HAS_DISTRIBUTED, reason="Distributed not available"),
    pytest.mark.distributed,
]


class TestValidateParallelConfig:
    """Test _validate_parallel_config function with various configurations."""

    @pytest.mark.parametrize(
        "device_ids,dp,cfg,tp,sp_ring,sp_ulysses",
        [
            ([0], 1, 1, 1, 1, 1),  # Single GPU
            ([0, 1, 2, 3], 4, 1, 1, 1, 1),  # Data parallel
            ([0, 1, 2, 3], 1, 1, 4, 1, 1),  # Tensor parallel
            ([0, 1, 2, 3], 1, 1, 1, 2, 2),  # Sequence parallel
            ([0, 1], 1, 2, 1, 1, 1),  # CFG parallel
            ([0, 1, 2, 3, 4, 5, 6, 7], 2, 2, 1, 2, 1),  # Combined
            ([0, 1, 2, 3], 1, 1, 2, 1, 2),  # Ulysses + tensor parallel
        ],
    )
    def test_valid_configurations(self, device_ids, dp, cfg, tp, sp_ring, sp_ulysses):
        """Test various valid parallel configurations."""
        config = ParallelConfig(
            device_ids=device_ids,
            dp_degree=dp,
            cfg_degree=cfg,
            tp_degree=tp,
            sp_ring_degree=sp_ring,
            sp_ulysses_degree=sp_ulysses,
        )
        # Should not raise any exception
        _validate_parallel_config(config)
        assert config.world_size == len(device_ids)

    def test_sp_and_tp_together(self):
        """Test that SP and TP can form a two-dimensional mesh."""
        config = ParallelConfig(
            device_ids=[0, 1, 2, 3],
            dp_degree=1,
            cfg_degree=1,
            tp_degree=2,
            sp_ring_degree=1,
            sp_ulysses_degree=2,
        )
        _validate_parallel_config(config)

    def test_invalid_world_size_mismatch(self):
        """Test that device count must match parallel degrees."""
        config = ParallelConfig(
            device_ids=[0, 1],  # Only 2 devices
            dp_degree=2,
            cfg_degree=2,  # Requires 4 devices
            tp_degree=1,
            sp_ring_degree=1,
            sp_ulysses_degree=1,
        )
        with pytest.raises((ValueError, RuntimeError)):
            _validate_parallel_config(config)


class TestCreateDeviceMeshFromConfig:
    """Test create_device_mesh_from_config function."""

    @pytest.mark.parametrize(
        "device_ids,dp,cfg,tp,sp_ring,sp_ulysses,expected_names",
        [
            ([0], 1, 1, 1, 1, 1, ["world"]),
            ([0, 1, 2, 3], 4, 1, 1, 1, 1, ["dp"]),
            ([0, 1, 2, 3], 1, 1, 4, 1, 1, ["tp"]),
            ([0, 1, 2, 3], 1, 1, 1, 2, 2, ["ring", "ulysses"]),  # USP
            ([0, 1, 2, 3], 1, 1, 1, 4, 1, ["ring"]),  # Ring only
            ([0, 1, 2, 3], 1, 1, 1, 1, 4, ["ulysses"]),  # Ulysses only
            ([0, 1], 1, 2, 1, 1, 1, ["cfg"]),
            ([0, 1, 2, 3, 4, 5, 6, 7], 2, 2, 1, 2, 1, ["dp", "cfg", "ring"]),  # DP+CFG+Ring
            ([0, 1, 2, 3], 1, 1, 2, 1, 2, ["ulysses", "tp"]),  # Ulysses + TP
        ],
    )
    @patch("telefuser.distributed.device_mesh.DeviceMesh")
    @patch("telefuser.distributed.device_mesh._validate_parallel_config")
    def test_mesh_creation_variants(
        self, mock_validate, mock_device_mesh, device_ids, dp, cfg, tp, sp_ring, sp_ulysses, expected_names
    ):
        """Test creating mesh with different parallelism configurations."""
        config = ParallelConfig(
            device_ids=device_ids,
            dp_degree=dp,
            cfg_degree=cfg,
            tp_degree=tp,
            sp_ring_degree=sp_ring,
            sp_ulysses_degree=sp_ulysses,
        )

        create_device_mesh_from_config(config)

        call_kwargs = mock_device_mesh.call_args.kwargs
        assert list(call_kwargs["mesh_dim_names"]) == expected_names

    @pytest.mark.parametrize("device_type", ["cuda", "cpu"])
    @patch("telefuser.distributed.device_mesh.DeviceMesh")
    @patch("telefuser.distributed.device_mesh._validate_parallel_config")
    def test_mesh_with_different_device_types(self, mock_validate, mock_device_mesh, device_type):
        """Test creating mesh with different device types."""
        config = ParallelConfig(device_ids=[0, 1], dp_degree=2)
        create_device_mesh_from_config(config, device_type=device_type)
        assert mock_device_mesh.call_args.kwargs["device_type"] == device_type


class TestUtilityFunctions:
    """Test utility functions with edge cases."""

    def test_get_group_success(self):
        """Test get_group returns group on success."""
        mock_mesh = MagicMock()
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        result = get_group(mock_mesh, "dp")
        assert result == mock_group

    def test_get_group_keyerror_returns_none(self):
        """Test get_group returns None on KeyError."""
        mock_mesh = MagicMock()
        mock_mesh.get_group.side_effect = KeyError("dp")

        result = get_group(mock_mesh, "dp")
        assert result is None

    def test_get_group_runtimeerror_returns_none(self):
        """Test get_group returns None on RuntimeError."""
        mock_mesh = MagicMock()
        mock_mesh.get_group.side_effect = RuntimeError("Not initialized")

        result = get_group(mock_mesh, "dp")
        assert result is None

    @patch("telefuser.distributed.device_mesh.dist.get_process_group_ranks")
    def test_get_ranks_with_group(self, mock_get_ranks):
        """Test get_ranks with valid group."""
        mock_mesh = MagicMock()
        mock_mesh.get_group.return_value = MagicMock()
        mock_get_ranks.return_value = [0, 1, 2, 3]

        result = get_ranks(mock_mesh, "dp")
        assert result == [0, 1, 2, 3]

    def test_get_ranks_no_group(self):
        """Test get_ranks returns empty list when group doesn't exist."""
        mock_mesh = MagicMock()
        mock_mesh.get_group.side_effect = KeyError("dp")

        assert get_ranks(mock_mesh, "dp") == []

    @pytest.mark.parametrize(
        "has_group,world_size,expected",
        [
            (True, 4, 4),  # With valid group
            (False, None, 1),  # Without group (fallback)
        ],
    )
    @patch("telefuser.distributed.device_mesh.dist.get_world_size")
    def test_get_world_size_variants(self, mock_get_world_size, has_group, world_size, expected):
        """Test get_world_size with and without group."""
        mock_mesh = MagicMock()

        if has_group:
            mock_mesh.get_group.return_value = MagicMock()
            mock_get_world_size.return_value = world_size
        else:
            mock_mesh.get_group.side_effect = KeyError("dp")

        assert get_world_size(mock_mesh, "dp") == expected

    @pytest.mark.parametrize(
        "has_group,rank,expected",
        [
            (True, 2, 2),  # With valid group
            (False, None, 0),  # Without group (fallback)
        ],
    )
    @patch("telefuser.distributed.device_mesh.dist.get_rank")
    def test_get_rank_variants(self, mock_get_rank, has_group, rank, expected):
        """Test get_rank with and without group."""
        mock_mesh = MagicMock()

        if has_group:
            mock_mesh.get_group.return_value = MagicMock()
            mock_get_rank.return_value = rank
        else:
            mock_mesh.get_group.side_effect = KeyError("dp")

        assert get_rank(mock_mesh, "dp") == expected


class TestConvenienceFunctions:
    """Test convenience wrapper functions return correct group types."""

    @pytest.mark.parametrize(
        "func_name,expected_group",
        [
            ("get_dp_group", "dp"),
            ("get_cfg_group", "cfg"),
            ("get_ring_group", "ring"),
            ("get_ulysses_group", "ulysses"),
            ("get_tp_group", "tp"),
        ],
    )
    def test_group_getters_return_expected_group(self, func_name, expected_group):
        """Test convenience group getter functions return correct group."""
        func = getattr(device_mesh, func_name)
        mock_mesh = MagicMock()
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        result = func(mock_mesh)

        # Verify the function returns the group from the mesh
        assert result == mock_group
        # Verify correct group type was requested
        mock_mesh.get_group.assert_called_once_with(expected_group)


# Integration tests that require actual distributed initialization
@pytest.mark.distributed
class TestDeviceMeshIntegration:
    """Integration tests for DeviceMesh with actual distributed environment."""

    def test_device_mesh_creation_integration(self):
        """Test creating DeviceMesh in actual distributed environment."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()

        config = ParallelConfig(
            device_ids=list(range(world_size)),
            dp_degree=world_size,
            cfg_degree=1,
            tp_degree=1,
            sp_ring_degree=1,
            sp_ulysses_degree=1,
        )

        mesh = create_device_mesh_from_config(config)
        assert mesh is not None
