"""Tests for Ulysses All-to-All communication."""

from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import torch.distributed as dist

    HAS_DISTRIBUTED = dist.is_available()
except ImportError:
    HAS_DISTRIBUTED = False

pytestmark = [
    pytest.mark.skipif(not HAS_DISTRIBUTED, reason="Distributed not available"),
    pytest.mark.distributed,
]


class TestUlyssesScatterHeads:
    """Test head-to-sequence redistribution."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_scatter_heads_accepts_even_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        tensor = torch.randn(2, 10, 32, 64)
        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single", return_value=tensor.flatten()):
            assert callable(ulysses_scatter_heads(tensor, MagicMock()))

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_scatter_heads_rejects_uneven_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        with pytest.raises(ValueError, match="divisible"):
            ulysses_scatter_heads(torch.randn(2, 10, 30, 64), MagicMock())


class TestUlyssesGatherHeads:
    """Test sequence-to-head redistribution."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_gather_heads_accepts_even_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads

        tensor = torch.randn(2, 40, 8, 64)
        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single", return_value=tensor.flatten()):
            assert callable(ulysses_gather_heads(tensor, MagicMock(), num_heads=32))

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_gather_heads_rejects_uneven_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads

        with pytest.raises(ValueError, match="divisible"):
            ulysses_gather_heads(torch.randn(2, 40, 8, 64), MagicMock(), num_heads=30)


def test_local_head_count_requires_even_partition() -> None:
    from telefuser.distributed.ulysses_comm import _local_head_count

    assert _local_head_count(32, 4) == 8
    with pytest.raises(ValueError, match="divisible"):
        _local_head_count(30, 4)


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_scatter_routes_tag_and_grouped_barrier_to_cuda_ipc(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

    tensor = torch.randn(2, 10, 32, 64)
    output = torch.randn(2, 40, 8, 64)
    handle = MagicMock()
    handle.wait.return_value = output
    backend = MagicMock()
    backend.all_to_all_single_4d_async.return_value = handle

    with patch("telefuser.distributed.ulysses_comm._get_cuda_ipc_group", return_value=backend):
        wait = ulysses_scatter_heads(tensor, MagicMock(), tag="q", barrier=False)

    backend.all_to_all_single_4d_async.assert_called_once_with(tensor, mode=0, tag="q", barrier=False)
    assert wait() is output


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_final_grouped_scatter_reuses_pending_cuda_ipc_group(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed import ulysses_comm

    tensor = torch.randn(2, 10, 32, 64)
    process_group = MagicMock()
    backend = MagicMock(has_pending_group=True)
    backend.all_to_all_single_4d_async.return_value.wait.return_value = tensor

    with (
        patch.dict(ulysses_comm._cuda_ipc_groups, {id(process_group): backend}, clear=True),
        patch.object(ulysses_comm, "_get_cuda_ipc_group", return_value=backend),
    ):
        wait = ulysses_comm.ulysses_scatter_heads(tensor, process_group, tag="v")

    backend.all_to_all_single_4d_async.assert_called_once_with(tensor, mode=0, tag="v", barrier=True)
    assert wait() is tensor


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_standalone_scatter_stays_on_nccl(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed import ulysses_comm

    tensor = torch.randn(2, 10, 32, 64)
    with (
        patch.object(ulysses_comm, "_get_cuda_ipc_group") as get_backend,
        patch.object(ulysses_comm.fc, "all_to_all_single", return_value=tensor.flatten()),
    ):
        wait = ulysses_comm.ulysses_scatter_heads(tensor, MagicMock(), tag="qkv")

    get_backend.assert_not_called()
    assert wait().shape == (2, 40, 8, 64)


class _FailingGroupedBackend:
    def __init__(self) -> None:
        self.pending = False

    @property
    def has_pending_group(self) -> bool:
        return self.pending

    def all_to_all_single_4d_async(self, *_args: object, **_kwargs: object) -> None:
        self.pending = True
        raise RuntimeError("injected grouped failure")


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_grouped_ipc_failure_does_not_attempt_partial_nccl_fallback(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed import ulysses_comm

    tensor = torch.randn(2, 10, 32, 64)
    backend = _FailingGroupedBackend()
    with (
        patch.object(ulysses_comm, "_get_cuda_ipc_group", return_value=backend),
        patch.object(ulysses_comm.fc, "all_to_all_single") as nccl,
        pytest.raises(RuntimeError, match="cannot safely fall back"),
    ):
        ulysses_comm.ulysses_scatter_heads(tensor, MagicMock(), tag="q", barrier=False)

    nccl.assert_not_called()


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_ipc_failure_before_group_start_falls_back_to_nccl(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed import ulysses_comm

    tensor = torch.randn(2, 10, 32, 64)
    process_group = MagicMock()
    backend = MagicMock(has_pending_group=False)
    backend.all_to_all_single_4d_async.side_effect = RuntimeError("injected setup failure")
    with (
        patch.dict(ulysses_comm._cuda_ipc_groups, {id(process_group): backend}, clear=True),
        patch.object(ulysses_comm, "_get_cuda_ipc_group", return_value=backend),
        patch.object(ulysses_comm.fc, "all_to_all_single", return_value=tensor.flatten()),
    ):
        wait = ulysses_comm.ulysses_scatter_heads(tensor, process_group, tag="q", barrier=False)

    backend.close.assert_called_once_with()
    assert wait().shape == (2, 40, 8, 64)


def test_close_cuda_ipc_groups_closes_initialized_groups_and_clears_cache() -> None:
    from telefuser.distributed import ulysses_comm

    first = MagicMock()
    second = MagicMock()
    with patch.dict(ulysses_comm._cuda_ipc_groups, {1: first, 2: None, 3: second}, clear=True):
        ulysses_comm._close_cuda_ipc_groups()
        assert ulysses_comm._cuda_ipc_groups == {}

    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
