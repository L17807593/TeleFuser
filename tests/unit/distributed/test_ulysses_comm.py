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
def test_scatter_routes_grouped_transfer_through_model_communicator(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

    tensor = torch.randn(2, 10, 32, 64)
    output = torch.randn(2, 40, 8, 64)
    communicator = MagicMock(has_pending_group=False)
    communicator.submit.return_value = lambda: output

    wait = ulysses_scatter_heads(
        tensor,
        MagicMock(),
        tag="q",
        barrier=False,
        communicator=communicator,
    )

    communicator.submit.assert_called_once_with(tensor, tag="q", barrier=False)
    assert wait() is output


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_final_grouped_scatter_requires_pending_communicator(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

    tensor = torch.randn(2, 10, 32, 64)
    communicator = MagicMock(has_pending_group=True)
    communicator.submit.return_value = lambda: tensor

    wait = ulysses_scatter_heads(tensor, MagicMock(), tag="v", communicator=communicator)

    communicator.submit.assert_called_once_with(tensor, tag="v", barrier=True)
    assert wait() is tensor


@patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
@patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
def test_standalone_scatter_stays_on_nccl(mock_rank, mock_world_size) -> None:
    del mock_rank, mock_world_size
    from telefuser.distributed import ulysses_comm

    tensor = torch.randn(2, 10, 32, 64)
    communicator = MagicMock(has_pending_group=False)
    with patch.object(ulysses_comm.fc, "all_to_all_single", return_value=tensor.flatten()):
        wait = ulysses_comm.ulysses_scatter_heads(
            tensor,
            MagicMock(),
            tag="qkv",
            communicator=communicator,
        )

    communicator.submit.assert_not_called()
    assert wait().shape == (2, 40, 8, 64)


def test_communicator_prefers_pytorch_symmetric_memory() -> None:
    from telefuser.distributed import ulysses_backend

    tensor = MagicMock(is_cuda=True, device=torch.device("cuda", 0))
    backend = MagicMock()
    symmetric_group = MagicMock()
    symmetric_group.is_available.return_value = True
    symmetric_group.return_value = backend
    communicator = ulysses_backend.UlyssesCommunicator(MagicMock())

    with (
        patch.object(ulysses_backend, "_SymmetricMemoryUlyssesGroup", symmetric_group),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        assert communicator._create_backend(tensor) is backend

    symmetric_group.assert_called_once_with(communicator.process_group, tensor.device)


def test_communicator_falls_back_from_symmetric_memory_to_cuda_ipc() -> None:
    from telefuser.distributed import ulysses_backend

    tensor = MagicMock(is_cuda=True, device=torch.device("cuda", 0))
    ipc_backend = MagicMock()
    ipc_class = MagicMock()
    ipc_class.is_available.return_value = True
    ipc_class.return_value = ipc_backend
    ipc_module = MagicMock(CudaIpcUlyssesGroup=ipc_class)
    communicator = ulysses_backend.UlyssesCommunicator(MagicMock())

    with (
        patch.object(ulysses_backend._SymmetricMemoryUlyssesGroup, "is_available", return_value=False),
        patch.object(ulysses_backend, "import_module", return_value=ipc_module),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        assert communicator._create_backend(tensor) is ipc_backend

    ipc_class.assert_called_once_with(communicator.process_group, tensor.device)


def test_symmetric_memory_backend_requires_pytorch_2_11() -> None:
    from telefuser.distributed.ulysses_backend import _SymmetricMemoryUlyssesGroup

    with (
        patch.object(torch, "__version__", "2.10.1"),
        patch.object(torch.cuda, "is_available") as cuda_available,
    ):
        assert not _SymmetricMemoryUlyssesGroup.is_available()
    cuda_available.assert_not_called()


class _FailingGroupedBackend:
    def __init__(self, *, pending_after_failure: bool) -> None:
        self.pending = False
        self.pending_after_failure = pending_after_failure
        self.closed = False

    @property
    def has_pending_group(self) -> bool:
        return self.pending

    def all_to_all_single_4d_async(self, *_args: object, **_kwargs: object) -> None:
        self.pending = self.pending_after_failure
        raise RuntimeError("injected grouped failure")

    def close(self) -> None:
        self.closed = True


def test_grouped_backend_failure_does_not_attempt_partial_fallback() -> None:
    from telefuser.distributed.ulysses_backend import UlyssesCommunicator

    communicator = UlyssesCommunicator(MagicMock())
    backend = _FailingGroupedBackend(pending_after_failure=True)
    communicator._backend = backend
    communicator._backend_name = "test"

    with pytest.raises(RuntimeError, match="cannot safely fall back"):
        communicator.submit(torch.randn(1), tag="q", barrier=False)

    assert not backend.closed


def test_communicator_close_releases_only_owned_backend() -> None:
    from telefuser.distributed.ulysses_backend import UlyssesCommunicator

    communicator = UlyssesCommunicator(MagicMock())
    backend = MagicMock()
    communicator._backend = backend
    communicator._backend_name = "test"

    communicator.close()

    backend.close.assert_called_once_with()
    assert communicator._backend is None


def test_base_model_owns_and_reuses_ulysses_communicator() -> None:
    from telefuser.core.base_model import BaseModel

    first_group = MagicMock()
    second_group = MagicMock()
    first_communicator = MagicMock(process_group=first_group)
    second_communicator = MagicMock(process_group=second_group)
    model = BaseModel()

    with patch(
        "telefuser.distributed.ulysses_backend.UlyssesCommunicator",
        side_effect=(first_communicator, second_communicator),
    ) as communicator_class:
        assert model._configure_ulysses_communicator(first_group) is first_communicator
        assert model._configure_ulysses_communicator(first_group) is first_communicator
        assert model._configure_ulysses_communicator(second_group) is second_communicator

    assert communicator_class.call_count == 2
    first_communicator.close.assert_called_once_with()


def test_base_model_offload_releases_ulysses_backend() -> None:
    from telefuser.core.base_model import BaseModel

    model = BaseModel()
    communicator = MagicMock()
    model._ulysses_communicator = communicator

    model.offload_device(pin_memory=False)

    communicator.close.assert_called_once_with()
