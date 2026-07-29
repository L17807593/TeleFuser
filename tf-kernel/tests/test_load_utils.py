import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package_root = Path(__file__).parents[1] / "tf_kernel"
    package = ModuleType("tf_kernel")
    package.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "tf_kernel", package)
    capabilities = _load_module("tf_kernel.capabilities", package_root / "capabilities.py")
    monkeypatch.setitem(sys.modules, "tf_kernel.capabilities", capabilities)
    return _load_module("tf_kernel.load_utils", package_root / "load_utils.py")


def test_visible_architecture_family_rejects_cpu_runtime(
    load_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load_utils.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ImportError, match="CUDA-only"):
        load_utils._visible_architecture_family()


def test_visible_architecture_family_rejects_heterogeneous_gpus(
    load_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    capabilities = [(8, 0), (9, 0)]
    monkeypatch.setattr(load_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(load_utils.torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(load_utils.torch.cuda, "get_device_capability", lambda index: capabilities[index])
    with pytest.raises(ImportError, match="heterogeneous"):
        load_utils._visible_architecture_family()


def test_validate_build_info_accepts_matching_runtime(load_utils: ModuleType) -> None:
    build_info = SimpleNamespace(
        TORCH_VERSION=torch.__version__,
        TORCH_CUDA_VERSION=torch.version.cuda,
        CXX11_ABI=int(torch._C._GLIBCXX_USE_CXX11_ABI),
        TARGET_SM="ALL",
    )
    load_utils._validate_build_info(build_info, "sm90")


def test_validate_build_info_reports_all_mismatches(load_utils: ModuleType) -> None:
    runtime_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    build_info = SimpleNamespace(
        TORCH_VERSION="0.0.0",
        TORCH_CUDA_VERSION="0.0",
        CXX11_ABI=1 - runtime_abi,
        TARGET_SM="SM80",
    )

    with pytest.raises(ImportError) as error:
        load_utils._validate_build_info(build_info, "sm90")

    message = str(error.value)
    assert "PyTorch 0.0.0" in message
    assert "PyTorch CUDA 0.0" in message
    assert "C++11 ABI" in message
    assert "wheel target is SM80" in message
