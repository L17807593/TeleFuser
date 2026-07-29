"""Load the tf-kernel extension matching the visible CUDA devices."""

import glob
import importlib.util
import logging
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Protocol

import torch

from tf_kernel.capabilities import (ArchitectureFamily, architecture_family,
                                    build_target_for_family, capability_label)

logger = logging.getLogger(__name__)


class _BuildInfo(Protocol):
    TORCH_VERSION: str
    TORCH_CUDA_VERSION: str
    CXX11_ABI: int
    TARGET_SM: str


def _public_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _visible_architecture_family() -> tuple[ArchitectureFamily, tuple[int, int]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise ImportError("tf-kernel is CUDA-only, but no CUDA device is available")

    capabilities = [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]
    families = {architecture_family(major, minor) for major, minor in capabilities}
    if len(families) != 1:
        labels = ", ".join(capability_label(major, minor) for major, minor in capabilities)
        raise ImportError(
            "tf-kernel does not support loading one process across heterogeneous "
            f"architecture families; visible devices: {labels}"
        )

    current_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    return families.pop(), current_capability


def _read_build_info() -> ModuleType | None:
    try:
        return import_module("tf_kernel._build_info")
    except ModuleNotFoundError as error:
        if error.name != "tf_kernel._build_info":
            raise
        logger.warning(
            "tf-kernel build metadata is missing; compatibility checks are unavailable "
            "for this legacy or source-tree installation"
        )
        return None


def _validate_build_info(build_info: _BuildInfo, family: ArchitectureFamily) -> None:
    errors: list[str] = []

    built_torch = str(build_info.TORCH_VERSION)
    if _public_version(built_torch) != _public_version(torch.__version__):
        errors.append(f"PyTorch {built_torch} was used to build the wheel, but runtime is {torch.__version__}")

    built_cuda = str(build_info.TORCH_CUDA_VERSION)
    runtime_cuda = torch.version.cuda or ""
    if built_cuda != runtime_cuda:
        runtime_cuda_label = runtime_cuda or "CPU-only"
        errors.append(f"PyTorch CUDA {built_cuda} was used to build the wheel, but runtime is {runtime_cuda_label}")

    built_abi = int(build_info.CXX11_ABI)
    runtime_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    if built_abi != runtime_abi:
        errors.append(f"C++11 ABI {built_abi} was used to build the wheel, but runtime is {runtime_abi}")

    target_sm = str(build_info.TARGET_SM).upper()
    expected_target = build_target_for_family(family)
    if target_sm not in {"ALL", expected_target}:
        errors.append(f"wheel target is {target_sm}, but the current GPU requires {expected_target}")

    if errors:
        details = "\n- ".join(errors)
        raise ImportError(f"tf-kernel wheel is incompatible with this runtime:\n- {details}")


def _compiled_extensions(pattern: str) -> list[Path]:
    return sorted(
        Path(file_path)
        for file_path in glob.glob(pattern)
        if Path(file_path).suffix in {".so", ".pyd", ".dll"}
    )


def _load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("common_ops", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an extension module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_architecture_specific_ops() -> ModuleType:
    """Load the extension for the single visible CUDA architecture family."""
    family, capability = _visible_architecture_family()
    build_info = _read_build_info()
    if build_info is not None:
        _validate_build_info(build_info, family)

    tf_kernel_dir = Path(__file__).parent
    pattern = str(tf_kernel_dir / family / "common_ops.*")
    matching_files = _compiled_extensions(pattern)
    label = capability_label(*capability)
    logger.debug("Loading tf-kernel %s extension for %s from %s", family, label, pattern)

    if len(matching_files) != 1:
        raise ImportError(
            f"Expected exactly one tf-kernel {family} extension for {label}, "
            f"but found {len(matching_files)} at {pattern}. Rebuild and install the wheel through the Makefile."
        )

    path = matching_files[0]
    try:
        module = _load_extension(path)
    except Exception as error:
        raise ImportError(f"Failed to load tf-kernel {family} extension at {path}: {error}") from error

    logger.debug("Loaded tf-kernel extension from %s", path)
    return module
