from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

CPU_TEST_MODULES = {
    "test_build_tools.py",
    "test_capabilities.py",
    "test_load_utils.py",
    "test_torch_defaults_reset.py",
}


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    markexpr = config.option.markexpr or ""
    if not collection_path.name.startswith("test_"):
        return False
    if "not gpu" in markexpr:
        return collection_path.name not in CPU_TEST_MODULES
    if markexpr.strip() == "smoke" or "not slow" in markexpr:
        return collection_path.name != "test_smoke.py"
    return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name in CPU_TEST_MODULES:
            continue
        item.add_marker(pytest.mark.gpu)
        if item.path.name != "test_smoke.py":
            item.add_marker(pytest.mark.slow)


# This fixture ensures the torch defaults don't get left in modified states between
# tests (e.g., when a test fails before restoring the original value), which
# can cause subsequent tests to fail.
@pytest.fixture(autouse=True)
def reset_torch_defaults() -> Iterator[None]:
    orig_default_device = torch.get_default_device()
    orig_default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig_default_dtype)
    torch.set_default_device(orig_default_device)
