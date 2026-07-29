import importlib.util
from pathlib import Path

import pytest


def _load_capabilities_module():
    module_path = Path(__file__).parents[1] / "tf_kernel" / "capabilities.py"
    spec = importlib.util.spec_from_file_location("tf_kernel_capabilities_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capabilities = _load_capabilities_module()


@pytest.mark.parametrize(
    ("capability", "family"),
    [
        ((8, 0), "sm80"),
        ((8, 9), "sm80"),
        ((9, 0), "sm90"),
        ((10, 0), "sm100"),
        ((12, 1), "sm100"),
    ],
)
def test_architecture_family(capability: tuple[int, int], family: str) -> None:
    assert capabilities.architecture_family(*capability) == family


@pytest.mark.parametrize("capability", [(7, 5), (9, 1)])
def test_architecture_family_rejects_unsupported_capabilities(capability: tuple[int, int]) -> None:
    with pytest.raises(capabilities.UnsupportedArchitectureError):
        capabilities.architecture_family(*capability)


@pytest.mark.parametrize(
    ("capability", "backend"),
    [
        ((8, 0), "cuda_fp16"),
        ((8, 6), "triton_fp16"),
        ((8, 9), "cuda_fp8_thread"),
        ((9, 0), "cuda_fp8_sm90"),
        ((12, 0), "cuda_fp8_warp"),
        ((12, 1), "cuda_fp8_warp"),
    ],
)
def test_sage_attention_backend(capability: tuple[int, int], backend: str) -> None:
    assert capabilities.sage_attention_backend(*capability) == backend


@pytest.mark.parametrize("capability", [(8, 7), (10, 0), (10, 3), (11, 0)])
def test_sage_attention_backend_rejects_unvalidated_capabilities(capability: tuple[int, int]) -> None:
    with pytest.raises(capabilities.UnsupportedArchitectureError, match="no validated backend"):
        capabilities.sage_attention_backend(*capability)
