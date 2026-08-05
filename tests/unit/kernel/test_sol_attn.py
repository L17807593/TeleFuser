from pathlib import Path

import pytest
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from telefuser.kernel.sol_attn.interface import _backend_for_arch, _validate_inputs, sol_attn


def test_sol_attn_is_packaged_with_telefuser() -> None:
    project_root = Path(__file__).parents[3]
    with (project_root / "pyproject.toml").open("rb") as config_file:
        package_data = tomllib.load(config_file)["tool"]["setuptools"]["package-data"]

    assert package_data["telefuser.kernel.sol_attn"] == [
        "THIRD_PARTY_NOTICES.md",
        "sm100/LICENSE.flash-attention",
    ]
    assert (project_root / "telefuser" / "kernel" / "sol_attn" / "__init__.py").is_file()
    assert not (project_root / "tf-kernel" / "tf_kernel" / "_sol_attn").exists()


def test_backend_selection_prefers_cute_and_falls_back_to_triton() -> None:
    assert _backend_for_arch((9, 0), cute_available=True) == "cute_sm90"
    assert _backend_for_arch((10, 0), cute_available=True) == "cute_sm100"
    assert _backend_for_arch((12, 0), cute_available=True) == "cute_sm120"
    assert _backend_for_arch((9, 0), cute_available=False) == "triton"
    assert _backend_for_arch((8, 0), cute_available=True) == "triton"

    with pytest.raises(RuntimeError, match="compute capability >= 8.0"):
        _backend_for_arch((7, 5), cute_available=False)


def test_input_validation_rejects_cpu_tensors() -> None:
    q = torch.randn(1, 64, 2, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same CUDA device"):
        _validate_inputs(q, q, q, "diag")


@pytest.mark.gpu
def test_sol_attn_dense_limit_matches_sdpa_on_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Sol-Attn SM90 correctness test requires H100")

    torch.manual_seed(0)
    q = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = sol_attn(q, k, v, tau=-1000.0, thresh_type="diag", kv_splits=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)

    torch.testing.assert_close(output, expected, atol=0.05, rtol=0.02)
