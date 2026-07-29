import importlib.util
from pathlib import Path


def _load_validate_build_env():
    script = Path(__file__).parents[1] / "scripts" / "validate_build_env.py"
    spec = importlib.util.spec_from_file_location("validate_build_env", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_nvcc_version():
    module = _load_validate_build_env()

    assert module._parse_nvcc_version("Cuda compilation tools, release 12.8, V12.8.61") == (12, 8)
    assert module._parse_nvcc_version("not an nvcc response") is None


def test_invalid_target_is_rejected():
    module = _load_validate_build_env()

    errors = module.validate_build_environment("SM75")

    assert any("Unsupported target SM" in error for error in errors)

def test_cmake_uses_cudatoolkit_version():
    cmake = (Path(__file__).parents[1] / "CMakeLists.txt").read_text()

    assert "${CUDAToolkit_VERSION}" in cmake
    assert "${CUDA_VERSION}" not in cmake


def test_sm90_target_disables_relocatable_device_code():
    cmake = (Path(__file__).parents[1] / "CMakeLists.txt").read_text()
    sm90_block = cmake.split(
        "# =========================== Common SM90 Build", maxsplit=1
    )[1].split("# =========================== Common SM80 Build", maxsplit=1)[0]

    assert "CUDA_SEPARABLE_COMPILATION OFF" in sm90_block
    assert "tf_kernel_configure_device_link(common_ops_sm90_build" not in sm90_block
