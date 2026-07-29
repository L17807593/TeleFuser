"""Validate the interpreter and CUDA toolchain used to build tf-kernel."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

SUPPORTED_TARGETS = {"AUTO", "ALL", "SM80", "SM90", "SM100"}
REQUIRED_TORCH_VERSION = "2.11.0"
MINIMUM_CUDA_VERSION = (12, 8)


def _parse_nvcc_version(output: str) -> tuple[int, int] | None:
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _find_nvcc() -> Path | None:
    path_nvcc = shutil.which("nvcc")
    if path_nvcc is not None:
        return Path(path_nvcc)

    candidates = [Path("/usr/local/cuda/bin/nvcc")]
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_build_environment(target_sm: str) -> list[str]:
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10 or newer is required, found {sys.version.split()[0]}")

    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    if torch_version != REQUIRED_TORCH_VERSION:
        errors.append(f"PyTorch {REQUIRED_TORCH_VERSION} is required, found {torch.__version__}")
    if torch.version.cuda is None:
        errors.append("The selected PyTorch installation does not provide CUDA support")

    normalized_target = target_sm.upper()
    if normalized_target not in SUPPORTED_TARGETS:
        errors.append(
            f"Unsupported target SM {target_sm!r}; expected one of {', '.join(sorted(SUPPORTED_TARGETS))}"
        )

    nvcc = _find_nvcc()
    if nvcc is None:
        errors.append("nvcc was not found on PATH or under /usr/local/cuda*")
    else:
        result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
        version = _parse_nvcc_version(result.stdout + result.stderr)
        if result.returncode != 0 or version is None:
            errors.append(f"Could not determine the CUDA toolkit version from {nvcc}")
        elif version < MINIMUM_CUDA_VERSION:
            errors.append(f"CUDA toolkit 12.8 or newer is required, found {version[0]}.{version[1]}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-sm", required=True)
    args = parser.parse_args()

    errors = validate_build_environment(args.target_sm)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "Build environment validated:",
        f"python={sys.version.split()[0]}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cxx11_abi={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f"target_sm={args.target_sm.upper()}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
