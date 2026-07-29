# tf-kernel

English | [中文](README_zh.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)](https://pytorch.org/)

`tf-kernel` provides optimized CUDA operations for TeleFuser, including fused elementwise operations, quantized
GEMM, SageAttention, and block-sparse attention kernels. It supports SM80, SM90, and SM100 GPU families.

> [!IMPORTANT]
> No prebuilt tf-kernel package is currently published. Build and install the extension from source with this
> directory's Makefile. Direct `pip install .` and `pip install -e .` source builds are intentionally rejected.

## Requirements

- Python 3.10 or newer
- PyTorch 2.11.0
- CUDA Toolkit 12.8 or newer
- CMake 3.26 or newer
- An NVIDIA GPU in the SM80, SM90, or SM100 family

## Build and Install

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser/tf-kernel
make build-auto PYTHON=/path/to/venv/bin/python
```

`build-auto` detects the local GPU architecture. The Makefile builds a correctly tagged wheel under `dist/` and
installs it into the interpreter selected by `PYTHON`. This does not install or depend on the TeleFuser Python package.
Local builds use a `linux_*` platform tag. Only the container build may emit `manylinux_2_28` after checking the
wheel's ELF symbol versions against that policy.

Use an explicit architecture target for reproducible builds:

| Target | GPU family |
|--------|------------|
| `make build-sm80` | Ampere and Ada |
| `make build-sm90` | Hopper, including H100 |
| `make build-sm100` | Blackwell |
| `make build` | All supported architectures |

For example:

```bash
make build-sm90 PYTHON=/path/to/venv/bin/python
```

## Parallel Compilation

`MAX_JOBS` controls concurrent build jobs. `TF_KERNEL_COMPILE_THREADS` controls NVCC threads within each job:

```bash
make build-auto \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=16 \
  TF_KERNEL_COMPILE_THREADS=4
```

Higher values can reduce build time on a sufficiently provisioned host, but also increase CPU and memory pressure.
For a resource-constrained build, start with `MAX_JOBS=2 TF_KERNEL_COMPILE_THREADS=1`.

## Verify

Run the smoke test with the same interpreter passed to Make:

```bash
/path/to/venv/bin/python - <<'PY'
from pathlib import Path

import torch
import tf_kernel

print("tf-kernel:", tf_kernel.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name())
print("extension:", Path(tf_kernel.common_ops.__file__).resolve())

x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
assert torch.isfinite(tf_kernel.rmsnorm(x, weight)).all()
print("RMSNorm smoke test: OK")
PY
```

FP4 operators require SM100 or newer, so FP4 is unavailable on Ampere and Hopper. On H100, `sageattn()`
and TeleFuser `SAGE_ATTN_2_8_8_SM90` use the validated SM90 FP8 implementation when the SM90 wheel is installed.
At import time, the wheel verifies the PyTorch public version, PyTorch CUDA version, C++11 ABI, and target GPU family
recorded during its build. A process must expose GPUs from only one architecture family. SageAttention v2 dispatch is
currently enabled for SM80, SM86, SM89, SM90, SM120, and SM121; other compute capabilities fail explicitly instead of
selecting an unvalidated backend.

## Development

```bash
make test-cpu PYTHON=/path/to/venv/bin/python
make test-smoke PYTHON=/path/to/venv/bin/python
make test PYTHON=/path/to/venv/bin/python       # bounded GPU suite
make test-full PYTHON=/path/to/venv/bin/python  # exhaustive matrix
make test-wheel PYTHON=/path/to/venv/bin/python
make format-check PYTHON=/path/to/venv/bin/python
make docs PYTHON=/path/to/venv/bin/python
```

GPU targets install the wheel into an isolated temporary directory before collection, so tests cannot accidentally
import `tf_kernel` from the source checkout or another environment.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and future release procedures. See the
[full installation and usage guide](../docs/en/tf_kernel.md) for compatibility, API examples, and troubleshooting.
