# tf-kernel

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)](https://pytorch.org/)

`tf-kernel` 为 TeleFuser 提供优化的 CUDA 算子，包括融合逐元素运算、量化 GEMM、SageAttention 和块稀疏
注意力内核，支持 SM80、SM90 和 SM100 GPU 架构。

> [!IMPORTANT]
> 目前没有发布 tf-kernel 预编译包。请使用本目录的 Makefile 从源码构建和安装扩展。项目会主动拒绝
> `pip install .` 和 `pip install -e .` 直接源码构建。

## 环境要求

- Python 3.10 或更高版本
- PyTorch 2.11.0
- CUDA Toolkit 12.8 或更高版本
- CMake 3.26 或更高版本
- SM80、SM90 或 SM100 系列 NVIDIA GPU

## 编译和安装

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser/tf-kernel
make build-auto PYTHON=/path/to/venv/bin/python
```

`build-auto` 会检测本机 GPU 架构。Makefile 在 `dist/` 下构建带有正确 tag 的 wheel，并安装到 `PYTHON`
指定的解释器。该流程不会安装或依赖 TeleFuser Python 包。

如需可复现的指定架构构建，请使用对应 target：

| Target | GPU 架构 |
|--------|----------|
| `make build-sm80` | Ampere 和 Ada |
| `make build-sm90` | Hopper，包括 H100 |
| `make build-sm100` | Blackwell |
| `make build` | 所有支持的架构 |

例如：

```bash
make build-sm90 PYTHON=/path/to/venv/bin/python
```

## 并行编译

`MAX_JOBS` 控制并发编译任务数，`TF_KERNEL_COMPILE_THREADS` 控制每个任务内部的 NVCC 线程数：

```bash
make build-auto \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=16 \
  TF_KERNEL_COMPILE_THREADS=4
```

资源充足时提高参数可以缩短编译时间，但也会增加 CPU 和内存压力。资源受限时可以从
`MAX_JOBS=2 TF_KERNEL_COMPILE_THREADS=1` 开始。

## 验证

使用传给 Make 的同一解释器执行 smoke test：

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

FP4 算子要求 SM100 或更高架构，因此在 Ampere 和 Hopper 上出现 FP4 不可用提示属于预期行为。当前验证的
H100 build 在架构选择的 SM90 SageAttention 路径存在已知错误；专项 GPU 测试通过前请使用其他 TeleFuser
注意力后端。

## 开发

```bash
make test PYTHON=/path/to/venv/bin/python
make format-check PYTHON=/path/to/venv/bin/python
make docs PYTHON=/path/to/venv/bin/python
```

开发和未来发布流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。兼容性、API 示例和常见问题见
[完整安装与使用指南](../docs/zh/tf_kernel.md)。
