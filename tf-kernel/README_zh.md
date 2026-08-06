# tf-kernel

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)](https://pytorch.org/)

`tf-kernel` 为 TeleFuser 提供优化的 CUDA 算子，包括融合逐元素运算、量化 GEMM、SageAttention 和块稀疏
注意力内核，支持 SM80、SM90 和 SM100 GPU 架构。

> [!IMPORTANT]
> 项目不向公共包索引发布 tf-kernel 预编译 wheel 或源码分发包。请使用本目录的 Makefile 从源码构建和
> 安装扩展。项目会主动拒绝 `pip install .` 和 `pip install -e .` 直接源码构建。

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
本机构建使用 `linux_*` platform tag。只有容器构建在检查 wheel 的 ELF 符号版本满足对应基线后，才会标记
为 `manylinux_2_28`。

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

## 分发已构建的 wheel

源码 commit、tf-kernel 版本、PyTorch 版本、PyTorch CUDA 版本、C++11 ABI、目标 SM 架构族、CPU 架构和
Linux/GLIBC 基线兼容时，可以将本地构建的 wheel 复制到其他主机，或保存到受控的制品库。wheel 导入时
会校验能够在运行时检查的兼容性信息。

SM80、SM90 和 SM100 的指定架构构建目前会生成相同的文件名。必须使用不同的制品路径隔离，并安装
明确选定的文件；不要通过同一个 simple package index 暴露多个目标 SM 变体，因为 pip 无法根据 GPU
架构选择 wheel。例如：

```text
tf-kernel/0.1.0/torch2.11.0-cu128/linux-x86_64/sm90/
```

分发前应运行 `make test-wheel`、`make test-smoke`，并将 `sha256sum dist/*.whl`、源码 commit 和测试结果
与制品一起保存。在已经安装匹配 PyTorch build 的环境中安装选定制品：

```bash
python -m pip install /path/to/tf_kernel-*.whl --no-deps
python -m pip check
```

不要手工把本地 `linux_*` wheel 改标为 `manylinux`；应在预期部署基线环境上重新构建。完整制品清单和
目标主机验证流程见安装指南。

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

FP4 算子要求 SM100 或更高架构，因此 Ampere 和 Hopper 不支持 FP4。安装 SM90 wheel 后，H100 上的
`sageattn()` 和 TeleFuser `SAGE_ATTN_2_8_8_SM90` 都会使用已验证的 SM90 FP8 实现。
wheel 导入时会校验构建时记录的 PyTorch 公共版本、PyTorch CUDA 版本、C++11 ABI 和目标 GPU 架构族。
一个进程只能暴露同一架构族的 GPU。SageAttention v2 当前仅对 SM80、SM86、SM89、SM90、SM120 和
SM121 启用自动分派；其他 compute capability 会明确报错，不会进入未经验证的后端。

## 开发

```bash
make test-cpu PYTHON=/path/to/venv/bin/python
make test-smoke PYTHON=/path/to/venv/bin/python
make test PYTHON=/path/to/venv/bin/python       # 有界 GPU 测试
make test-full PYTHON=/path/to/venv/bin/python  # 全量测试矩阵
make test-wheel PYTHON=/path/to/venv/bin/python
make format-check PYTHON=/path/to/venv/bin/python
make docs PYTHON=/path/to/venv/bin/python
```

GPU 测试会先把 wheel 安装到隔离的临时目录，再开始收集用例，避免误从源码目录或其他环境导入
`tf_kernel`。

开发和 wheel 分发策略见 [CONTRIBUTING.md](CONTRIBUTING.md)。兼容性、API 示例和常见问题见
[完整安装与使用指南](../docs/zh/tf_kernel.md)。
