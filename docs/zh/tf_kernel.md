# tf-kernel 安装与使用

`tf-kernel` 是 TeleFuser 的可选 CUDA 扩展包，提供融合逐元素算子、量化 GEMM、SageAttention 和块稀疏
注意力内核。它位于本仓库的 `tf-kernel/` 目录，拥有独立的包元数据和版本。目前没有发布预编译包；
本地安装需要在配置好 CUDA/NVCC 的机器上从源码构建。

TeleFuser 不安装 `tf-kernel` 也可以运行：对于已经实现回退的算子，`telefuser.ops` 层会保留 PyTorch 原生
或 Triton 路径。当 Pipeline 需要 `tf-kernel` 提供的优化 CUDA 路径时再安装它。

!!! important "重要"

    TeleFuser 的模型代码仍然必须从 `telefuser.ops` 导入算子，不应直接依赖 `tf_kernel`。下文的直接导入
    仅用于独立使用、安装诊断和内核开发。

## 兼容性

| 组件 | 要求或目标 |
|------|------------|
| Python | 3.10 或更高版本 |
| PyTorch | `2.11.0`（`2.11.0+cu128` 这样的 CUDA local version 也满足要求） |
| CUDA Toolkit | 从源码编译要求 12.8 或更高版本 |
| CMake | 从源码编译要求 3.26 或更高版本 |
| GPU 目标 | SM80、SM90 和 SM100 |

具体可用内核取决于编译目标。FP4 内核要求 Blackwell（SM100 或更高）；在 Ampere 或 Hopper 上
`tf_kernel.FP4_AVAILABLE` 为 false，导入时不会打印提示。目前核心算子已在 Python 3.11、PyTorch
2.11.0+cu128、CUDA 12.8 和 H100（SM90a）组合上验证。其他目标和算子族用于生产前仍应在目标 GPU 上验证。

wheel 导入时会校验构建时记录的 PyTorch 公共版本、PyTorch CUDA 版本、C++11 ABI 和目标 GPU 架构族。
一个进程暴露不同架构族的 GPU 时会明确失败。SageAttention v2 自动分派目前仅支持 SM80、SM86、
SM89、SM90、SM120 和 SM121。

!!! note "H100 SageAttention 分派"

    H100 上的 `tf_kernel.sageattn()` 会选择已验证的 SM90 FP8 实现。配置 `SAGE_ATTN_2_8_8_SM90` 且
    `tf-kernel` 可用时，TeleFuser 会使用同一个内核。

## 从源码编译和安装

克隆 TeleFuser 单仓库，选择已经安装 PyTorch 2.11.0 的解释器，然后进入内核项目：

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser/tf-kernel
```

本地工作站可以自动检测当前 GPU：

```bash
make build-auto PYTHON=/path/to/venv/bin/python
```

本地构建独立于 TeleFuser 安装。Make 会构建带有正确 tag 的 wheel，并将其安装到 `PYTHON` 指定的解释器。
直接执行 `pip install .` 或 `pip install -e .` 会失败并提示改用 Make；当前不提供 pip 包索引安装。

本地构建使用 `linux_*` platform tag。容器构建只有在检查所有共享库的 GLIBC 符号版本满足策略后，
才可以使用 `manylinux_2_28` tag。

需要可复现的指定架构编译时：

```bash
make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere 和 Ada
make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper，包括 H100
make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell
```

H100 上限制主机资源占用的完整示例：

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
make build-sm90 \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  TF_KERNEL_COMPILE_THREADS=1
```

`make build` 会编译所有支持的目标。每个编译目标都会把 wheel 写入 `dist/`，添加包含 Torch/CUDA ABI
信息的合法 wheel build tag，并将其安装到 `PYTHON` 指定的解释器。首次编译需要联网获取固定版本的
CUTLASS、FlashInfer 和其他 CMake 依赖。

`MAX_JOBS` 控制并发编译任务数，`TF_KERNEL_COMPILE_THREADS` 控制每个任务内部使用的 NVCC 线程数。
在资源充足的机器上提高两者可以缩短编译时间，但它们的乘积也会增加 CPU 和内存压力。

## 验证安装

使用实际启动 TeleFuser 的同一个解释器运行：

```bash
python - <<'PY'
from pathlib import Path

import torch
import tf_kernel

print("tf-kernel:", tf_kernel.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("extension:", Path(tf_kernel.common_ops.__file__).resolve())

x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
y = tf_kernel.rmsnorm(x, weight)
assert y.shape == x.shape and torch.isfinite(y).all()
print("RMSNorm smoke test: OK")
PY
```

H100 专用 wheel 的 common extension 应从 `sm90` 包目录加载。还应运行 `python -m pip check` 检查环境中的
依赖冲突。

开发验证依次运行 `make test-cpu`、`make test-smoke` 和 `make test-wheel`。smoke 与 GPU target 会先将
wheel 安装到隔离的临时目录再收集用例。`make test` 是有界 GPU 测试；超过 6,000 条用例的
`make test-full` 应只在专用验证机器执行。

## 使用示例

TeleFuser 用户应调用公共 ops 层；它会在支持的 eager CUDA 路径选择 `tf-kernel`，同时保留框架回退：

```python
import torch

from telefuser.ops.activations import silu_and_mul

x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
y = silu_and_mul(x)  # 最后一维拆分成两个宽度为 1024 的张量。
assert y.shape == (4, 1024)
```

独立用户可以直接调用内核包：

```python
import torch
import tf_kernel

# RMSNorm
x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
y = tf_kernel.rmsnorm(x, weight, eps=1e-6)

# 已在 H100 验证的 SM90 FP8 SageAttention 路径。
# HND 布局：[batch, heads, sequence, head_dim]
q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
attn_output = tf_kernel.sageattn_qk_int8_pv_fp8_cuda_sm90(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    pv_accum_dtype="fp32+fp32",
)
```

per-token FP8 量化会写入调用者预先分配的输出张量：

```python
x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
x_scale = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
tf_kernel.tf_per_token_quant_fp8(x, x_q, x_scale)
```

更多底层接口契约请参考 `tf-kernel/docs/` 下的 API 文档和 `tf-kernel/tests/` 中的测试。

## 常见问题

### 使用了错误的 Python 环境

始终使用 `python -m pip`，并向 Make 传入 `PYTHON=/path/to/venv/bin/python`。可通过
`python -m pip show tf-kernel` 和 `python -c "import sys; print(sys.executable)"` 确认路径。

### PyTorch 被替换或依赖解析失败

`tf-kernel` 要求 PyTorch 2.11.0，因为已编译扩展与 PyTorch/CUDA ABI 绑定。如果其他包固定了不兼容的
PyTorch 版本，请在干净环境中安装 TeleFuser 和 `tf-kernel`。除非重新编译并针对替代版本完成验证，
否则不要绕过该约束。

### CMake 找不到 CUDA 或使用了错误 Toolkit

检查 `nvcc --version`，将 `CUDA_HOME` 指向 CUDA 12.8+ Toolkit，并在 `PATH` 中把 `$CUDA_HOME/bin` 放在
旧版 Toolkit 之前。PyTorch CUDA runtime 与所选 Toolkit 应保持 ABI 兼容。

### 扩展针对错误的 GPU 架构编译

在目标机器使用 `make build-auto` 重新编译，或显式选择 `build-sm80`、`build-sm90`、`build-sm100`。
指定架构 wheel 无法提供编译时未包含的内核。

### 验证 SM90 SageAttention 部署

SM90 专用内核已在 H100 上启用。构建 wheel 后，应在新的部署主机上运行带同步的 tf-kernel smoke test
和 TeleFuser 公共 ops GPU 集成测试。

### 编译耗尽 CPU 或内存

降低 `MAX_JOBS`、`CMAKE_BUILD_PARALLEL_LEVEL` 和 `TF_KERNEL_COMPILE_THREADS`。只编译一个 SM
架构也能显著降低编译时间和产物体积。
