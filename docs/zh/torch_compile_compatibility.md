# PyTorch `torch.compile` 推理兼容性指南

本指南介绍编写高度兼容 `torch.compile` 的 PyTorch 推理代码的最佳实践。

## 引言

`torch.compile` 是 PyTorch 2.0 引入的即时（JIT）编译器，它能够捕获模型的计算图并进行内核融合、内存规划等优化，从而显著提升模型执行速度。要充分发挥其性能优势，模型的 `forward` 代码必须遵循特定的规范。

**核心目标：编写"纯 PyTorch"风格的 `forward` 函数，消除所有会导致图断裂的 Python 运行时交互。**

## 核心原则：避免图断裂（Graph Break）

当编译器遇到无法静态分析的 Python 动态特性时，就会发生**图断裂**，即计算图被切割，编译器被迫回退到慢速的 Python 解释器执行模式。

基本原则如下：
- **张量优先**：尽量使用 PyTorch 的张量操作（如 `torch.where`、`torch.gather`）替代 Python 控制流
- **避免外部库**：在 `forward` 中不要调用 `numpy`、`scipy` 或 `pandas` 函数
- **稳定输入**：保持输入张量的数据类型（`dtype`）、设备（`device`）和形状（`shape`）的相对稳定
- **严格模式开发**：在开发阶段使用 `torch.compile(model, fullgraph=True)`，任何图断裂都会导致程序报错

## 编写兼容 `torch.compile` 的 `forward` 指南

### 数据结构处理：列表（List）与字典（Dict）

动态数据结构是导致图断裂的常见原因。

| 数据类型 | ❌ 不推荐做法（可能导致断裂/重编译） | ✅ 推荐做法 |
|:---------|:------------------------------------|:-----------|
| **列表 (List)** | - 在 `forward` 内使用 `list.append()`、`list.pop()`、`list.sort()`<br>- 列表中包含的张量数量动态变化 | - 作为简单的输入/输出容器使用<br>- 若需动态拼接，使用 `torch.cat` 替代循环追加<br>- 使用元组（Tuple）作为返回容器更安全 |
| **字典 (Dict)** | - 将复杂的嵌套字典作为 `forward` 的输入参数<br>- 在 `forward` 内遍历字典的键值对进行逻辑判断 | - **在进入模型前解包**：在 `DataLoader` 的 `collate_fn` 中就将字典拍平为张量列表或具名元组<br>- 在 `forward` 开头显式提取所需张量：`x = input_dict['image']` |

### 控制流处理：条件语句（If）与循环（For）

控制流的兼容性取决于判断条件是否依赖于张量的**值**。

| 语句类型 | ❌ 动态依赖（导致图断裂） | ✅ 静态依赖（编译友好） |
|:---------|:-------------------------|:-----------------------|
| **If 条件** | `if x.sum() > 0:` <br> `if x.shape[0] > 10:` | `if self.training:` <br> `if self.config.use_bias:` |
| **For 循环** | `for i in range(x.shape[0]):` <br>（若每次调用形状变化，触发重编译） | `for i in range(10):` <br>（迭代次数为常量） |

**替代方案**：
- 对于依赖张量值的条件选择，使用 **`torch.where(condition, a, b)`**
- 若必须处理动态形状的循环，可考虑启用动态形状支持：`torch.compile(model, dynamic=True)`，但这会牺牲部分性能

### 减少不必要的重编译（Recompilation）

即使没有图断裂，频繁的**重编译**也会抵消加速效果。每次函数调用时，若编译器认为"图结构发生了变化"，就会触发重编译。

**主要诱因与解决方案**：

1. **变化的张量形状**：
   - **诱因**：本次输入是 `(1, 3, 224, 224)`，下次是 `(1, 3, 256, 256)`
   - **对策**：通过填充（Padding）固定尺寸，或使用 `torch.compile(dynamic=True)` 处理特定维度变化

2. **变化的非张量参数**：
   - **诱因**：`forward(self, x, multiplier)` 中的 `multiplier` 是 `float` 且频繁变值
   - **对策**：将标量包装为张量传入：`multiplier_tensor = torch.tensor(multiplier, device=x.device)`。编译器对张量的值变化容忍度更高

3. **变化的设备或数据类型**：
   - **诱因**：有时在 CPU 上跑，有时在 CUDA 上跑
   - **对策**：确保输入始终在同一设备、同一 `dtype` 下

## 集成自定义算子（CUDA / Triton Kernel）

当使用手写的 CUDA 或 Triton 内核时，必须将其注册为 PyTorch 自定义算子，`torch.compile` 才能识别并将其视为一个"黑盒"算子。

### 标准集成步骤

使用 `torch.library.custom_op` 装饰器进行注册，**关键是要提供 `impl_abstract` 函数**。

```python
import torch
from torch.library import custom_op

# 1. 定义内核调用入口
def my_triton_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ... 实际调用 Triton 内核的代码 ...
    return output

# 2. 注册为 PyTorch 自定义算子
@custom_op("mylib::my_fast_op", mutates_args=())
def my_fast_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return my_triton_kernel(a, b)

# 3. 必须实现抽象推断函数（FakeTensor 支持）
@my_fast_op.impl_abstract("mylib::my_fast_op")
def my_fast_op_abstract(a, b):
    # 只需返回描述输出形状、dtype 的空张量
    return torch.empty_like(a)
```

### 在模型中使用

```python
class MyModel(nn.Module):
    def forward(self, x):
        # 通过 torch.ops 命名空间调用
        return torch.ops.mylib.my_fast_op(x, x)

model = MyModel()
compiled_model = torch.compile(model, fullgraph=True)
```

### 重要注意事项

- **`impl_abstract` 是必须的**：没有它，`torch.compile` 在追踪 FakeTensor 时会失败
- **Triton 专用 API**：对于纯 Triton 内核，可以关注实验性 API `torch._library.triton.triton_op`，它可能简化集成流程

## 性能权衡：Triton 算子 vs. 原生 PyTorch 编译

这是一个常见的抉择：是将逻辑写成 Triton 算子再注册，还是直接用 PyTorch 原生 API 让 `torch.compile` 去融合？

### 算子内部优化能力对比

| 场景 | Triton 自定义算子 | PyTorch 原生 + `compile` |
|:-----|:------------------|:-------------------------|
| **高计算密度（Compute-Bound）** <br>（如 FlashAttention、复杂激活函数） | ✅ **显著更快**。手动控制 SRAM 和流水线，可达 1.5x-3x 提升 | ⚠️ 受限于基础算子库，无法凭空产生极致融合 |
| **低计算密度（Overhead-Bound）** <br>（如 `x+1`, `x*scale+bias` 等点操作） | ⚠️ 手写 Triton 繁琐且易出错，性能提升有限 | ✅ **极优**。Inductor 后端会自动进行垂直/水平融合，消除 Python 开销 |

### 全局图优化能力对比

将自定义算子注册后，`torch.compile` 会将其视为不透明的"黑盒"。

| 全局优化类型 | Triton 自定义算子 | PyTorch 原生算子 |
|:-------------|:------------------|:-----------------|
| **跨算子融合** | ❌ **阻断**。无法与前后相邻的 PyTorch 操作融合 | ✅ **支持**。可将前后操作融合为单个 CUDA 内核 |
| **内存布局传播** | ⚠️ 需手动适配 `channels_last` 等格式 | ✅ **自动处理**。自动选择最优显存步幅 |

### 决策指南

```text
该逻辑是否属于行业经典优化范本？
    │
    ├─ 是（如 FlashAttention, RMSNorm, Fused MLP）
    │      └─> 【手写 Triton 并注册 Custom Op】
    │
    └─ 否
           │
           ├─ 该逻辑包含复杂 Python 控制流（必然图断裂）？
           │      └─> 【手写 Triton 保平安】
           │
           └─ 该逻辑只是基础算子的排列组合？
                  └─> 【原生 PyTorch + torch.compile】（零开发成本，且不阻断全局融合）
```

## TeleFuser 混合策略（实践案例）

TeleFuser 根据算子特性和执行流程实现了**混合策略**处理 torch.compile 兼容性：

### 不同算子类型的策略

| 算子类型 | 策略 | 原因 |
|:---------|:-----|:-----|
| **Attention**（高计算密度） | `@torch.compiler.disable` | FlashAttention/SageAttention 性能远优于原生 PyTorch；融合收益有限 |
| **RoPE**（中等计算密度） | `@torch.compiler.disable` | Triton 内核优于原生实现；后续 Attention 已阻断融合 |
| **RMSNorm/LayerNorm**（低计算密度） | 编译时使用原生实现 | Overhead-bound；Inductor 可与相邻算子融合 |
| **modulate**（点操作） | 编译时使用原生实现 | 计算量极小；Inductor 自动融合最优 |

### 执行流程分析

```
Linear → RMSNorm(q_norm) → RoPE → Attention
                      ↑        ↑         ↑
               原生+融合    Triton    Triton (disabled)
```

关键洞察：由于 Attention 使用了 `@torch.compiler.disable`，RoPE 之后任何融合都被阻断。因此：
- RoPE 应使用 Triton 内核（反正没有融合机会）
- RMSNorm 应使用原生实现（可能与前序 Linear 融合）

### 实现示例

```python
# Attention - 始终使用优化内核，禁用编译
@torch.compiler.disable
def attention(q, k, v, **kwargs):
    return flash_attn2(q, k, v, **kwargs)

# RoPE - 使用 Triton 内核，禁用编译
@torch.compiler.disable
def apply_rotary_emb(x, cos, sin):
    return apply_rotary_embedding(x, cos, sin)  # Triton

# RMSNorm - 编译感知分发
class RMSNorm(CustomOp):
    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)  # 允许融合
        return self.forward_cuda(x)  # Eager 模式用 Triton
```

## 推理场景特有优化

### 使用 `torch.inference_mode`

推理时使用 `torch.inference_mode` 比 `no_grad` 更快：

```python
# 推荐用于推理
with torch.inference_mode():
    output = compiled_model(input)

# 或在模型类中标记
model.eval()
compiled_model = torch.compile(model)
```

### CUDA Graph 固定形状优化

对于固定形状的推理，可启用 CUDA Graph 实现极致优化：

```python
# 内部使用 CUDA Graph 减少 kernel launch 开销
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### 编译模式选择

```python
# 不同编译模式及其适用场景
torch.compile(model)                        # 默认：自动选择
torch.compile(model, mode="default")        # 平衡编译时间和性能
torch.compile(model, mode="reduce-overhead")  # 减少 Python 开销，适合小批量推理
torch.compile(model, mode="max-autotune")   # 最大优化，编译时间长，适合固定形状
```

### 服务化部署最佳实践

**生产环境 Warmup**：
```python
# 首次推理会有编译开销
model = torch.compile(model)

# 生产服务前先 warmup
with torch.inference_mode():
    _ = model(dummy_input)  # 触发编译

# 现在后续调用都是快速的
output = model(real_input)
```

**编译产物缓存**：
```python
import torch._inductor.config as inductor_config

# 设置缓存目录
inductor_config.cache_dir = "/path/to/cache"

# 编译产物可跨会话持久化
compiled_model = torch.compile(model)
```

## 调试与性能分析工具

当遇到性能瓶颈或编译失败时，以下工具能帮你定位问题：

| 工具 / 环境变量 | 用途 |
|:--------------- |:-----|
| `TORCH_LOGS=recompiles` | 在终端打印每次重编译的**具体原因**（如形状变化、标量值变化）。是定位性能问题的首选 |
| `torch.compile(..., fullgraph=True)` | 强制全图编译。一旦有 Python 图断裂即报错，用于开发阶段严格自检 |
| `torch._dynamo.explain(model)(x)` | 打印详细的图断裂报告，指出具体是哪一行代码导致的断裂 |
| `torch.profiler` | 结合 `torch.compile` 使用，查看融合后的内核执行情况 |

## 核心技巧速查表

| 问题现象 | 诊断 / 解决方案 |
|:---------|:----------------|
| 编译后的模型比不编译还慢 | 使用 `TORCH_LOGS=recompiles` 检查是否频繁重编译。检查输入形状或标量参数是否变化 |
| 报错 `Graph break in user code` | 在 `forward` 中使用了依赖张量值的 `if` 或 `for`。改用 `torch.where` 或固定形状 |
| 自定义 CUDA 内核报错 `FakeTensor` | 未提供 `impl_abstract` 函数。补充 `@op.impl_abstract` 定义 |
| 列表操作导致警告 | 避免在 `forward` 内动态修改列表长度。将动态拼接逻辑移至张量操作（如 `torch.cat`） |

## 总结

编写高度兼容 `torch.compile` 的代码，本质上是一场从 **Python 动态特性**向 **静态计算图描述** 的思维转变。

- **短期收益**：避免 `if` 判断张量值、固定输入形状、注册自定义算子
- **长期收益**：模型推理速度可提升 30%-200%

遵循本指南的原则，你可以构建出既保留 Python 开发灵活性，又能享受编译器极致性能优化的 PyTorch 模型。

## 相关文档

- [Ops 模块文档](./ops.md) - TeleFuser 自定义算子实现
- [Profiler 指南](./profiler.md) - 性能分析工具
- [Attention 实现](./attention.md) - 注意力模块优化
