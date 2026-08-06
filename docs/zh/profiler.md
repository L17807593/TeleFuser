# Profiler 性能分析系统

TeleFuser 提供了三层渐进式性能分析系统，基于 PyTorch Profiler 构建，支持分布式环境下的性能调试和内存追踪。

## 功能特性

- **三层渐进式分析** - Stage 时序 → Kernel 分析 → NCU 深度分析
- **上下文管理器和装饰器** - 灵活的使用方式
- **同步和异步支持** - 同时支持同步和异步函数
- **分布式感知** - 自动处理多 rank 性能分析
- **内存追踪** - 监控峰值内存分配
- **Stage I/O Signature 捕获** - 记录 tensor shape 用于独立分析
- **Stage Bench Harness** - 使用 mock input 独立分析各 stage（避免 DiT 40+ 次迭代）
- **自动生成测试脚本** - 无需 harness 基础设施即可复现分析
- **有序输出目录** - `work_dirs/profiler_output/{pipeline_name}/{YYYYMMDD_HHMM}`

## Stage I/O Signature（Layer 1）

当使用 `TELEFUSER_PROFILE_DEBUG=true` 进行性能分析时，profiler 自动捕获每个 stage 的输入输出 tensor signature。这使 Layer 2 独立分析无需运行完整 pipeline。

### 捕获信息

每个 stage 的 signature 包括：
- 输入 tensor shape、dtype、device
- 输出 tensor shape（如适用）
- 非 tensor 参数（int、float、str）作为 metadata

### 输出位置

默认输出目录结构：
```
work_dirs/profiler_output/{TELEFUSER_PIPELINE_NAME}/{YYYYMMDD_HHMM}/
├── timing.json                    # Layer 1 stage 时序报告
├── timing_io_signature.json       # I/O signature 用于 harness
├── denoise_trace.json.gz          # Layer 2 Chrome trace（单次迭代）
├── denoise_breakdown.json         # Layer 2 top 50 kernels
└── profile_denoise.py             # 自动生成的测试脚本
```

设置 `TELEFUSER_PIPELINE_NAME` 环境变量来组织输出。

### Signature 格式

```json
{
  "request_id": "req_20260402_...",
  "timestamp": "2026-04-02T...",
  "stages": {
    "denoise": {
      "stage_name": "denoise",
      "input_signatures": {
        "latents": {"shape": [1, 16, 21, 60, 104], "dtype": "bfloat16", "device": "cuda:0"},
        "prompt_emb_posi": {"shape": [1, 512, 4096], "dtype": "bfloat16", "device": "cuda:0"},
        "cfg_scale": 5.0
      },
      "output_signature": {...},
      "metadata": {"num_inference_steps": 40, "sigma_shift": 8.0}
    }
  }
}
```

## Stage Bench Harness（Layer 2）

`StageBenchHarness` 使用捕获的 I/O signature 实现独立 stage 分析。这对需要迭代 40+ 次的 DiT 模型特别有用，完整 pipeline 分析会产生巨大的 trace 文件。

### 优势

| 方面 | 完整 Pipeline | 独立 Harness |
|------|---------------|--------------|
| Trace 大小 | 100MB+ | <10MB |
| 迭代次数 | 40+（冗余） | 1（干净） |
| 内存占用 | 完整 pipeline | 仅 stage |
| 分析难度 | 难以隔离 | 清晰视图 |
| 可复现性 | 手动配置 | 自动生成脚本 |

### 使用方法

```python
from telefuser.utils.stage_bench_harness import StageBenchHarness, HarnessConfig

# 从 signature 文件创建 harness
config = HarnessConfig(
    warmup=1,
    profile_steps=1,
    # output_dir 默认为 work_dirs/profiler_output/{pipeline_name}/{date}
)

harness = StageBenchHarness.from_signature_file(
    signature_path="work_dirs/profiler_output/wan21_t2v/20260402/timing_io_signature.json",
    stage_name="denoise",
    stage_instance=pipeline.denoise_stage,  # 传入已加载的 stage
    config=config,
)

# 设置并分析
harness.setup()
results = harness.profile()

# 输出文件：
# - denoise_trace.json.gz（Chrome trace，单次迭代）
# - denoise_breakdown.json（top 50 kernels）
# - profile_denoise.py（自动生成的测试脚本）
```

### 动态单步执行

对于包含内部循环的 DiT stage，harness 通过检测 `dit` 和 `scheduler` 属性自动创建单步函数。无需修改 stage 代码。

单步逻辑从 denoising 循环中提取一次迭代：
1. 使用最小步数（2）设置 scheduler
2. 取第一个 timestep
3. 运行单次 forward + scheduler step

### Kernel Breakdown 输出

breakdown JSON 包含按时间排序的 top 50 kernels（无分类）：

```json
{
  "name": "denoise",
  "total_kernel_time_ms": 150.0,
  "num_kernels": 200,
  "top_kernels": [
    {"name": "flash_attn_fwd", "ms": 75.0, "cuda_ms": 75.0, "cpu_ms": 0.5},
    {"name": "ampere_fp16_s1688gemm", "ms": 50.0, "cuda_ms": 50.0, "cpu_ms": 0.3},
    {"name": "fused_add_rms_norm", "ms": 10.0, "cuda_ms": 10.0, "cpu_ms": 0.1}
  ]
}
```

### 生成的测试脚本

Harness 生成独立 Python 脚本用于可复现分析：

```python
# profile_denoise.py - 由 harness 生成
# 包含：
# - 基于 I/O signature 的输入 tensor 创建
# - DiT stage 的单步执行逻辑
# - 带 warmup 和 trace 导出的分析函数
```

使用已加载的 stage 实例运行生成的脚本进行独立分析。

## 快速开始

### 基本用法

```python
from telefuser.utils.profiler import ProfilingContext

# 作为上下文管理器
with ProfilingContext("my_operation"):
    # 你的代码
    result = model(input_data)

# 作为装饰器
@ProfilingContext("my_function")
def process_data(data):
    return model(data)

# 作为异步装饰器
@ProfilingContext("async_operation")
async def process_async(data):
    return await model(data)
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `TELEFUSER_PROFILE_DEBUG` | 启用所有调试 profiling 上下文（Layer 1+） | "false" |
| `TELEFUSER_PIPELINE_NAME` | Pipeline 名称用于输出目录 | "default" |
| `TELEFUSER_PROFILER_OUTPUT_DIR` | 覆盖输出目录（默认：work_dirs/profiler_output/{name}/{date}） | None |
| `ENABLE_PROFILER_NAMES` | 逗号分隔的 stage 名称用于 torch.profiler（已废弃，使用 harness） | "" |

**默认输出目录：**

```
work_dirs/profiler_output/{TELEFUSER_PIPELINE_NAME}/{YYYYMMDD_HHMM}/
```

**快速参考：**

```bash
# Layer 1：Stage 时序 + I/O signature 捕获
export TELEFUSER_PROFILE_DEBUG=true
export TELEFUSER_PIPELINE_NAME="wan21_t2v"  # 可选，用于组织输出
python examples/wan_video/wan21_1_3b_text_to_video_h100.py
# 输出：work_dirs/profiler_output/wan21_t2v/20260402/timing.json

# Layer 2：独立 stage 分析（推荐）
# 使用 StageBenchHarness 以编程方式配合已加载的 stage
```

### 程序化控制 Profiler

```python
from telefuser.utils.profiler import (
    enable_profiler_for_names,
    set_profiler_output_dir,
    set_pipeline_name,
    get_profiler_output_dir,
)

# 编程方式设置 pipeline 名称
set_pipeline_name("wan21_t2v")

# 覆盖输出目录
set_profiler_output_dir("/path/to/traces")

# 获取当前输出目录
output_dir = get_profiler_output_dir()
```

## ProfilingContext 与 ProfilingContext4Debug

### ProfilingContext

始终激活的性能分析上下文：

```python
from telefuser.utils.profiler import ProfilingContext

@ProfilingContext("operation_name")
def process():
    # 总是记录执行时间和峰值内存
    pass
```

### ProfilingContext4Debug

根据 `TELEFUSER_PROFILE_DEBUG` 条件激活：

```python
from telefuser.utils.profiler import ProfilingContext4Debug

@ProfilingContext4Debug("debug_operation")
def process():
    # 仅当 TELEFUSER_PROFILE_DEBUG=true 时进行性能分析
    # 否则无任何开销
    pass
```

**推荐在 Stage 中使用：**

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.utils.profiler import ProfilingContext4Debug

class MyStage(BaseStage):
    @with_model_offload(["model"])
    @ProfilingContext4Debug("my_stage_process")
    @torch.inference_mode()
    def process(self, input_data):
        # 仅在调试模式下进行性能分析
        return self.model(input_data)
```

## 输出结果

### 控制台日志

使用 `ProfilingContext` 时，会记录以下信息：

```
[Profile] my_operation cost 0.123456 seconds
Rank 0 - Function 'my_operation' Peak Memory: 4.50 GB
```

当 Layer 1 分析启用时：

```
[Profiler] Timing report saved to: work_dirs/profiler_output/wan21_t2v/20260402/timing.json
[Profiler] I/O signature saved to: work_dirs/profiler_output/wan21_t2v/20260402/timing_io_signature.json
```

当 Layer 2 harness 分析运行时：

```
[Harness] Setup complete for stage 'denoise'
[Harness] Running 1 warmup iteration(s)...
[Harness] Running 1 profile iteration(s)...
[Harness] Chrome trace saved to: work_dirs/.../denoise_trace.json.gz
[Harness] Kernel breakdown saved to: work_dirs/.../denoise_breakdown.json
[Harness] Test script saved to: work_dirs/.../profile_denoise.py
[Harness] Average iteration time: 150.00 ms
```

### Chrome Trace 文件

Chrome trace 文件可视化方法：

1. **Chrome 浏览器：** `chrome://tracing` → 加载 `.json.gz` 文件
2. **TensorBoard：** `tensorboard --logdir work_dirs/profiler_output`
3. **Perfetto：** https://ui.perfetto.dev/

## 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `name` | str | 必填 | Profiler 名称，用于标识 |
| `reset_peak_memory` | bool | True | 性能分析前重置峰值内存统计 |

```python
# 自定义内存追踪行为
with ProfilingContext("operation", reset_peak_memory=False):
    # 不重置峰值内存 - 捕获累积峰值
    pass
```

## 分布式支持

Profiler 自动处理分布式环境：

```python
# 在分布式环境中（如 2 个 GPU）
with ProfilingContext("distributed_op"):
    # Rank 0 日志: "Rank 0 - Function 'distributed_op' Peak Memory: 4.50 GB"
    # Rank 1 日志: "Rank 1 - Function 'distributed_op' Peak Memory: 4.50 GB"
    pass
```

## 硬件平台支持

Profiler 支持多种硬件平台：

| 平台 | Profiler Activity |
|------|-------------------|
| CUDA (NVIDIA) | `torch.profiler.ProfilerActivity.CUDA` |
| XPU (Intel) | `torch.profiler.ProfilerActivity.XPU` |
| NPU (华为) | `torch.profiler.ProfilerActivity.PrivateUse1` |
| CPU | `torch.profiler.ProfilerActivity.CPU` (始终启用) |

## Stage 中的集成使用

### 典型使用模式

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.utils.profiler import ProfilingContext4Debug
import torch

class VAEDecodeStage(BaseStage):
    def __init__(self, name, module_manager, model_runtime_config):
        super().__init__(name, model_runtime_config)
        self.vae = module_manager.fetch_module("vae")
        self.model_names = ["vae"]

    @with_model_offload(["vae"])
    @ProfilingContext4Debug("vae_decode")
    @torch.inference_mode()
    def process(self, latents):
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            return self.vae.decode(latents)
```

### 分析多个操作

```python
class TextEncodingStage(BaseStage):
    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    def encode_text(self, prompts):
        # 整体编码被分析
        with ProfilingContext4Debug("tokenization"):
            tokens = self.tokenizer(prompts)
        with ProfilingContext4Debug("embedding"):
            embeddings = self.text_encoder(tokens)
        return embeddings
```

## 最佳实践

### 1. 使用有意义的名称

```python
# 推荐 - 描述性强且唯一
@ProfilingContext4Debug("vae_decode_video")
def decode_video() -> None:
    ...

@ProfilingContext4Debug("dit_denoising_step_0")
def denoise_step() -> None:
    ...

# 避免 - 通用或重复
@ProfilingContext4Debug("process")
def process() -> None:
    ...

@ProfilingContext4Debug("model")
def model() -> None:
    ...
```

### 2. 在 Stage 中使用 ProfilingContext4Debug

```python
# 推荐 - 生产环境无开销
@ProfilingContext4Debug("stage_name")
def process(self, data):
    pass

# 避免在生产代码中使用 - 总是激活
@ProfilingContext("stage_name")
def process(self, data):
    pass
```

### 3. 与其他装饰器组合使用

装饰器顺序很重要 - profiler 应包裹实际计算：

```python
@with_model_offload(["model"])      # 外层: 处理模型加载
@ProfilingContext4Debug("process")  # 中层: 分析计算
@torch.inference_mode()             # 内层: 禁用梯度
def process(self, data):
    return self.model(data)
```

## 故障排除

### Trace 文件过大

使用独立 Stage Bench Harness 替代完整 pipeline 分析：

```python
# 替代完整 pipeline（产生 100MB+ traces）
# 使用 harness 进行单次迭代分析
from telefuser.utils.stage_bench_harness import StageBenchHarness

harness = StageBenchHarness.from_signature_file(
    signature_path="timing_io_signature.json",
    stage_name="denoise",
    stage_instance=pipeline.denoise_stage,
)
harness.setup()
harness.profile()  # 产生 <10MB trace
```

### GPU Activity 缺失

如果 GPU 活动未被记录：

1. 验证平台支持（CUDA、XPU、NPU）
2. 检查 CUDA 同步是否正常工作

```python
from telefuser.platforms import current_platform
print(current_platform.device_type)  # 应为 "cuda"、"xpu" 或 "npu"
```

### 内存统计不准确

确保性能分析前进行同步：

```python
# Profiler 自动同步，但自定义计时需手动同步
from telefuser.platforms import current_platform
current_platform.synchronize()
with ProfilingContext("operation"):
    pass
```

### CLI 无法获取 Stage 实例

CLI 模式无法在没有加载模型的情况下执行 stage 分析。使用编程方式配合已加载的 stage 实例：

```python
# 先加载 pipeline
from my_pipeline import get_pipeline
pipeline = get_pipeline()

# 然后使用 harness
harness = StageBenchHarness.from_signature_file(
    signature_path="timing_io_signature.json",
    stage_name="denoise",
    stage_instance=pipeline.denoise_stage,
)
```

## 相关文档

- [添加新 Stage](./adding_new_stage.md) - Stage 开发中的 profiler 集成
- [Metrics 指标系统](./metrics.md) - 生产环境监控和可观测性
- [Logging 日志系统](./logging.md) - 日志配置和使用
- [Configuration 配置](./configuration.md) - 运行时配置选项
