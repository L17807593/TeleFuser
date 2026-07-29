# 指标系统

TeleFuser 提供了一个与 Prometheus 兼容的综合指标收集系统，专为生产监控和可观测性设计。

## 特性

- **Prometheus 兼容指标** - Counter、Gauge、Histogram、Summary 类型
- **服务级指标** - 任务、队列、GPU 监控
- **阶段级指标** - 流水线阶段自动追踪
- **GPU 指标** - 通过 pynvml 进行 NVIDIA GPU 监控
- **线程安全注册表** - 支持并发访问
- **Prometheus 导出** - 标准文本展示格式
- **可配置桶** - 可自定义直方图桶

## 快速开始

### 基础用法

```python
from telefuser.metrics import (
    get_metrics_registry,
    Counter,
    Gauge,
    Histogram,
)

# 获取全局注册表
registry = get_metrics_registry()

# 创建计数器
requests_total = registry.counter(
    "requests_total",
    "Total number of requests",
)
requests_total.inc()

# 创建仪表
queue_size = registry.gauge(
    "queue_size",
    "Current queue size",
)
queue_size.set(10)

# 创建直方图
latency = registry.histogram(
    "request_latency_seconds",
    "Request latency in seconds",
)
latency.observe(0.5)

# 导出 Prometheus 格式
print(registry.get_prometheus_format())
```

### 配置

```python
from telefuser.metrics import MetricsConfig, create_metrics_config

# 使用默认配置
config = MetricsConfig()

# 自定义配置
config = create_metrics_config(
    enabled=True,
    enable_stage_metrics=True,
    enable_gpu_metrics=True,
    gpu_metrics_interval=5.0,
    namespace="telefuser",
)
```

## 指标类型

### Counter（计数器）

单调递增的计数器，用于累计值：

```python
from telefuser.metrics import get_metrics_registry

registry = get_metrics_registry()
counter = registry.counter("tasks_completed", "Tasks completed")

counter.inc()       # 增加 1
counter.inc(5)      # 增加 5
# counter.dec()     # 错误！计数器不能减少

print(counter.value)  # 当前值
```

适用场景：
- 总请求数
- 完成的任务数
- 遇到的错误数
- 处理的字节数

### Gauge（仪表）

可增可减的即时值：

```python
gauge = registry.gauge("memory_used_bytes", "Memory used in bytes")

gauge.set(1024)     # 设置为特定值
gauge.inc()         # 增加 1
gauge.dec()         # 减少 1
gauge.set_to_current_time()  # 设置为 Unix 时间戳
```

适用场景：
- 当前队列大小
- 内存使用量
- 活跃连接数
- 温度

### Histogram（直方图）

具有可配置桶的值分布：

```python
# 默认桶: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
histogram = registry.histogram(
    "request_duration_seconds",
    "Request duration in seconds",
)

# 自定义桶
histogram = registry.histogram(
    "file_size_bytes",
    "File size in bytes",
    buckets=[100, 1000, 10000, 100000, 1000000],
)

histogram.observe(0.5)
histogram.observe(1.2)
histogram.observe(0.05)

print(histogram.count)   # 总观察次数
print(histogram.sum)     # 所有值的总和
print(histogram.average) # 平均值
```

适用场景：
- 请求延迟
- 响应大小
- 处理时间

### Summary（摘要）

具有可配置分位数的摘要指标：

```python
# 默认分位数: [0.5, 0.9, 0.95, 0.99]
summary = registry.summary(
    "response_time_seconds",
    "Response time in seconds",
)

# 自定义分位数
summary = registry.summary(
    "latency_seconds",
    "Latency in seconds",
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
)

summary.observe(0.1)
summary.observe(0.5)
summary.observe(1.0)
```

### Labels（标签）

为指标添加维度：

```python
# 使用字典
counter = registry.counter(
    "http_requests_total",
    "Total HTTP requests",
    labels={"method": "GET", "status": "200"},
)

# 使用 MetricLabel 列表
from telefuser.metrics import MetricLabel

counter = registry.counter(
    "http_requests_total",
    "Total HTTP requests",
    labels=[
        MetricLabel("method", "GET"),
        MetricLabel("status", "200"),
    ],
)
```

## 阶段指标

### 启用阶段指标

```python
from telefuser.metrics import enable_stage_metrics, disable_stage_metrics
from telefuser.core import BaseStage

class MyStage(BaseStage):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 为此阶段启用指标
        enable_stage_metrics(self)

    def process(self, data):
        # 如果使用 @with_metrics，指标会自动追踪
        return result

# 禁用指标
disable_stage_metrics(my_stage)
```

### 使用装饰器

```python
from telefuser.metrics import with_metrics

class MyStage(BaseStage):
    @with_metrics
    def process(self, data):
        # 执行时间、成功/失败自动追踪
        return self._do_work(data)
```

### 阶段指标上下文

每个阶段自动获得以下指标：

| 指标 | 类型 | 描述 |
|------|------|------|
| `telefuser_stage_{name}_duration_seconds` | Histogram | 执行时长 |
| `telefuser_stage_{name}_total` | Counter | 总执行次数 |
| `telefuser_stage_{name}_errors_total` | Counter | 总错误数 |

```python
from telefuser.metrics import enable_stage_metrics

context = enable_stage_metrics(my_stage)

# 访问单个指标
context.duration.observe(0.5)
context.total.inc()
context.errors.inc()

# 记录完整执行
context.record_execution(duration=0.5, success=True)
```

## 服务指标

### 概述

用于监控 TeleFuser 服务的服务级指标：

```python
from telefuser.metrics import get_service_metrics, ServiceMetrics

# 获取全局服务指标
service_metrics = get_service_metrics()

# 记录任务事件
service_metrics.record_task_created()
service_metrics.record_task_completed(duration=1.5)
service_metrics.record_task_failed()
service_metrics.record_task_cancelled()

# 更新队列指标
service_metrics.update_queue_metrics(
    size=10,
    pending=5,
    processing=2,
)

# 收集 GPU 指标
service_metrics.collect_gpu_metrics()

# 获取 Prometheus 格式
output = service_metrics.get_prometheus_format()
```

### 任务指标

| 指标 | 类型 | 描述 |
|------|------|------|
| `tasks_created_total` | Counter | 创建的任务总数 |
| `tasks_completed_total` | Counter | 成功完成的任务 |
| `tasks_failed_total` | Counter | 失败的任务 |
| `tasks_cancelled_total` | Counter | 取消的任务 |
| `task_duration_seconds` | Histogram | 任务执行时长 |
| `task_queue_wait_seconds` | Histogram | 队列等待时间 |

### 队列指标

| 指标 | 类型 | 描述 |
|------|------|------|
| `queue_size` | Gauge | 队列总大小 |
| `queue_pending` | Gauge | 待处理任务数 |
| `queue_processing` | Gauge | 处理中任务数 |

### GPU 指标

| 指标 | 类型 | 描述 |
|------|------|------|
| `gpu_{id}_memory_used_bytes` | Gauge | GPU 已用显存 |
| `gpu_{id}_memory_total_bytes` | Gauge | GPU 总显存 |
| `gpu_{id}_utilization_ratio` | Gauge | GPU 利用率 (0-1) |
| `gpu_{id}_temperature_celsius` | Gauge | GPU 温度 |

### 周期性收集

```python
import asyncio
from telefuser.metrics import get_service_metrics

async def main():
    service_metrics = get_service_metrics()

    # 启动周期性 GPU 指标收集
    asyncio.create_task(service_metrics.start_periodic_collection())

    # 你的应用逻辑
    await run_application()
```

## Prometheus 集成

### 内置端点

`telefuser serve` 无需额外应用代码即可暴露两种格式：

```bash
curl http://localhost:8000/v1/service/metrics
curl http://localhost:8000/v1/service/metrics/json
```

### Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'telefuser'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /v1/service/metrics
```

### 示例输出

```
# HELP telefuser_tasks_created_total Total number of tasks created
# TYPE telefuser_tasks_created_total counter
telefuser_tasks_created_total 100

# HELP telefuser_task_duration_seconds Duration of task execution in seconds
# TYPE telefuser_task_duration_seconds histogram
telefuser_task_duration_seconds_bucket{le="0.005"} 10
telefuser_task_duration_seconds_bucket{le="0.01"} 25
telefuser_task_duration_seconds_bucket{le="0.025"} 45
telefuser_task_duration_seconds_bucket{le="0.05"} 60
telefuser_task_duration_seconds_bucket{le="0.1"} 75
telefuser_task_duration_seconds_bucket{le="0.25"} 85
telefuser_task_duration_seconds_bucket{le="0.5"} 92
telefuser_task_duration_seconds_bucket{le="1.0"} 97
telefuser_task_duration_seconds_bucket{le="2.5"} 99
telefuser_task_duration_seconds_bucket{le="5.0"} 100
telefuser_task_duration_seconds_bucket{le="10.0"} 100
telefuser_task_duration_seconds_bucket{le="+Inf"} 100
telefuser_task_duration_seconds_sum 45.5
telefuser_task_duration_seconds_count 100

# HELP telefuser_gpu_0_memory_used_bytes GPU 0 memory used in bytes
# TYPE telefuser_gpu_0_memory_used_bytes gauge
telefuser_gpu_0_memory_used_bytes 8589934592
```

## 运行时基准事实

`telefuser.metrics.runtime` 测量同步后的目标侧阶段耗时，以及可选的 CUDA allocator 峰值。它只返回原始事实；
AIPerf 负责排除 warmup、聚合、语义映射、产物和可视化。客户端交付指标（如 `stream_fps`）必须与目标计算指标
（如 `chunk_compute_fps`）分开。

```python
from telefuser.metrics import (
    finish_runtime_measurement,
    start_runtime_measurement,
)

measurement = start_runtime_measurement(
    ["cuda:0"],
    capture_peak_memory=True,
)

# 执行待测目标侧阶段。
result = run_phase()

facts = finish_runtime_measurement(measurement)
# {
#   "seconds": ...,
#   "memory": [{
#       "device": "cuda:0",
#       "peak_allocated_bytes": ...,
#       "peak_reserved_bytes": ...,
#   }],
# }
```

## 配置选项

| 选项 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `enabled` | bool | True | 启用指标收集 |
| `enable_stage_metrics` | bool | True | 启用阶段级指标 |
| `enable_gpu_metrics` | bool | True | 启用 GPU 指标 |
| `enable_http_metrics` | bool | True | 启用 HTTP 请求指标 |
| `enable_queue_metrics` | bool | True | 启用队列指标 |
| `gpu_metrics_interval` | float | 5.0 | GPU 收集间隔（秒） |
| `histogram_buckets` | list[float] | [0.005, 0.01, ...] | 默认直方图桶 |
| `metrics_path` | str | "/metrics" | HTTP 端点路径 |
| `namespace` | str | "telefuser" | 指标名前缀 |
| `gpu_platform` | str | "auto" | GPU 平台 (nvidia/amd/auto) |

`MetricsConfig.metrics_path` 对自定义集成默认使用 `/metrics`。内置服务路由是
`/v1/service/metrics` 和 `/v1/service/metrics/json`。
## 高级用法
### 自定义收集器

从外部源添加自定义指标：

```python
from telefuser.metrics import get_metrics_registry

def custom_collector():
    """收集自定义指标。"""
    return [
        "# HELP custom_metric My custom metric",
        "# TYPE custom_metric gauge",
        f"custom_metric {get_custom_value()}",
    ]

registry = get_metrics_registry()
registry.add_custom_collector(custom_collector)
```

### 多注册表

```python
from telefuser.metrics import MetricRegistry

# 为不同目的创建独立注册表
app_registry = MetricRegistry(namespace="app")
system_registry = MetricRegistry(namespace="system")

# 每个注册表独立
app_registry.counter("requests", "App requests")
system_registry.gauge("cpu_usage", "CPU usage")
```

### 重置指标

```python
# 重置特定指标
counter.reset()

# 重置注册表中所有指标
registry.reset_all()

# 清除所有指标
registry.clear()
```

## 最佳实践

### 1. 命名约定

遵循 Prometheus 命名约定：

```python
# 推荐
registry.counter("http_requests_total", "...")
registry.histogram("request_duration_seconds", "...")
registry.gauge("memory_used_bytes", "...")

# 避免
registry.counter("httpRequests", "...")  # 使用 snake_case
registry.gauge("memory", "...")  # 包含单位
```

### 2. 使用适当的类型

- **Counter**: 累计值（请求、错误、字节）
- **Gauge**: 当前状态（队列大小、内存、温度）
- **Histogram**: 分布（延迟、大小）
- **Summary**: 分位数（p50、p95、p99）

### 3. 有意义的标签

```python
# 推荐 - 用于聚合的维度
registry.counter(
    "http_requests_total",
    "Total HTTP requests",
    labels={"method": "GET", "endpoint": "/api/users"},
)

# 避免 - 高基数
registry.counter(
    "http_requests_total",
    "...",
    labels={"user_id": "12345"},  # 唯一值太多！
)
```

### 4. 直方图桶

选择适合数据的桶：

```python
# 延迟（秒）
buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

# 文件大小（字节）
buckets=[100, 1000, 10000, 100000, 1000000, 10000000]
```