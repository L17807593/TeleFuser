# Metrics System

TeleFuser provides a comprehensive metrics collection system compatible with Prometheus, designed for production monitoring and observability.

## Features

- **Prometheus-compatible metrics** - Counter, Gauge, Histogram, Summary types
- **Service-level metrics** - Tasks, queue, GPU monitoring
- **Stage-level metrics** - Automatic tracking for pipeline stages
- **GPU metrics** - NVIDIA GPU monitoring via pynvml
- **Thread-safe registry** - Concurrent access support
- **Prometheus export** - Standard text exposition format
- **Configurable buckets** - Customizable histogram buckets

## Quick Start

### Basic Usage

```python
from telefuser.metrics import (
    get_metrics_registry,
    Counter,
    Gauge,
    Histogram,
)

# Get the global registry
registry = get_metrics_registry()

# Create a counter
requests_total = registry.counter(
    "requests_total",
    "Total number of requests",
)
requests_total.inc()

# Create a gauge
queue_size = registry.gauge(
    "queue_size",
    "Current queue size",
)
queue_size.set(10)

# Create a histogram
latency = registry.histogram(
    "request_latency_seconds",
    "Request latency in seconds",
)
latency.observe(0.5)

# Export in Prometheus format
print(registry.get_prometheus_format())
```

### Configuration

```python
from telefuser.metrics import MetricsConfig, create_metrics_config

# Use default configuration
config = MetricsConfig()

# Custom configuration
config = create_metrics_config(
    enabled=True,
    enable_stage_metrics=True,
    enable_gpu_metrics=True,
    gpu_metrics_interval=5.0,
    namespace="telefuser",
)
```

## Metric Types

### Counter

Monotonically increasing counter for cumulative values:

```python
from telefuser.metrics import get_metrics_registry

registry = get_metrics_registry()
counter = registry.counter("tasks_completed", "Tasks completed")

counter.inc()       # Increment by 1
counter.inc(5)      # Increment by 5
# counter.dec()     # Error! Counters cannot be decremented

print(counter.value)  # Current value
```

Use cases:
- Total requests
- Tasks completed
- Errors encountered
- Bytes processed

### Gauge

Point-in-time value that can go up or down:

```python
gauge = registry.gauge("memory_used_bytes", "Memory used in bytes")

gauge.set(1024)     # Set to specific value
gauge.inc()         # Increment by 1
gauge.dec()         # Decrement by 1
gauge.set_to_current_time()  # Set to Unix timestamp
```

Use cases:
- Current queue size
- Memory usage
- Active connections
- Temperature

### Histogram

Distribution of values with configurable buckets:

```python
# Default buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
histogram = registry.histogram(
    "request_duration_seconds",
    "Request duration in seconds",
)

# Custom buckets
histogram = registry.histogram(
    "file_size_bytes",
    "File size in bytes",
    buckets=[100, 1000, 10000, 100000, 1000000],
)

histogram.observe(0.5)
histogram.observe(1.2)
histogram.observe(0.05)

print(histogram.count)   # Total observations
print(histogram.sum)     # Sum of all values
print(histogram.average) # Average value
```

Use cases:
- Request latency
- Response sizes
- Processing times

### Summary

Summary metric with configurable quantiles:

```python
# Default quantiles: [0.5, 0.9, 0.95, 0.99]
summary = registry.summary(
    "response_time_seconds",
    "Response time in seconds",
)

# Custom quantiles
summary = registry.summary(
    "latency_seconds",
    "Latency in seconds",
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
)

summary.observe(0.1)
summary.observe(0.5)
summary.observe(1.0)
```

### Labels

Add dimensions to metrics:

```python
# Using dictionary
counter = registry.counter(
    "http_requests_total",
    "Total HTTP requests",
    labels={"method": "GET", "status": "200"},
)

# Using list of MetricLabel
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

## Stage Metrics

### Enable Stage Metrics

```python
from telefuser.metrics import enable_stage_metrics, disable_stage_metrics
from telefuser.core import BaseStage

class MyStage(BaseStage):
    def __init__(self, name, config):
        super().__init__(name, config)
        # Enable metrics for this stage
        enable_stage_metrics(self)

    def process(self, data):
        # Metrics are automatically tracked if using @with_metrics
        return result

# Disable metrics
disable_stage_metrics(my_stage)
```

### Using Decorators

```python
from telefuser.metrics import with_metrics

class MyStage(BaseStage):
    @with_metrics
    def process(self, data):
        # Execution time, success/failure automatically tracked
        return self._do_work(data)
```

### Stage Metric Context

Each stage gets the following metrics automatically:

| Metric | Type | Description |
|--------|------|-------------|
| `telefuser_stage_{name}_duration_seconds` | Histogram | Execution duration |
| `telefuser_stage_{name}_total` | Counter | Total executions |
| `telefuser_stage_{name}_errors_total` | Counter | Total errors |

```python
from telefuser.metrics import enable_stage_metrics

context = enable_stage_metrics(my_stage)

# Access individual metrics
context.duration.observe(0.5)
context.total.inc()
context.errors.inc()

# Record a complete execution
context.record_execution(duration=0.5, success=True)
```

## Service Metrics

### Overview

Service-level metrics for monitoring TeleFuser service:

```python
from telefuser.metrics import get_service_metrics, ServiceMetrics

# Get global service metrics
service_metrics = get_service_metrics()

# Record task events
service_metrics.record_task_created()
service_metrics.record_task_completed(duration=1.5)
service_metrics.record_task_failed()
service_metrics.record_task_cancelled()

# Update queue metrics
service_metrics.update_queue_metrics(
    size=10,
    pending=5,
    processing=2,
)

# Collect GPU metrics
service_metrics.collect_gpu_metrics()

# Get Prometheus format
output = service_metrics.get_prometheus_format()
```

### Task Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `tasks_created_total` | Counter | Total tasks created |
| `tasks_completed_total` | Counter | Tasks completed successfully |
| `tasks_failed_total` | Counter | Tasks that failed |
| `tasks_cancelled_total` | Counter | Tasks cancelled |
| `task_duration_seconds` | Histogram | Task execution duration |
| `task_queue_wait_seconds` | Histogram | Queue wait time |

### Queue Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `queue_size` | Gauge | Total queue size |
| `queue_pending` | Gauge | Pending tasks |
| `queue_processing` | Gauge | Processing tasks |

### GPU Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gpu_{id}_memory_used_bytes` | Gauge | GPU memory used |
| `gpu_{id}_memory_total_bytes` | Gauge | GPU total memory |
| `gpu_{id}_utilization_ratio` | Gauge | GPU utilization (0-1) |
| `gpu_{id}_temperature_celsius` | Gauge | GPU temperature |

### Periodic Collection

```python
import asyncio
from telefuser.metrics import get_service_metrics

async def main():
    service_metrics = get_service_metrics()

    # Start periodic GPU metrics collection
    asyncio.create_task(service_metrics.start_periodic_collection())

    # Your application logic here
    await run_application()
```

## Prometheus Integration

### Built-in Endpoints

`telefuser serve` exposes both formats without additional application code:

```bash
curl http://localhost:8000/v1/service/metrics
curl http://localhost:8000/v1/service/metrics/json
```

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'telefuser'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /v1/service/metrics
```

### Example Output

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

## Runtime Benchmark Facts

`telefuser.metrics.runtime` measures synchronized target-side phase duration and optional CUDA allocator peaks.
It returns raw facts; AIPerf owns warmup exclusion, aggregation, semantic mapping, artifacts, and visualization.
Client delivery metrics such as `stream_fps` must remain separate from target compute metrics such as
`chunk_compute_fps`.

```python
from telefuser.metrics import (
    finish_runtime_measurement,
    start_runtime_measurement,
)

measurement = start_runtime_measurement(
    ["cuda:0"],
    capture_peak_memory=True,
)

# Run the measured target-side phase.
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

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | True | Enable metrics collection |
| `enable_stage_metrics` | bool | True | Enable stage-level metrics |
| `enable_gpu_metrics` | bool | True | Enable GPU metrics |
| `enable_http_metrics` | bool | True | Enable HTTP request metrics |
| `enable_queue_metrics` | bool | True | Enable queue metrics |
| `gpu_metrics_interval` | float | 5.0 | GPU collection interval (seconds) |
| `histogram_buckets` | list[float] | [0.005, 0.01, ...] | Default histogram buckets |
| `metrics_path` | str | "/metrics" | HTTP endpoint path |
| `namespace` | str | "telefuser" | Metric name prefix |
| `gpu_platform` | str | "auto" | GPU platform (nvidia/amd/auto) |

`MetricsConfig.metrics_path` defaults to `/metrics` for custom integrations. The built-in service routes are
`/v1/service/metrics` and `/v1/service/metrics/json`.
## Advanced Usage
### Custom Collectors

Add custom metrics from external sources:

```python
from telefuser.metrics import get_metrics_registry

def custom_collector():
    """Collect custom metrics."""
    return [
        "# HELP custom_metric My custom metric",
        "# TYPE custom_metric gauge",
        f"custom_metric {get_custom_value()}",
    ]

registry = get_metrics_registry()
registry.add_custom_collector(custom_collector)
```

### Multiple Registries

```python
from telefuser.metrics import MetricRegistry

# Create separate registries for different purposes
app_registry = MetricRegistry(namespace="app")
system_registry = MetricRegistry(namespace="system")

# Each registry is independent
app_registry.counter("requests", "App requests")
system_registry.gauge("cpu_usage", "CPU usage")
```

### Reset Metrics

```python
# Reset specific metric
counter.reset()

# Reset all metrics in registry
registry.reset_all()

# Clear all metrics
registry.clear()
```

## Best Practices

### 1. Naming Conventions

Follow Prometheus naming conventions:

```python
# Good
registry.counter("http_requests_total", "...")
registry.histogram("request_duration_seconds", "...")
registry.gauge("memory_used_bytes", "...")

# Avoid
registry.counter("httpRequests", "...")  # Use snake_case
registry.gauge("memory", "...")  # Include unit
```

### 2. Use Appropriate Types

- **Counter**: Cumulative values (requests, errors, bytes)
- **Gauge**: Current state (queue size, memory, temperature)
- **Histogram**: Distributions (latency, size)
- **Summary**: Quantiles (p50, p95, p99)

### 3. Meaningful Labels

```python
# Good - dimensions for aggregation
registry.counter(
    "http_requests_total",
    "Total HTTP requests",
    labels={"method": "GET", "endpoint": "/api/users"},
)

# Avoid - high cardinality
registry.counter(
    "http_requests_total",
    "...",
    labels={"user_id": "12345"},  # Too many unique values!
)
```

### 4. Histogram Buckets

Choose buckets appropriate for your data:

```python
# For latency (seconds)
buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

# For file sizes (bytes)
buckets=[100, 1000, 10000, 100000, 1000000, 10000000]
```