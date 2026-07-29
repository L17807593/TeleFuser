# Logging System

TeleFuser provides an industrial-grade logging system based on [loguru](https://github.com/Delgan/loguru), designed for production deployment scenarios.

## Features

- **Structured JSON logging** - Machine-parseable output for log aggregation systems
- **Log rotation** - Automatic rotation by size/time with compression
- **Multiple sinks** - Console, file, syslog, or both simultaneously
- **Environment-based configuration** - Configure via environment variables
- **Module-level filtering** - Different log levels for different modules
- **Async logging** - Non-blocking logging for high-performance scenarios
- **Context binding** - Attach request_id, trace_id, etc. to logs
- **Distributed tracing** - Built-in trace_id/span_id support for microservices
- **Multi-process support** - PID and Rank info with dynamic injection
- **Syslog support** - RFC 5424 compliant UDP logging to remote syslog servers

## Quick Start

### Basic Usage

Import the shared logger with:

```python
from telefuser.utils.logging import logger

logger.info("Application started")
logger.debug("Debug information: {}", some_value)
logger.error("An error occurred")
```

### Configuration via Environment Variables

```bash
# Set log level
export TELEFUSER_LOG_LEVEL=INFO

# Use JSON format for structured logging
export TELEFUSER_LOG_FORMAT=json

# Log to file with rotation
export TELEFUSER_LOG_FILE=/var/log/telefuser/app.log
export TELEFUSER_LOG_ROTATION="1 day"
export TELEFUSER_LOG_RETENTION="30 days"

# Use both console and file
export TELEFUSER_LOG_SINK=both

# Multi-process mode
export TELEFUSER_LOG_MULTIPROCESS=per_rank

# Syslog configuration
export TELEFUSER_LOG_SINK=syslog
export TELEFUSER_LOG_SYSLOG_HOST=syslog.example.com
export TELEFUSER_LOG_SYSLOG_PORT=514
```

### Configuration via Code

```python
from telefuser.utils.logging import configure_logging, LogConfig, LogFormat, LogSink

# Simple configuration
configure_logging(LogConfig(level="DEBUG"))

# Advanced configuration
configure_logging(LogConfig(
    level="INFO",
    format=LogFormat.JSON,
    sink=LogSink.BOTH,
    log_file="/var/log/telefuser/app.log",
    rotation="100 MB",
    retention="30 days",
    compression="gz",
))

# Syslog configuration
configure_logging(LogConfig(
    sink=LogSink.SYSLOG,
    syslog_host="syslog.example.com",
    syslog_port=514,
    syslog_facility="user",
))
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEFUSER_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `TELEFUSER_LOG_FORMAT` | Output format (simple, detailed, json, structured) | detailed |
| `TELEFUSER_LOG_SINK` | Output destination (console, file, syslog, both) | console |
| `TELEFUSER_LOG_FILE` | Path to log file | None |
| `TELEFUSER_LOG_ROTATION` | Rotation condition (e.g., "100 MB", "1 day") | 100 MB |
| `TELEFUSER_LOG_RETENTION` | Retention period (e.g., "30 days", "10 files") | 30 days |
| `TELEFUSER_LOG_COMPRESSION` | Compression format (gz, bz2, xz) | gz |
| `TELEFUSER_LOG_ENQUEUE` | Use async logging | true |
| `TELEFUSER_LOG_BACKTRACE` | Include backtrace in errors | true |
| `TELEFUSER_LOG_DIAGNOSE` | Include diagnostic info | true |
| `TELEFUSER_LOG_SERIALIZE` | Serialize as JSON | false |
| `TELEFUSER_LOG_MULTIPROCESS` | Multi-process mode (single, per_process, per_rank) | single |
| `TELEFUSER_LOG_SYSLOG_HOST` | Syslog server hostname | localhost |
| `TELEFUSER_LOG_SYSLOG_PORT` | Syslog server port | 514 |
| `TELEFUSER_LOG_SYSLOG_FACILITY` | Syslog facility (user, daemon, local0) | user |
| `TELEFUSER_LOG_EXTRA_FIELDS` | JSON string of extra fields to include | {} |
| `TELEFUSER_LOG_FILTER_MODULES` | JSON string of module filters | {} |

## Advanced Features

### Distributed Tracing

Use `TracingContext` to add trace_id and span_id for distributed tracing:

```python
from telefuser.utils.logging import logger, TracingContext, get_trace_id, set_trace_id

# Auto-generate trace_id and span_id
with TracingContext():
    logger.info("Processing request")  # Includes auto-generated trace_id
    trace_id = get_trace_id()  # Get current trace_id

# Use specific trace_id (e.g., from incoming request header)
with TracingContext(trace_id="abc123", span_id="def456"):
    logger.info("Processing with custom trace")  # Uses provided trace_id

# Manually set trace_id
set_trace_id("xyz789")
logger.info("Message with trace_id")
```

The trace_id and span_id are automatically included in:
- JSON output format
- Structured format
- All log records via `extra` fields

### Context Binding

Attach contextual information to all logs within a scope:

```python
from telefuser.utils.logging import logger, LoggingContext

with LoggingContext(request_id="req-123", user_id="user-456"):
    logger.info("Processing request")  # Logs include request_id and user_id
    process_data()
    logger.info("Request completed")   # Still includes context
```

### Performance Monitoring

Supports both synchronous and asynchronous functions:

```python
from telefuser.utils.logging import log_execution_time

@log_execution_time
def heavy_computation():
    # This will log execution time at DEBUG level
    pass

@log_execution_time
async def async_computation():
    # Also works with async functions
    pass
```

### Exception Logging

Supports both synchronous and asynchronous functions:

```python
from telefuser.utils.logging import log_exceptions

@log_exceptions
def risky_operation():
    # Exceptions are automatically logged with full context
    raise ValueError("Something went wrong")

@log_exceptions
async def async_risky_operation():
    # Also works with async functions
    raise ValueError("Async error")
```

### Module-Level Filtering

```python
from telefuser.utils.logging import configure_logging, LogConfig

configure_logging(LogConfig(
    level="INFO",
    filter_modules={
        "telefuser.models": "WARNING",  # Only WARNING and above for models
        "telefuser.distributed": "DEBUG",  # DEBUG and above for distributed
    }
))
```

## Output Formats

### Simple
```
2024-01-15 10:30:00 | INFO     | Application started
```

### Detailed (default)
```
2024-01-15 10:30:00.123 | INFO     | [pid:12345 rank:0] mymodule:myfunc:42 | Application started
```

The `pid` and `rank` are dynamically injected at log time, ensuring correct values even if the process environment changes.

### JSON
```json
{
  "text": "Application started\n",
  "record": {
    "time": {"repr": "2024-01-15 10:30:00.123456"},
    "level": {"name": "INFO"},
    "extra": {
      "pid": "12345",
      "rank": "0",
      "trace_id": "abc123def456",
      "span_id": "789ghi"
    }
  }
}
```

### Structured
```
time=2024-01-15 10:30:00.123 level=INFO pid=12345 rank=0 trace_id=abc123 span_id=789ghi module=mymodule func=myfunc line=42 msg=Application started
```

## Production Deployment

### Docker Example

```dockerfile
ENV TELEFUSER_LOG_LEVEL=INFO
ENV TELEFUSER_LOG_FORMAT=json
ENV TELEFUSER_LOG_FILE=/var/log/telefuser/app.log
ENV TELEFUSER_LOG_ROTATION="1 day"
ENV TELEFUSER_LOG_RETENTION="7 days"
```

### Kubernetes Example

```yaml
env:
  - name: TELEFUSER_LOG_LEVEL
    value: "INFO"
  - name: TELEFUSER_LOG_FORMAT
    value: "json"
  - name: TELEFUSER_LOG_FILE
    value: "/var/log/telefuser/app.log"
volumeMounts:
  - name: logs
    mountPath: /var/log/telefuser
```

### Syslog Integration

For centralized logging with syslog servers:

```python
from telefuser.utils.logging import configure_logging, LogConfig, LogSink

configure_logging(LogConfig(
    sink=LogSink.SYSLOG,
    syslog_host="logs.example.com",
    syslog_port=514,
    syslog_facility="user",
    level="INFO",
))
```

Messages are sent in RFC 5424 format via UDP.

## Multi-Process Logging

Three multi-process logging modes are supported:

### 1. single (default) - Unified Log File
All processes write to the same log file. Thread/process safety is ensured via `enqueue=True`.

```python
from telefuser.utils.logging import configure_logging, LogConfig

configure_logging(LogConfig(
    log_file="/var/log/telefuser/app.log",
    multiprocess="single",  # default
))
```

### 2. per_process - Separate File per Process
Each process writes to its own log file with process ID in the filename.

```python
configure_logging(LogConfig(
    log_file="/var/log/telefuser/app.log",
    multiprocess="per_process",
))
# Generates: app_12345.log, app_12346.log, etc.
```

### 3. per_rank - Separate File per Distributed Rank
For distributed training scenarios, logs are split by rank number.

```python
configure_logging(LogConfig(
    log_file="/var/log/telefuser/app.log",
    multiprocess="per_rank",
))
# Generates: app_rank0.log, app_rank1.log, etc.
```

### Manual Placeholders
You can also manually specify placeholders in the path:

```python
configure_logging(LogConfig(
    log_file="/var/log/telefuser/app_{pid}_{rank}.log",
))
# Generates: app_12345_0.log, etc.
```

Supported placeholders:
- `{pid}` - Process ID (dynamically resolved at log time)
- `{rank}` - Distributed training rank (read from RANK/LOCAL_RANK env vars)
- `{time}` - Timestamp

## Startup Information

When the logging system initializes, it automatically prints configuration information to stderr:

```
╔══════════════════════════════════════════════════════════════════╗
║              TeleFuser Logging System Initialized               ║
╠══════════════════════════════════════════════════════════════════╣
║  Log Level:      INFO                                             ║
║  Format:         DETAILED                                         ║
║  Sink:           BOTH                                             ║
║  Log File:       /var/log/telefuser/app.log                       ║
║  Rotation:       1 day                                            ║
║  Retention:      30 days                                          ║
║  Compression:    gz                                               ║
║  Async Mode:     True                                             ║
║  Backtrace:      True                                             ║
║  Colorize:       True                                             ║
║  Multiprocess:   per_rank                                         ║
║  PID:            12345                                            ║
║  Rank:           0                                                ║
╚══════════════════════════════════════════════════════════════════╝
```

This allows you to immediately verify the logging configuration and file location when the application starts.

## Migration from loguru

The logging system is a drop-in replacement for direct loguru usage:

```python
# Before
from telefuser.utils.logging import logger

# After
from telefuser.utils.logging import logger

# Everything else stays the same!
logger.info("This works exactly the same")
```

## Complete Example

Here's a complete runnable example demonstrating all logging features:

```python
from __future__ import annotations

import time

from telefuser.utils.logging import (
    LogConfig,
    LogFormat,
    LogSink,
    LoggingContext,
    configure_logging,
    log_exceptions,
    log_execution_time,
    logger,
)


def demo_basic_logging():
    """Demonstrate basic logging usage."""
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


def demo_context_binding():
    """Demonstrate context binding for request tracking."""
    # Simulate a request context
    with LoggingContext(request_id="req-12345", user_id="user-789"):
        logger.info("Processing request")
        logger.debug("Request details: method=POST, path=/api/v1/generate")

        # Nested context
        with LoggingContext(stage="inference"):
            logger.info("Running inference")
            time.sleep(0.1)
            logger.info("Inference completed")

        logger.info("Request completed")

    # Outside context - no request_id
    logger.info("This log has no request context")


def demo_performance_monitoring():
    """Demonstrate performance monitoring decorators."""
    @log_execution_time
    def heavy_computation():
        """Simulate heavy computation."""
        time.sleep(0.5)
        return sum(range(10000))

    # Configure DEBUG level to see timing logs
    configure_logging(LogConfig(level="DEBUG"))

    result = heavy_computation()
    logger.info(f"Computation result: {result}")


def demo_exception_logging():
    """Demonstrate exception logging."""
    @log_exceptions
    def risky_operation():
        """Function that may raise exceptions."""
        raise ValueError("Something went wrong!")

    try:
        risky_operation()
    except ValueError:
        logger.info("Exception was caught and logged")


def demo_json_format():
    """Demonstrate JSON structured logging."""
    configure_logging(
        LogConfig(
            level="INFO",
            format=LogFormat.JSON,
            serialize=True,
        )
    )

    logger.info(
        "Application event",
        extra={
            "event_type": "user_action",
            "action": "generate_image",
            "model": "wan21",
        },
    )


def demo_production_config():
    """Demonstrate production-ready configuration."""
    config = LogConfig(
        level="INFO",
        format=LogFormat.JSON,
        sink=LogSink.BOTH,
        log_file="/var/log/telefuser/app.log",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        enqueue=True,  # Async logging for performance
        backtrace=True,
        diagnose=True,
    )

    print("Production config:")
    print(f"  Level: {config.level}")
    print(f"  Format: {config.format.name}")
    print(f"  Sink: {config.sink.name}")
    print(f"  File: {config.log_file}")


if __name__ == "__main__":
    print("TeleFuser Logging System Demo")
    print("=" * 50)

    demo_basic_logging()
    demo_context_binding()
    demo_performance_monitoring()
    demo_exception_logging()
    demo_json_format()
    demo_production_config()

    print("\n" + "=" * 50)
    print("Demo completed!")
```

Run with different configurations:

```bash
TELEFUSER_LOG_LEVEL=DEBUG python your_script.py
TELEFUSER_LOG_FORMAT=json python your_script.py
```
