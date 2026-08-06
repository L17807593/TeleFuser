# PyTorch `torch.compile` Compatibility Guide for Inference

This guide covers best practices for writing PyTorch code that is highly compatible with `torch.compile` for inference optimization.

## Introduction

`torch.compile` is PyTorch 2.0's JIT compiler that captures the model's computation graph and performs kernel fusion, memory planning, and other optimizations to significantly improve execution speed. To fully leverage its performance benefits, the model's `forward` code must follow specific conventions.

**Core Goal: Write "pure PyTorch" style `forward` functions that eliminate all Python runtime interactions that cause graph breaks.**

## Core Principles: Avoiding Graph Breaks

When the compiler encounters Python dynamic features that cannot be statically analyzed, a **graph break** occurs—the computation graph is split, and the compiler falls back to slow Python interpreter mode.

Basic principles:
- **Tensor-first**: Use PyTorch tensor operations (`torch.where`, `torch.gather`) instead of Python control flow
- **Avoid external libraries**: Do not call `numpy`, `scipy`, or `pandas` in `forward`
- **Stable inputs**: Keep input tensor dtype, device, and shape relatively stable
- **Strict mode development**: Use `torch.compile(model, fullgraph=True)` during development to catch all graph breaks

## Writing `torch.compile` Compatible `forward` Functions

### Data Structure Handling: Lists and Dicts

Dynamic data structures are a common cause of graph breaks.

| Data Type | ❌ Not Recommended (May Cause Breaks/Recompilation) | ✅ Recommended |
|:----------|:---------------------------------------------------|:--------------|
| **List** | - Using `list.append()`, `list.pop()`, `list.sort()` inside `forward`<br>- Number of tensors in list changes dynamically | - Use as simple input/output container<br>- Use `torch.cat` instead of loop appending<br>- Use Tuple as return container (safer) |
| **Dict** | - Complex nested dict as `forward` input parameter<br>- Iterating dict key-value pairs for logic inside `forward` | - **Unpack before entering model**: Flatten dict to tensor list or named tuple in `DataLoader.collate_fn`<br>- Explicitly extract tensors at `forward` start: `x = input_dict['image']` |

### Control Flow Handling: Conditionals and Loops

Control flow compatibility depends on whether the condition depends on tensor **values**.

| Statement Type | ❌ Dynamic Dependency (Causes Graph Break) | ✅ Static Dependency (Compile-friendly) |
|:---------------|:-------------------------------------------|:---------------------------------------|
| **If Conditional** | `if x.sum() > 0:` <br> `if x.shape[0] > 10:` | `if self.training:` <br> `if self.config.use_bias:` |
| **For Loop** | `for i in range(x.shape[0]):` <br> (If shape changes each call, triggers recompilation) | `for i in range(10):` <br> (Iteration count is constant) |

**Alternatives**:
- For conditionals depending on tensor values, use **`torch.where(condition, a, b)`**
- For dynamic shape loops, consider enabling dynamic shape support: `torch.compile(model, dynamic=True)` (sacrifices some performance)

### Reducing Unnecessary Recompilation

Even without graph breaks, frequent **recompilation** negates speed gains. Each function call triggers recompilation if the compiler detects "graph structure changes".

**Main Causes and Solutions**:

1. **Changing Tensor Shapes**:
   - **Cause**: Input is `(1, 3, 224, 224)` this call, `(1, 3, 256, 256)` next call
   - **Solution**: Fix dimensions via padding, or use `torch.compile(dynamic=True)` for specific dimension changes

2. **Changing Non-Tensor Parameters**:
   - **Cause**: `forward(self, x, multiplier)` where `multiplier` is a `float` that frequently changes
   - **Solution**: Wrap scalar as tensor: `multiplier_tensor = torch.tensor(multiplier, device=x.device)`. Compiler tolerates tensor value changes better

3. **Changing Device or Data Type**:
   - **Cause**: Sometimes running on CPU, sometimes on CUDA
   - **Solution**: Ensure inputs are always on same device and dtype

## Integrating Custom Operators (CUDA / Triton Kernel)

When using hand-written CUDA or Triton kernels, register them as PyTorch custom operators so `torch.compile` recognizes them as "black-box" operators.

### Standard Integration Steps

Use `torch.library.custom_op` decorator for registration. **Key: provide `impl_abstract` function**.

```python
import torch
from torch.library import custom_op

# 1. Define kernel entry point
def my_triton_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ... Actual Triton kernel call code ...
    return output

# 2. Register as PyTorch custom operator
@custom_op("mylib::my_fast_op", mutates_args=())
def my_fast_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return my_triton_kernel(a, b)

# 3. Must implement abstract inference function (FakeTensor support)
@my_fast_op.impl_abstract("mylib::my_fast_op")
def my_fast_op_abstract(a, b):
    # Only need to return empty tensor describing output shape, dtype
    return torch.empty_like(a)
```

### Using in Models

```python
class MyModel(nn.Module):
    def forward(self, x):
        # Call through torch.ops namespace
        return torch.ops.mylib.my_fast_op(x, x)

model = MyModel()
compiled_model = torch.compile(model, fullgraph=True)
```

### Important Notes

- **`impl_abstract` is required**: Without it, `torch.compile` fails when tracing FakeTensor
- **Triton-specific API**: For pure Triton kernels, check experimental API `torch._library.triton.triton_op` which may simplify integration

## Performance Trade-offs: Triton Operator vs. Native PyTorch Compile

A common decision: write logic as Triton operator then register, or let `torch.compile` fuse native PyTorch APIs?

### Internal Optimization Capability Comparison

| Scenario | Triton Custom Operator | PyTorch Native + `compile` |
|:---------|:-----------------------|:---------------------------|
| **High Compute Density** (Compute-Bound)<br>(e.g., FlashAttention, complex activations) | ✅ **Significantly faster**. Manual SRAM and pipeline control, 1.5x-3x improvement | ⚠️ Limited by base operator library, cannot achieve extreme fusion |
| **Low Compute Density** (Overhead-Bound)<br>(e.g., `x+1`, `x*scale+bias` point ops) | ⚠️ Hand-written Triton tedious and error-prone, limited performance gain | ✅ **Excellent**. Inductor backend auto-performs vertical/horizontal fusion, eliminates Python overhead |

### Global Graph Optimization Capability Comparison

After registering custom operator, `torch.compile` treats it as opaque "black-box".

| Global Optimization Type | Triton Custom Operator | PyTorch Native Operator |
|:-------------------------|:-----------------------|:------------------------|
| **Cross-Operator Fusion** | ❌ **Blocked**. Cannot fuse with adjacent PyTorch operations | ✅ **Supported**. Can fuse adjacent ops into single CUDA kernel |
| **Memory Layout Propagation** | ⚠️ Must manually adapt `channels_last` etc. formats | ✅ **Auto-handled**. Auto-selects optimal memory stride |

### Decision Guide

```text
Is this logic a classic optimization pattern?
    │
    ├─ Yes (e.g., FlashAttention, RMSNorm, Fused MLP)
    │      └─> 【Hand-write Triton and register Custom Op】
    │
    └─ No
           │
           ├─ Logic includes complex Python control flow (inevitable graph break)?
           │      └─> 【Hand-write Triton】
           │
           └─ Logic is just basic operator arrangement?
                  └─> 【Native PyTorch + torch.compile】
                      (Zero dev cost, doesn't block global fusion)
```

## TeleFuser's Mixed Strategy (Practice Case)

TeleFuser implements a **mixed strategy** for torch.compile compatibility based on operator characteristics and execution flow:

### Strategy by Operator Type

| Operator Type | Strategy | Reason |
|:--------------|:---------|:-------|
| **Attention** (High compute density) | `@torch.compiler.disable` | FlashAttention/SageAttention outperform native PyTorch; fusion gains limited |
| **RoPE** (Medium compute density) | `@torch.compiler.disable` | Triton kernel outperforms native; subsequent Attention blocks fusion anyway |
| **RMSNorm/LayerNorm** (Low compute density) | Native in compile mode | Overhead-bound; Inductor can fuse with adjacent ops |
| **modulate** (Point operations) | Native in compile mode | Minimal compute; Inductor auto-fusion optimal |

### Execution Flow Analysis

```
Linear → RMSNorm(q_norm) → RoPE → Attention
                      ↑        ↑         ↑
               Native+Fuse  Triton    Triton (disabled)
```

Key insight: Since Attention uses `@torch.compiler.disable`, any fusion beyond RoPE is blocked. Therefore:
- RoPE should use Triton kernel (no fusion opportunity anyway)
- RMSNorm should use native (potential fusion with preceding Linear)

### Implementation Example

```python
# Attention - always use optimized kernel, disable compile
@torch.compiler.disable
def attention(q, k, v, **kwargs):
    return flash_attn2(q, k, v, **kwargs)

# RoPE - use Triton kernel, disable compile
@torch.compiler.disable
def apply_rotary_emb(x, cos, sin):
    return apply_rotary_embedding(x, cos, sin)  # Triton

# RMSNorm - compile-aware dispatch
class RMSNorm(CustomOp):
    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)  # Allow fusion
        return self.forward_cuda(x)  # Triton in eager
```

## Inference-Specific Optimizations

### Using `torch.inference_mode`

For inference, `torch.inference_mode` is faster than `no_grad`:

```python
# Recommended for inference
with torch.inference_mode():
    output = compiled_model(input)

# Or mark in model class
model.eval()
compiled_model = torch.compile(model)
```

### CUDA Graph for Fixed Shapes

For fixed-shape inference, enable CUDA Graph for extreme optimization:

```python
# Internally uses CUDA Graph for kernel launch overhead reduction
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### Compilation Modes

```python
# Different compilation modes and use cases
torch.compile(model)                        # Default: auto-select
torch.compile(model, mode="default")        # Balance compile time and performance
torch.compile(model, mode="reduce-overhead")  # Reduce Python overhead, for small batch inference
torch.compile(model, mode="max-autotune")   # Max optimization, long compile time, for fixed shapes
```

### Deployment Best Practices

**Warmup for Production**:
```python
# First inference has compile overhead
model = torch.compile(model)

# Warmup before production serving
with torch.inference_mode():
    _ = model(dummy_input)  # Trigger compilation

# Now subsequent calls are fast
output = model(real_input)
```

**Compilation Cache**:
```python
import torch._inductor.config as inductor_config

# Set cache directory for compiled artifacts
inductor_config.cache_dir = "/path/to/cache"

# Compiled artifacts persist across sessions
compiled_model = torch.compile(model)
```

## Debugging and Profiling Tools

When encountering performance bottlenecks or compile failures, these tools help identify issues:

| Tool / Environment Variable | Usage |
|:--------------------------- |:------|
| `TORCH_LOGS=recompiles` | Print each recompilation's **specific cause** in terminal (shape change, scalar value change). First choice for performance issues |
| `torch.compile(..., fullgraph=True)` | Force full graph compile. Errors on any Python graph break, for strict development self-check |
| `torch._dynamo.explain(model)(x)` | Print detailed graph break report, pointing to specific line causing break |
| `torch.profiler` | Combined with `torch.compile`, view fused kernel execution |

## Quick Reference Table

| Issue | Diagnosis / Solution |
|:------|:---------------------|
| Compiled model slower than uncompiled | Use `TORCH_LOGS=recompiles` check for frequent recompilation. Check if input shape or scalar params change |
| Error `Graph break in user code` | Used tensor-value-dependent `if` or `for` in `forward`. Use `torch.where` or fix shape |
| Custom CUDA kernel `FakeTensor` error | Missing `impl_abstract` function. Add `@op.impl_abstract` definition |
| List operation warnings | Avoid dynamic list length modification in `forward`. Move dynamic concat logic to tensor ops (`torch.cat`) |

## Summary

Writing highly `torch.compile` compatible code is essentially a mindset shift from **Python dynamic features** to **static computation graph description**.

- **Short-term gains**: Avoid `if` checking tensor values, fix input shapes, register custom operators
- **Long-term gains**: Model inference speed can improve 30%-200%

Following this guide's principles, you can build PyTorch models that retain Python development flexibility while enjoying compiler extreme performance optimization.

## Related Documentation

- [Ops Module Documentation](./ops.md) - Custom operator implementation in TeleFuser
- [Profiler Guide](./profiler.md) - Performance profiling tools
- [Attention Implementation](./attention.md) - Attention module optimizations
