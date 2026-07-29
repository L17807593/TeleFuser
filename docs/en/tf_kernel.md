# tf-kernel installation and usage

`tf-kernel` is TeleFuser's optional CUDA extension package. It provides fused elementwise operations, quantized
GEMM, SageAttention, and block-sparse attention kernels. The package lives in the `tf-kernel/` directory of this
repository and has its own package metadata and version. No prebuilt package is currently published; local
installation requires a source build on a provisioned CUDA/NVCC host.

TeleFuser can run without `tf-kernel`: the `telefuser.ops` layer keeps native PyTorch or Triton fallbacks where they
are implemented. Install `tf-kernel` when a pipeline uses one of its optimized CUDA paths.

!!! important

    TeleFuser model code must continue importing operations from `telefuser.ops`, not directly from `tf_kernel`.
    Direct imports below are intended for standalone kernel use, diagnostics, and kernel development.

## Compatibility

| Component | Requirement or target |
|-----------|-----------------------|
| Python | 3.10 or newer |
| PyTorch | `2.11.0` (a CUDA local version such as `2.11.0+cu128` satisfies this requirement) |
| CUDA Toolkit | 12.8 or newer for source builds |
| CMake | 3.26 or newer for source builds |
| GPU targets | SM80, SM90, and SM100 |

Kernel availability depends on the selected build target. FP4 kernels require Blackwell (SM100 or newer), and
`tf_kernel.FP4_AVAILABLE` is false without an import-time warning on Ampere and Hopper. Core operations are currently
validated with Python 3.11, PyTorch 2.11.0+cu128, CUDA 12.8, and H100 (SM90a). Other targets and operation families
should be validated on their target GPU before production use.

The wheel records and verifies its PyTorch public version, PyTorch CUDA version, C++11 ABI, and target GPU family at
import. A process exposing GPUs from different architecture families is rejected. SageAttention v2 auto-dispatch is
enabled only for SM80, SM86, SM89, SM90, SM120, and SM121.

!!! note "H100 SageAttention dispatch"

    On H100, `tf_kernel.sageattn()` selects the validated SM90 FP8 implementation. TeleFuser uses the same kernel
    when `SAGE_ATTN_2_8_8_SM90` is configured and `tf-kernel` is available.

## Build and install from source

Clone the TeleFuser monorepo, select the interpreter that already contains PyTorch 2.11.0, and enter the kernel
project:

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser/tf-kernel
```

For a local workstation, auto-detect the installed GPU:

```bash
make build-auto PYTHON=/path/to/venv/bin/python
```

The local build is independent of the TeleFuser installation. Make builds a correctly tagged wheel and installs it
into `PYTHON`. Direct `pip install .` and `pip install -e .` source builds fail with instructions to use Make; pip
package-index installation is not available.

Local builds use a `linux_*` platform tag. The container build may use `manylinux_2_28` only after checking every
shared object's GLIBC symbol versions against that policy.

For a reproducible target-specific build:

```bash
make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere and Ada
make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper, including H100
make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell
```

An H100 build with bounded host resource use can be run as follows:

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
make build-sm90 \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  TF_KERNEL_COMPILE_THREADS=1
```

`make build` builds all supported targets. Every build target writes a wheel to `dist/`, adds a legal wheel build
tag containing the Torch/CUDA ABI, and installs that wheel into the interpreter selected by `PYTHON`. The initial
build needs network access to obtain pinned CUTLASS, FlashInfer, and other CMake dependencies.

`MAX_JOBS` controls the number of concurrent build jobs. `TF_KERNEL_COMPILE_THREADS` controls the internal NVCC
threads used by each job. Increasing them can reduce build time on a sufficiently provisioned host; their product
also increases CPU and memory pressure.

## Verify the installation

Run the check with the same interpreter that will start TeleFuser:

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

An H100-specific wheel should load its common extension from an `sm90` package directory. Also run
`python -m pip check` to expose dependency conflicts in the environment.

For development validation, run `make test-cpu`, `make test-smoke`, and `make test-wheel`. The smoke and GPU targets
install the wheel into an isolated temporary directory before collecting tests. `make test` is the bounded GPU suite;
reserve the 6,000+ case `make test-full` matrix for a dedicated validation host.

## Usage

TeleFuser users should call the public ops layer; it selects `tf-kernel` for supported eager CUDA paths and keeps the
framework fallback behavior:

```python
import torch

from telefuser.ops.activations import silu_and_mul

x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
y = silu_and_mul(x)  # The last dimension is split into two 1024-wide tensors.
assert y.shape == (4, 1024)
```

Standalone users can call the kernel package directly:

```python
import torch
import tf_kernel

# RMSNorm
x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
y = tf_kernel.rmsnorm(x, weight, eps=1e-6)

# H100-tested SM90 FP8 SageAttention path.
# HND layout: [batch, heads, sequence, head_dim]
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

Per-token FP8 quantization writes into caller-provided output tensors:

```python
x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
x_scale = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
tf_kernel.tf_per_token_quant_fp8(x, x_q, x_scale)
```

See the API reference under `tf-kernel/docs/` and the tests in `tf-kernel/tests/` for lower-level contracts.

## Troubleshooting

### The wrong Python environment is used

Always use `python -m pip` and pass `PYTHON=/path/to/venv/bin/python` to Make. Confirm both paths with
`python -m pip show tf-kernel` and `python -c "import sys; print(sys.executable)"`.

### PyTorch is replaced or dependency resolution fails

`tf-kernel` requires PyTorch 2.11.0 because compiled extensions are tied to the PyTorch/CUDA ABI. Install TeleFuser
and `tf-kernel` into a clean environment if another package pins an incompatible PyTorch version. Do not bypass the
constraint unless you rebuild and validate the extension against the replacement version.

### CMake cannot find CUDA or uses the wrong toolkit

Check `nvcc --version`, set `CUDA_HOME` to the CUDA 12.8+ toolkit, and put `$CUDA_HOME/bin` before older toolkits in
`PATH`. The PyTorch CUDA runtime and the selected toolkit should be ABI-compatible.

### The extension was built for the wrong GPU

Rebuild with `make build-auto` on the target machine or use the explicit `build-sm80`, `build-sm90`, or
`build-sm100` target. Architecture-specific wheels cannot provide kernels that were omitted at build time.

### Validate an SM90 SageAttention deployment

The SM90-specific kernel is enabled on H100. After building the wheel, run the synchronized tf-kernel smoke test and
the TeleFuser public-ops GPU integration test before deploying that artifact on a new host.


### The build exhausts CPU or memory

Lower `MAX_JOBS`, `CMAKE_BUILD_PARALLEL_LEVEL`, and `TF_KERNEL_COMPILE_THREADS`. Targeting one SM architecture also
substantially reduces build time and artifact size.
