Installation
============

``tf-kernel`` is an independently versioned CUDA extension stored in the
TeleFuser monorepo. No prebuilt package is currently published. Installation
requires a local Makefile build and remains independent of the TeleFuser
installation.

Requirements
------------

- Python 3.10 or newer
- PyTorch 2.11.0
- An NVIDIA GPU in the SM80, SM90, or SM100 target families
- CUDA Toolkit 12.8 or newer and CMake 3.26 or newer for source builds

A PyTorch local version such as ``2.11.0+cu128`` satisfies the package's
``torch==2.11.0`` requirement. FP4 operations require SM100 or newer.

Source Build and Installation
-----------------------------

Clone the TeleFuser monorepo and build tf-kernel independently through its
Makefile. This does not install or depend on the TeleFuser Python package:

.. code-block:: bash

   git clone https://github.com/Tele-AI/TeleFuser.git
   cd TeleFuser/tf-kernel
   make build-auto PYTHON=/path/to/venv/bin/python

Local source compilation is supported only through the Makefile. Direct
``pip install ./tf-kernel`` and ``pip install -e ./tf-kernel`` commands fail
during CMake configuration. Make builds a correctly tagged wheel and installs
that artifact into the selected interpreter.

Local Make builds use the native ``linux_*`` platform tag. A
``manylinux_2_28`` tag is reserved for the container build, which checks the
wheel's ELF symbol versions against that policy before retagging it.

Build an Architecture-Specific Wheel
------------------------------------

Enter the kernel project and select an architecture. Every target writes a
wheel to ``dist/`` and installs it into the interpreter selected by
``PYTHON``.

.. code-block:: bash

   cd TeleFuser/tf-kernel
   make build-auto PYTHON=/path/to/venv/bin/python
   make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere and Ada
   make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper
   make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell

``make build`` compiles all supported targets. A resource-bounded H100 build
can be run with:

.. code-block:: bash

   PATH=/usr/local/cuda-12.8/bin:$PATH \
   CUDA_HOME=/usr/local/cuda-12.8 \
   make build-sm90 \
     PYTHON=/path/to/venv/bin/python \
     MAX_JOBS=2 \
     CMAKE_BUILD_PARALLEL_LEVEL=2 \
     TF_KERNEL_COMPILE_THREADS=1

The first source build needs network access for pinned CUTLASS, FlashInfer,
and other CMake dependencies.

``MAX_JOBS`` controls concurrent build jobs, while
``TF_KERNEL_COMPILE_THREADS`` controls the internal NVCC threads used by each
job. On a host with enough CPU and memory, a faster build can be requested with:

.. code-block:: bash

   make build-auto MAX_JOBS=16 TF_KERNEL_COMPILE_THREADS=4

Increasing their product also increases peak CPU and memory pressure.

Verify Installation
-------------------

Run the smoke test with the interpreter that will load the package:

.. code-block:: bash

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
   output = tf_kernel.rmsnorm(x, weight)
   assert output.shape == x.shape and torch.isfinite(output).all()
   print("RMSNorm smoke test: OK")
   PY

Run ``python -m pip check`` afterward to find dependency conflicts. An H100
wheel should load its common extension from an ``sm90`` package directory.
Import also checks the build-time PyTorch public version, PyTorch CUDA version,
C++11 ABI, and target GPU family against the runtime. A process exposing GPUs
from different architecture families is rejected explicitly.

Troubleshooting
---------------

Wrong Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``python -m pip`` consistently and pass the intended interpreter as the
Make ``PYTHON`` variable. Check it with ``python -m pip show tf-kernel`` and
``python -c "import sys; print(sys.executable)"``.

CUDA Toolkit Not Found
^^^^^^^^^^^^^^^^^^^^^^

Check ``nvcc --version`` and set ``CUDA_HOME`` to the CUDA 12.8+ toolkit. Put
``$CUDA_HOME/bin`` before older toolkits in ``PATH``.

Wrong GPU Architecture
^^^^^^^^^^^^^^^^^^^^^^

Rebuild with ``make build-auto`` on the target machine or explicitly select
``build-sm80``, ``build-sm90``, or ``build-sm100``.

High Build Resource Use
^^^^^^^^^^^^^^^^^^^^^^^

Reduce ``MAX_JOBS``, ``CMAKE_BUILD_PARALLEL_LEVEL``, and
``TF_KERNEL_COMPILE_THREADS``. Building one SM target also reduces compilation
time and wheel size.

FP4 Availability on Ampere or Hopper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FP4 kernels are built only for SM100 and newer GPUs. On SM80 and SM90,
``tf_kernel.FP4_AVAILABLE`` is false; import does not print a warning.

Unvalidated SageAttention Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SageAttention v2 auto-dispatch is enabled for SM80, SM86, SM89, SM90, SM120,
and SM121. Other compute capabilities raise an explicit unsupported-backend
error instead of selecting an implementation that has not been validated.

Validated SM90 SageAttention Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The automatic ``sageattn`` entry point selects the validated SM90 FP8
implementation on H100. TeleFuser selects the same implementation for
``SAGE_ATTN_2_8_8_SM90`` when ``tf-kernel`` is available. Run the synchronized
smoke and public-ops GPU integration tests on each newly built wheel.
