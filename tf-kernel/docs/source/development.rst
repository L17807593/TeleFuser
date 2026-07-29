Development Guide
=================

This guide is for developers who want to contribute to tf-kernel.

Setting up Development Environment
----------------------------------

1. Clone the TeleFuser monorepo and enter the kernel project:

   .. code-block:: bash

      git clone https://github.com/Tele-AI/TeleFuser.git
      cd TeleFuser/tf-kernel

2. Install development tools without installing tf-kernel from source through pip:

   .. code-block:: bash

      python -m pip install pytest pytest-cov sphinx sphinx-rtd-theme \
        sphinx-autodoc-typehints pre-commit isort ruff clang-format

3. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

4. Build and install the project through its Makefile:

   .. code-block:: bash

      make build-auto PYTHON=/path/to/venv/bin/python

   On a host with sufficient CPU and memory, enable more compilation
   parallelism with:

   .. code-block:: bash

      make build-auto MAX_JOBS=16 TF_KERNEL_COMPILE_THREADS=4 \
        PYTHON=/path/to/venv/bin/python

Project Structure
-----------------

.. code-block:: text

   tf-kernel/
   ├── csrc/              # C++/CUDA source files
   │   ├── elementwise/   # Elementwise operations
   │   ├── gemm/          # GEMM operations
   │   ├── sageattn2/     # SageAttention v2
   │   ├── sageattn3/     # SageAttention v3
   │   └── block_sparse_attn/  # Block sparse attention
   ├── tf_kernel/         # Python package
   ├── include/           # C++ headers
   ├── tests/             # Test suite
   ├── benchmark/         # Benchmarks
   └── docs/              # Documentation

Adding a New Kernel
-------------------

1. **Implement the kernel** in ``csrc/<category>/your_kernel.cu``

2. **Declare the interface** in ``include/tf_kernel_ops.h``

3. **Register with PyTorch** in ``csrc/common_extension.cc``:

   .. code-block:: cpp

      m.def("your_kernel(Tensor input, Tensor! output) -> ()");
      m.impl("your_kernel", torch::kCUDA, &your_kernel);

4. **Update CMakeLists.txt**: Add source file to ``SOURCES``

5. **Create Python wrapper** in ``tf_kernel/<category>.py``

6. **Export in** ``tf_kernel/__init__.py``

7. **Add tests** in ``tests/test_your_kernel.py``

8. **Add benchmarks** in ``benchmark/`` (if applicable)

Coding Standards
----------------

C++/CUDA
^^^^^^^^

- Use clang-format with the provided ``.clang-format`` config
- 2-space indentation
- 120 column limit
- Left pointer alignment (``int* ptr`` not ``int *ptr``)

Format C++/CUDA files:

.. code-block:: bash

   make format

Python
^^^^^^

- **isort**: Import sorting
- ~~**black**: Code formatting~~ (Disabled)
- **ruff**: Linting

Format Python files:

.. code-block:: bash

   make format

Running Tests
-------------

Run pure-Python build and capability tests without importing GPU modules:

.. code-block:: bash

   make test-cpu

Run the synchronized wheel smoke suite, then the bounded GPU suite:

.. code-block:: bash

   make test-smoke
   make test

Run the exhaustive GPU matrix only on a dedicated validation host:

.. code-block:: bash

   make test-full

Run wheel architecture and symbol checks:

.. code-block:: bash

   make test-wheel

The GPU targets install the selected ``WHEEL`` into an isolated temporary
directory before test collection. This prevents the source checkout or a
different installed package from shadowing the artifact under test.

Building Documentation
----------------------

1. Install documentation dependencies:

   .. code-block:: bash

      pip install ".[docs]"

2. Build the documentation:

   .. code-block:: bash

      cd docs
      make html

3. View the documentation:

   .. code-block:: bash

      open build/html/index.html
