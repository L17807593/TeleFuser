.PHONY: help check-deps install-deps validate-build-env tree ln submodule install build build-wheel install-wheel build-auto build-sm80 build-sm90 build-sm100 clean rebuild test test-smoke test-gpu test-full test-cov test-cpu format format-check docs docs-clean docs-serve update test-wheel

# ---------------------------
# Build resource controls
# ---------------------------
# By default, build uses at most eight jobs, but users can override:
#   make build-auto MAX_JOBS=16 TF_KERNEL_COMPILE_THREADS=4
NPROC ?= $(shell nproc 2>/dev/null || echo 1)
MAX_JOBS ?= $(shell jobs=$(NPROC); if [ "$$jobs" -gt 8 ]; then echo 8; else echo "$$jobs"; fi)
CMAKE_BUILD_PARALLEL_LEVEL ?= $(MAX_JOBS)
TF_KERNEL_COMPILE_THREADS ?= 2
CMAKE_POLICY_VERSION_MINIMUM ?= 3.5
PYTHON ?= python3
WHEEL ?= dist/*.whl

# Target SM architecture selection
# Options: AUTO (detect local GPU), ALL (all supported architectures), SM80, SM90, SM100
TF_KERNEL_TARGET_SM ?= ALL
UV_BUILD_DIR ?= build-$(TF_KERNEL_TARGET_SM)

# Show help for each target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Build parallelism:"
	@echo "  MAX_JOBS=<n>                    Parallel build jobs (default: min(CPU count, 8))"
	@echo "  TF_KERNEL_COMPILE_THREADS=<n>   NVCC threads per compilation job (default: 2)"
	@echo "  Example: make build-auto MAX_JOBS=16 TF_KERNEL_COMPILE_THREADS=4"

check-deps: ## Check and install required Python formatting dependencies
	@$(PYTHON) -c "import isort, ruff" >/dev/null 2>&1 || \
		(echo "Installing isort and ruff..." && $(PYTHON) -m pip install isort ruff)

install-deps: ## Install Python formatting and wheel build tools
	$(PYTHON) -m pip install scikit-build-core isort ninja ruff uv wheel

validate-build-env: ## Validate Python, PyTorch, CUDA toolkit, and target SM
	@$(PYTHON) scripts/validate_build_env.py --target-sm "$(TF_KERNEL_TARGET_SM)"

tree: ## Show project directory structure
	@tree --prune -I "__pycache__|*.egg-info|*.so|build|3rdparty|dist"

submodule: ## Initialize and update git submodules
	@git submodule update --init --recursive

ln: submodule ## Create compilation database
	@rm -rf build && mkdir build && cd build && cmake .. -DTF_KERNEL_BUILD_VIA_MAKE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DCMAKE_POLICY_VERSION_MINIMUM=3.5

install: build-auto ## Build and install a wheel for the auto-detected GPU architecture

build: install-deps submodule validate-build-env ## Build and install wheel package (default: ALL)
	@$(MAKE) build-wheel
	@$(MAKE) install-wheel

build-wheel: ## Build a wheel without installing it
	@mkdir -p dist
	@rm -f dist/*.whl
	@CMAKE_POLICY_VERSION_MINIMUM=$(CMAKE_POLICY_VERSION_MINIMUM) \
		MAX_JOBS=$(MAX_JOBS) \
		CMAKE_BUILD_PARALLEL_LEVEL=$(CMAKE_BUILD_PARALLEL_LEVEL) \
		CMAKE_ARGS="$(CMAKE_ARGS) -DTF_KERNEL_BUILD_VIA_MAKE=ON -DTF_KERNEL_TARGET_SM=$(TF_KERNEL_TARGET_SM) -DTF_KERNEL_COMPILE_THREADS=$(TF_KERNEL_COMPILE_THREADS)" \
		$(PYTHON) -m uv build --wheel -Cbuild-dir=$(UV_BUILD_DIR) . --verbose --color=always --no-build-isolation
	@$(MAKE) rename-wheels PYTHON="$(PYTHON)"

install-wheel: ## Install the wheel currently present under dist/
	@set -- dist/*.whl; \
		if [ ! -f "$$1" ]; then echo "No wheel found under dist/"; exit 1; fi; \
		$(PYTHON) -m pip install "$$1" --force-reinstall --no-deps

build-auto: ## Build for auto-detected GPU architecture
	@$(MAKE) build TF_KERNEL_TARGET_SM=AUTO

build-sm80: ## Build for SM80 architecture (Ampere, Ada Lovelace)
	@$(MAKE) build TF_KERNEL_TARGET_SM=SM80

build-sm90: ## Build for SM90 architecture (Hopper H100)
	@$(MAKE) build TF_KERNEL_TARGET_SM=SM90

build-sm100: ## Build for SM100 architecture (Blackwell)
	@$(MAKE) build TF_KERNEL_TARGET_SM=SM100

rename-wheels: ## Add Torch/CUDA build metadata and the local Linux platform tag
	@echo "Renaming wheel files..."
	@PYTHON="$(PYTHON)" bash rename_wheels.sh local

clean: ## Remove build artifacts
	@case "$(UV_BUILD_DIR)" in build|build-*) ;; *) echo "Unsafe UV_BUILD_DIR: $(UV_BUILD_DIR)"; exit 1;; esac
	@rm -rf -- "$(UV_BUILD_DIR)" dist *.egg-info

rebuild: clean submodule build ## Clean and rebuild the project
	@echo "Succeed to rebuild"

test: test-gpu ## Run the bounded GPU correctness suite against the built wheel

test-smoke: ## Run synchronized smoke tests against the built wheel
	@$(PYTHON) scripts/run_wheel_tests.py --wheel "$(WHEEL)" -- -m "smoke"

test-gpu: ## Run GPU tests excluding the exhaustive slow matrix
	@$(PYTHON) scripts/run_wheel_tests.py --wheel "$(WHEEL)" -- -m "gpu and not slow"

test-full: ## Run the complete GPU test matrix
	@$(PYTHON) scripts/run_wheel_tests.py --wheel "$(WHEEL)" -- -m "gpu"

test-cov: ## Run the bounded suite with Python coverage
	@$(PYTHON) scripts/run_wheel_tests.py --wheel "$(WHEEL)" -- \
		-m "gpu and not slow" --cov=tf_kernel --cov-report=term-missing --cov-report=html

test-cpu: ## Run pure-Python tests without importing GPU test modules
	@$(PYTHON) -m pytest tests/ -v -m "not gpu"

test-wheel: ## Check wheel architecture contents and exported symbols
	@$(PYTHON) scripts/check_wheel_symbols.py

format: check-deps ## Format and lint the standalone tf-kernel Python project
	@echo "Formatting Python imports with isort..."
	@find tf_kernel tests scripts benchmark -name '*.py' -exec $(PYTHON) -m isort {} +
	@echo "Applying safe Ruff fixes..."
	@$(PYTHON) -m ruff check tf_kernel tests scripts --fix

format-check: check-deps ## Check standalone tf-kernel formatting without modifying files
	@echo "Checking Python imports with isort..."
	@find tf_kernel tests benchmark -name '*.py' -exec $(PYTHON) -m isort --check-only --diff {} +
	@echo "Running Ruff..."
	@$(PYTHON) -m ruff check tf_kernel tests scripts

docs: ## Build documentation
	@cd docs && LC_ALL=C.UTF-8 LANG=C.UTF-8 make html

docs-clean: ## Clean documentation build
	@cd docs && make clean

docs-serve: ## Serve documentation locally (requires sphinx-autobuild)
	@cd docs && make livehtml

update: ## Update version numbers across project files. Usage: make update <new_version>
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Version required. Usage: make update <new_version>"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/set_version.py "$(filter-out $@,$(MAKECMDGOALS))"

%:
	@:
