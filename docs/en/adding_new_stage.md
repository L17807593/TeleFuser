# Adding New Stage Development Guide

This document describes how to create new Pipeline Stages for TeleFuser, including basic concepts, implementation steps, and best practices.

## Overview

In TeleFuser, a **Stage** is a processing unit in a Pipeline that executes specific computational tasks. Each Stage can:

- Encapsulate one or more models
- Process input data and produce output
- Manage model lifecycle (loading, unloading, parallelization)
- Compose with other Stages to form a complete Pipeline

### Types of Stages

| Type | Description | Examples |
|------|-------------|----------|
| Model Stage | Contains deep learning models for inference | `RealESRGANStage`, `RiftVFIStage` |
| Processing Stage | No models, performs data transformation or saving | `ArtifactSaveStage` |

## Quick Start

Here's a minimal Stage implementation:

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

class MyCustomStage(BaseStage):
    """Custom Stage example"""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        # Get model from ModuleManager
        self.my_model = module_manager.fetch_module("my_model")
        # Register model names (for automatic offloading)
        self.model_names = ["my_model"]

    @with_model_offload(["my_model"])
    @torch.inference_mode()
    def process(self, input_data):
        """Process input data"""
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            output = self.my_model(input_data.to(self.device))
        return output
```

## Detailed Steps

### Step 1: Create Stage Class File

Create Stage files under `telefuser/pipelines/`. Organize by functional modules:

```
telefuser/pipelines/
├── common/           # Common Stages (e.g., super-resolution, frame interpolation)
│   ├── realesrgan_upscale.py
│   └── rift_vfi.py
├── wan_video/        # Wan Video related Stages
├── qwen_image/       # Qwen Image related Stages
└── ...
```

### Step 2: Implement Stage Class

Inherit from `BaseStage` and implement necessary initialization and processing methods:

```python
# telefuser/pipelines/common/my_upscale_stage.py

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.profiler import ProfilingContext4Debug


class MyUpscaleStage(BaseStage):
    """Image super-resolution Stage.

    Upscales images to higher resolution using a custom model.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        """Initialize the Stage.

        Args:
            name: Stage name for logging and debugging
            module_manager: Model manager for fetching loaded models
            model_runtime_config: Model runtime configuration
        """
        super().__init__(name, model_runtime_config)

        # Get model from ModuleManager
        # Note: Model must be pre-loaded via module_manager.load_model()
        self.upscale_model = module_manager.fetch_module("upscale_model")

        # Register model names list
        # This is used by @with_model_offload decorator for automatic model load/unload
        self.model_names = ["upscale_model"]

    @with_model_offload(["upscale_model"])
    @ProfilingContext4Debug("my_upscale")
    @torch.inference_mode()
    def process(
        self,
        input_images: List[Image.Image],
        scale_factor: int = 4,
    ) -> List[Image.Image]:
        """Process image super-resolution.

        Args:
            input_images: List of input PIL Images
            scale_factor: Upscale factor

        Returns:
            List of upscaled PIL Images
        """
        if not input_images:
            return input_images

        # Convert PIL images to Tensor [N, H, W, C], range [0, 1]
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
            for image in input_images
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)

        # Execute inference
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscale_model.upscale(
                src_tensor,
                scale_factor=scale_factor,
                device=self.device.type
            )

        # Convert back to PIL images
        frames = ((result_tensor.float()) * 255).clip(0, 255).numpy().astype(np.uint8)
        result_images = [Image.fromarray(frame) for frame in frames]

        return result_images
```

### Step 3: Understand BaseStage Key Attributes

After inheriting `BaseStage`, the following attributes are automatically available:

```python
class BaseStage:
    def __init__(self, name: str, model_runtime_config: ModelRuntimeConfig):
        self.name = name                    # Stage name
        self.model_runtime_config = config  # Runtime configuration
        self.torch_dtype = config.torch_dtype  # Data type (e.g., torch.bfloat16)
        self.device_type = config.device_type  # Device type (e.g., "cuda")
        self.device = torch.device(...)       # Concrete device object
        self.model_names = []                 # Model names list (needs manual setting)
        self.onload_models_flag = False       # Model loading status flag
```

### Step 4: Using Decorators

#### `@with_model_offload`

Automatically manages model loading and unloading:

```python
@with_model_offload(["model_a", "model_b"])
def process(self, input_data):
    # Before execution: models automatically loaded to GPU
    # After execution: models automatically offloaded to CPU (if offload is enabled)
    pass
```

**How it works**:

1. Before method execution, checks if models are loaded or need reloading
2. If needed, moves models from CPU to GPU
3. Executes method body
4. After method completes, if CPU offload is configured, moves models back to CPU

#### `@ProfilingContext4Debug`

Adds performance profiling logs:

```python
@ProfilingContext4Debug("stage_name")
def process(self, input_data):
    # Automatically logs execution time
    pass
```

#### `@torch.inference_mode`

Disables gradient computation to save memory:

```python
@torch.inference_mode()
def process(self, input_data):
    # Within this block, no operations are tracked for gradients
    pass
```

### Step 5: Add Model Support

Models used by the Stage need to be added to TeleFuser first. For detailed steps, refer to [Adding New Model Development Guide](./adding_new_model.md).

Brief overview:

1. **Implement Model Class**: Create a model class inheriting `BaseModel`
2. **Implement StateDictConverter**: Handle weight format conversion
3. **Calculate Model Hash**: Use `weight_viewer.py` tool
4. **Register model detection**: Add `register_model_config()` beside the model class that owns the checkpoint format

```bash
# Calculate model hash
python tools/viewer/weight_viewer.py /path/to/model.safetensors --quiet
```

### Step 6: Using Stage in Pipeline

```python
from telefuser.core.module_manager import ModuleManager
from telefuser.core.config import ModelRuntimeConfig
from telefuser.pipelines.common.my_upscale_stage import MyUpscaleStage

# Create ModuleManager and load model
module_manager = ModuleManager(device="cuda", torch_dtype=torch.bfloat16)
module_manager.load_model("/path/to/upscale_model.safetensors")

# Create configuration
config = ModelRuntimeConfig(
    torch_dtype=torch.bfloat16,
    device_type="cuda",
    device_id=0,
)

# Create Stage
upscale_stage = MyUpscaleStage(
    name="upscale",
    module_manager=module_manager,
    model_runtime_config=config,
)

# Use Stage
from PIL import Image
input_images = [Image.open("input.jpg")]
output_images = upscale_stage.process(input_images)
```

## Advanced Features

### Multi-Model Stage

When a Stage requires multiple models:

```python
class MultiModelStage(BaseStage):
    def __init__(self, name, module_manager, model_runtime_config):
        super().__init__(name, model_runtime_config)

        # Get multiple models
        self.encoder = module_manager.fetch_module("encoder")
        self.decoder = module_manager.fetch_module("decoder")

        # Register all model names
        self.model_names = ["encoder", "decoder"]

    @with_model_offload(["encoder", "decoder"])
    def process(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded
```

### Conditional Model Offloading

Use different decorator parameters to control offloading behavior:

```python
# Always keep model on GPU
@with_model_offload(["model"])
def process_keep_on_gpu(self, input_data):
    pass

# Manual load/unload control
def process_manual(self, input_data):
    self.onload_models()  # Manual load
    try:
        result = self.model(input_data)
    finally:
        self.offload_models()  # Manual unload
    return result
```

### Handling Different Input Types

A Stage can provide multiple processing methods for different input types:

```python
class VersatileStage(BaseStage):
    @with_model_offload(["model"])
    @torch.inference_mode()
    def process_pil(self, images: List[Image.Image]):
        """Process PIL image list"""
        # Convert and process
        pass

    @with_model_offload(["model"])
    @torch.inference_mode()
    def process_tensor(self, tensor: torch.Tensor):
        """Process Tensor"""
        # Direct processing
        pass
```

### Non-Model Stage

For processing Stages without models, you don't need to inherit `BaseStage`:

```python
class ArtifactSaveStage:
    """Stage for saving results (no model)"""

    def __init__(self, name: str = "artifact_save"):
        self.name = name

    def process(self, frames, output_path: str, fps: int = 24):
        """Save frames to video file"""
        # Implement save logic
        pass
```

## Complete Example: RealESRGAN Stage

Here's the complete implementation of `RealESRGANStage` for reference:

```python
# telefuser/pipelines/common/realesrgan_upscale.py

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.realesrgan import RealESRGAN
from telefuser.utils.profiler import ProfilingContext4Debug


class RealESRGANStage(BaseStage):
    """Image super-resolution Stage using Real-ESRGAN.

    Upscales images using Real-ESRGAN model, supporting both SRVGGNetCompact
    (lightweight) and RRDBNet (heavier, higher quality) architectures.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.upscaler_model: RealESRGAN = module_manager.fetch_module("upscaler_model")
        self.model_names = ["upscaler_model"]

    @with_model_offload(["upscaler_model"])
    @ProfilingContext4Debug("realesrgan_upscale")
    @torch.inference_mode()
    def process(
        self,
        input_images: List[Image.Image],
    ) -> List[Image.Image]:
        """Upscale a list of PIL images.

        Args:
            input_images: List of PIL Image objects to upscale

        Returns:
            List of upscaled PIL Image objects
        """
        if not input_images:
            return input_images

        # Convert PIL images to tensor [N, H, W, C] in range [0, 1]
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
            for image in input_images
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)

        # Upscale frames
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscaler_model.upscale_frames(
                src_tensor, device=self.device.type
            )

        # Convert back to PIL images
        frames = ((result_tensor.float()) * 255).clip(0, 255).numpy().astype(np.uint8)
        result_images = [Image.fromarray(frame) for frame in frames]
        return result_images

    @with_model_offload(["upscaler_model"])
    @ProfilingContext4Debug("realesrgan_upscale_tensor")
    @torch.inference_mode()
    def process_tensor(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Upscale a tensor of images.

        Args:
            input_tensor: Input tensor [N, H, W, C] in range [0, 1]

        Returns:
            Upscaled tensor [N, H*scale, W*scale, C] in range [0, 1]
        """
        if input_tensor.numel() == 0:
            return input_tensor

        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscaler_model.upscale_frames(
                input_tensor, device=self.device.type
            )

        return result_tensor
```

## Best Practices

### 1. Naming Conventions

- Stage class names end with `Stage`: `RealESRGANStage`, `RiftVFIStage`
- Use descriptive names: `VideoEncodeStage` instead of `VidEncStage`
- Model attributes use `_model` suffix: `upscale_model`, `vfi_model`

### 2. Input Validation

```python
def process(self, input_images: List[Image.Image]):
    # Check empty input
    if not input_images:
        return input_images

    # Check input types
    if not all(isinstance(img, Image.Image) for img in input_images):
        raise TypeError("All inputs must be PIL Images")

    # Continue processing...
```

### 3. Type Annotations

```python
from typing import List
from PIL import Image

def process(self, input_images: List[Image.Image]) -> List[Image.Image]:
    pass

def process_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
    pass
```

### 4. Docstrings

```python
def process(self, input_data, param1=10):
    """Brief description.

    Detailed description (optional).

    Args:
        input_data: Input data description
        param1: Parameter description, default is 10

    Returns:
        Return value description

    Raises:
        ValueError: Exception condition description
    """
    pass
```

### 5. Resource Management

```python
@with_model_offload(["model"])
@torch.inference_mode()
def process(self, input_data):
    # Use autocast for mixed precision
    with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
        output = self.model(input_data)

    # Clean up intermediate results promptly
    del input_data
    return output
```

## Testing Stage

Create test scripts to verify Stage functionality:

```python
# tests/unit/pipelines/test_my_stage.py

import pytest
import torch
from PIL import Image

from telefuser.core.module_manager import ModuleManager
from telefuser.core.config import ModelRuntimeConfig
from telefuser.pipelines.common.my_upscale_stage import MyUpscaleStage


@pytest.fixture
def module_manager():
    """Create ModuleManager and load test model"""
    manager = ModuleManager(device="cpu", torch_dtype=torch.float32)
    manager.load_model("/path/to/test_model.safetensors")
    return manager


@pytest.fixture
def model_config():
    """Create test configuration"""
    return ModelRuntimeConfig(
        torch_dtype=torch.float32,
        device_type="cpu",
        device_id=0,
    )


def test_stage_initialization(module_manager, model_config):
    """Test Stage initialization"""
    stage = MyUpscaleStage("test", module_manager, model_config)
    assert stage.name == "test"
    assert "upscale_model" in stage.model_names


def test_stage_process(module_manager, model_config):
    """Test Stage processing"""
    stage = MyUpscaleStage("test", module_manager, model_config)

    # Create test images
    test_images = [Image.new("RGB", (64, 64), color="red")]

    # Execute processing
    result = stage.process(test_images)

    # Verify results
    assert len(result) == 1
    assert result[0].size == (256, 256)  # 4x upscale
```

## Related Documentation

- [Adding New Model Development Guide](./adding_new_model.md) - How to add new model support
- [Model Loading User Guide](./model_loading.md) - Model loading and configuration
- [CPU Offloading Guide](./offload.md) - Memory optimization strategies
- [Parallel Inference Guide](./parallel.md) - Multi-GPU inference configuration
