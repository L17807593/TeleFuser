# TeleFuser Model Loading Guide

This document explains how to load internally implemented models using `ModuleManager` in TeleFuser.

## Overview

TeleFuser adopts a **Hash-based automatic model recognition** mechanism. The system automatically identifies model types and initializes corresponding model classes by computing MD5 hash values of model weight file keys. This design ensures complete control over internally implemented models and prevents unexpected errors.

## Core Concepts

### ModuleManager

`ModuleManager` is TeleFuser's model loading manager, responsible for:
- Automatic model type identification (via weight hash)
- Loading and initializing model weights
- Managing the lifecycle of multiple models

### Hash Matching Mechanism

```
Model File → Extract state_dict keys → Compute MD5 hash → Match registry entry → Initialize corresponding model class
```

Hash registrations live beside the model classes that own them. `ModelRegistry` discovers modules that call
`register_model_config()` on first use; see `telefuser/core/model_registry.py`.

## Quick Start

### Basic Usage

```python
from telefuser.core.module_manager import ModuleManager
import torch

# Create ModuleManager instance
module_manager = ModuleManager(
    torch_dtype=torch.bfloat16,
    device="cpu"  # Load on CPU first, offload later
)

# Load model (auto-recognize type)
module_manager.load_model("/path/to/model.safetensors")

# Or use load_models for batch loading
module_manager.load_models([
    "/path/to/vae.safetensors",
    "/path/to/text_encoder.safetensors",
])
```

### Using in Pipeline

```python
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline, Wan21VideoPipelineConfig

# 1. Load models
module_manager = ModuleManager(device="cpu")
module_manager.load_models([
    "/path/to/clip_encoder.pth",
    "/path/to/vae.safetensors",
    "/path/to/dit.safetensors",
    "/path/to/text_encoder.safetensors",
], torch_dtype=torch.bfloat16)

# 2. Initialize Pipeline
pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
pipe_config = Wan21VideoPipelineConfig()
pipe.init(module_manager, pipe_config)

# 3. Get specific models (optional)
vae_model = module_manager.fetch_module("wan_video_vae")
text_encoder = module_manager.fetch_module("wan_video_text_encoder")
```

## Advanced Usage

### Specifying Data Types

Different data types can be specified for different models:

```python
# Image Encoder uses float16
module_manager.load_models(
    ["/path/to/image_encoder.pth"],
    torch_dtype=torch.float16
)

# DiT and VAE use bfloat16
module_manager.load_models(
    ["/path/to/dit.safetensors", "/path/to/vae.safetensors"],
    torch_dtype=torch.bfloat16
)
```

### Low Memory Loading

Enable `low_cpu_mem_usage` to reduce CPU memory consumption:

```python
module_manager.load_model(
    "/path/to/large_model.safetensors",
    low_cpu_mem_usage=True  # Don't copy to CPU, load directly to target device
)
```

### Multi-File Model Loading

For sharded models (e.g., sharded safetensors):

```python
module_manager.load_model([
    "/path/to/model-00001-of-00007.safetensors",
    "/path/to/model-00002-of-00007.safetensors",
    # ... other shards
], torch_dtype=torch.bfloat16)
```

### Fetching Loaded Models

```python
# Get single model
vae = module_manager.fetch_module("wan_video_vae")

# Get model with its source path
vae, path = module_manager.fetch_module("wan_video_vae", require_model_path=True)

# When multiple models have the same name, specify index
dit = module_manager.fetch_module("wan_video_dit", index=0)
```

### HuggingFace Model Loading

For models without a registered hash entry, use HuggingFace loading:

```python
# Load from HuggingFace
module_manager.load_from_huggingface(
    module_path="stabilityai/stable-diffusion-xl-base-1.0",
    module_source="diffusers",  # or "transformers"
    module_name="sdxl_unet",
    torch_dtype=torch.bfloat16,
)
```

## Supported Model Formats

ModuleManager supports the following model file formats:

| Format | Extension | Description |
|------|--------|------|
| Safetensors | `.safetensors` | Recommended format, safe and efficient |
| PyTorch | `.bin`, `.pt`, `.pth`, `.ckpt` | Standard PyTorch format |

## Troubleshooting

### Model Not Recognized

If the model cannot be automatically recognized, possible reasons:

1. **Model not registered**
   - Check the module that owns the model class for a `register_model_config()` call
   - If it's a new model, follow the [development guide](./adding_new_model.md) to add its registration

2. **Model file corrupted or incomplete**
   - Verify file integrity
   - Re-download model files

3. **Unsupported format used**
   - Convert to `.safetensors` format

### Out of Memory

```python
# Use low memory mode
module_manager.load_model(
    "/path/to/model.safetensors",
    low_cpu_mem_usage=True
)

# Or load to CPU first, then offload
module_manager = ModuleManager(device="cpu")
module_manager.load_model(...)
# Then set offload strategy in Pipeline configuration
```

### Hash Mismatch

If you see hash in logs but it doesn't match:

```
load model /path/to/model.safetensors with state hash xxxxxxxxxx
```

This means the model has no matching registry entry. You need to:
1. Use `weight_viewer.py` tool to calculate hash:
   ```bash
   python tools/viewer/weight_viewer.py /path/to/model.safetensors --quiet
   ```
2. Follow the development guide to add a model registration

## Best Practices

1. **Always load models on CPU**
   ```python
   module_manager = ModuleManager(device="cpu")
   ```
   Let Pipeline handle moving models to GPU and managing offload.

2. **Choose appropriate data types**
   - Image Encoder: `float16` is usually sufficient
   - DiT/VAE/Text Encoder: `bfloat16` provides better numerical stability
   - For FP8 quantization, use `float8_e4m3fn` when loading

3. **Batch load related models**
   ```python
   # Good: Load related models at once
   module_manager.load_models([vae_path, dit_path, text_encoder_path])
   
   # Avoid: Multiple individual calls (unless different dtypes needed)
   ```

4. **Use Safetensors format**
   - Faster loading
   - More secure (prevents code execution)
   - Better cross-platform compatibility

5. **Use weight_viewer.py tool**
   ```bash
   # Analyze before adding new models
   python tools/viewer/weight_viewer.py /path/to/new_model.safetensors
   ```

## Related Documentation

- [Adding New Model Development Guide](./adding_new_model.md)
- [Hash Configuration Management Guide](./hash_config_management.md)
- [Service Guide](./service.md)
