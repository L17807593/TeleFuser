# 添加新 Stage 开发指南

本文档介绍如何为 TeleFuser 创建新的 Pipeline Stage，包括 Stage 的基本概念、实现步骤和最佳实践。

## 概述

在 TeleFuser 中，**Stage** 是 Pipeline 中的一个处理单元，负责执行特定的计算任务。每个 Stage 可以：

- 封装一个或多个模型
- 处理输入数据并产生输出
- 管理模型的生命周期（加载、卸载、并行化）
- 与其他 Stage 组合成完整的 Pipeline

### Stage 的类型

| 类型 | 描述 | 示例 |
|------|------|------|
| 模型 Stage | 包含深度学习模型，执行推理 | `RealESRGANStage`, `RiftVFIStage` |
| 处理 Stage | 不含模型，执行数据转换或保存 | `ArtifactSaveStage` |

## 快速开始

以下是一个最简单的 Stage 实现：

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

class MyCustomStage(BaseStage):
    """自定义 Stage 示例"""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        # 从 ModuleManager 获取模型
        self.my_model = module_manager.fetch_module("my_model")
        # 注册模型名称（用于自动卸载）
        self.model_names = ["my_model"]

    @with_model_offload(["my_model"])
    @torch.inference_mode()
    def process(self, input_data):
        """处理输入数据"""
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            output = self.my_model(input_data.to(self.device))
        return output
```

## 详细步骤

### 步骤 1：创建 Stage 类文件

在 `telefuser/pipelines/` 目录下创建 Stage 文件。建议按功能模块组织：

```
telefuser/pipelines/
├── common/           # 通用 Stage（如超分辨率、帧插值）
│   ├── realesrgan_upscale.py
│   └── rift_vfi.py
├── wan_video/        # Wan Video 相关 Stage
├── qwen_image/       # Qwen Image 相关 Stage
└── ...
```

### 步骤 2：实现 Stage 类

继承 `BaseStage` 并实现必要的初始化和处理方法：

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
    """图像超分辨率 Stage。

    使用自定义模型将图像放大到更高分辨率。
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        """初始化 Stage。

        Args:
            name: Stage 名称，用于日志和调试
            module_manager: 模型管理器，用于获取已加载的模型
            model_runtime_config: 模型运行时配置
        """
        super().__init__(name, model_runtime_config)

        # 从 ModuleManager 获取模型
        # 注意：模型需要预先通过 module_manager.load_model() 加载
        self.upscale_model = module_manager.fetch_module("upscale_model")

        # 注册模型名称列表
        # 这用于 @with_model_offload 装饰器自动管理模型加载/卸载
        self.model_names = ["upscale_model"]

    @with_model_offload(["upscale_model"])
    @ProfilingContext4Debug("my_upscale")
    @torch.inference_mode()
    def process(
        self,
        input_images: List[Image.Image],
        scale_factor: int = 4,
    ) -> List[Image.Image]:
        """处理图像超分辨率。

        Args:
            input_images: 输入的 PIL Image 列表
            scale_factor: 放大倍数

        Returns:
            放大后的 PIL Image 列表
        """
        if not input_images:
            return input_images

        # 转换 PIL 图像为 Tensor [N, H, W, C]，范围 [0, 1]
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
            for image in input_images
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)

        # 执行推理
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscale_model.upscale(
                src_tensor,
                scale_factor=scale_factor,
                device=self.device.type
            )

        # 转换回 PIL 图像
        frames = ((result_tensor.float()) * 255).clip(0, 255).numpy().astype(np.uint8)
        result_images = [Image.fromarray(frame) for frame in frames]

        return result_images
```

### 步骤 3：理解 BaseStage 的关键属性

继承 `BaseStage` 后，以下属性自动可用：

```python
class BaseStage:
    def __init__(self, name: str, model_runtime_config: ModelRuntimeConfig):
        self.name = name                    # Stage 名称
        self.model_runtime_config = config  # 运行时配置
        self.torch_dtype = config.torch_dtype  # 数据类型（如 torch.bfloat16）
        self.device_type = config.device_type  # 设备类型（如 "cuda"）
        self.device = torch.device(...)       # 具体设备对象
        self.model_names = []                 # 模型名称列表（需要手动设置）
        self.onload_models_flag = False       # 模型加载状态标志
```

### 步骤 4：使用装饰器

#### `@with_model_offload`

自动管理模型的加载和卸载：

```python
@with_model_offload(["model_a", "model_b"])
def process(self, input_data):
    # 方法执行前：模型自动加载到 GPU
    # 方法执行后：模型自动卸载到 CPU（如果启用了 offload）
    pass
```

**工作原理**：

1. 方法执行前，检查模型是否已加载或是否需要重新加载
2. 如果需要，将模型从 CPU 移动到 GPU
3. 执行方法体
4. 方法结束后，如果配置了 CPU offload，将模型移回 CPU

#### `@ProfilingContext4Debug`

添加性能分析日志：

```python
@ProfilingContext4Debug("stage_name")
def process(self, input_data):
    # 自动记录执行时间
    pass
```

#### `@torch.inference_mode`

禁用梯度计算，节省显存：

```python
@torch.inference_mode()
def process(self, input_data):
    # 在此区域内，所有操作都不会被跟踪梯度
    pass
```

### 步骤 5：添加模型支持

Stage 使用的模型需要先添加到 TeleFuser 系统。详细步骤请参考 [添加新模型开发指南](./adding_new_model.md)。

简要流程：

1. **实现模型类**：创建继承 `BaseModel` 的模型类
2. **实现 StateDictConverter**：处理权重格式转换
3. **计算模型 Hash**：使用 `weight_viewer.py` 工具
4. **注册模型检测信息**：在定义模型类、拥有该 checkpoint 格式的模块中调用 `register_model_config()`

```bash
# 计算模型 hash
python tools/viewer/weight_viewer.py /path/to/model.safetensors --quiet
```

### 步骤 6：在 Pipeline 中使用 Stage

```python
from telefuser.core.module_manager import ModuleManager
from telefuser.core.config import ModelRuntimeConfig
from telefuser.pipelines.common.my_upscale_stage import MyUpscaleStage

# 创建 ModuleManager 并加载模型
module_manager = ModuleManager(device="cuda", torch_dtype=torch.bfloat16)
module_manager.load_model("/path/to/upscale_model.safetensors")

# 创建配置
config = ModelRuntimeConfig(
    torch_dtype=torch.bfloat16,
    device_type="cuda",
    device_id=0,
)

# 创建 Stage
upscale_stage = MyUpscaleStage(
    name="upscale",
    module_manager=module_manager,
    model_runtime_config=config,
)

# 使用 Stage
from PIL import Image
input_images = [Image.open("input.jpg")]
output_images = upscale_stage.process(input_images)
```

## 高级功能

### 多模型 Stage

当 Stage 需要多个模型时：

```python
class MultiModelStage(BaseStage):
    def __init__(self, name, module_manager, model_runtime_config):
        super().__init__(name, model_runtime_config)

        # 获取多个模型
        self.encoder = module_manager.fetch_module("encoder")
        self.decoder = module_manager.fetch_module("decoder")

        # 注册所有模型名称
        self.model_names = ["encoder", "decoder"]

    @with_model_offload(["encoder", "decoder"])
    def process(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded
```

### 条件性模型卸载

使用不同的装饰器参数控制卸载行为：

```python
# 始终保持模型在 GPU 上
@with_model_offload(["model"])
def process_keep_on_gpu(self, input_data):
    pass

# 手动控制加载/卸载
def process_manual(self, input_data):
    self.onload_models()  # 手动加载
    try:
        result = self.model(input_data)
    finally:
        self.offload_models()  # 手动卸载
    return result
```

### 处理不同输入类型

Stage 可以提供多个处理方法以支持不同输入类型：

```python
class VersatileStage(BaseStage):
    @with_model_offload(["model"])
    @torch.inference_mode()
    def process_pil(self, images: List[Image.Image]):
        """处理 PIL 图像列表"""
        # 转换并处理
        pass

    @with_model_offload(["model"])
    @torch.inference_mode()
    def process_tensor(self, tensor: torch.Tensor):
        """处理 Tensor"""
        # 直接处理
        pass
```

### 无模型 Stage

对于不需要模型的处理 Stage，可以不继承 `BaseStage`：

```python
class ArtifactSaveStage:
    """保存结果的 Stage（无模型）"""

    def __init__(self, name: str = "artifact_save"):
        self.name = name

    def process(self, frames, output_path: str, fps: int = 24):
        """保存帧到视频文件"""
        # 实现保存逻辑
        pass
```

## 完整示例：RealESRGAN Stage

以下是 `RealESRGANStage` 的完整实现，作为参考：

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
    """图像超分辨率 Stage。

    使用 Real-ESRGAN 模型进行图像放大，支持 SRVGGNetCompact（轻量级）
    和 RRDBNet（较重，更高质量）架构。
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
        """放大 PIL 图像列表。

        Args:
            input_images: 待放大的 PIL Image 列表

        Returns:
            放大后的 PIL Image 列表
        """
        if not input_images:
            return input_images

        # 转换 PIL 图像为 Tensor [N, H, W, C]，范围 [0, 1]
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
            for image in input_images
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)

        # 执行超分辨率
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscaler_model.upscale_frames(
                src_tensor, device=self.device.type
            )

        # 转换回 PIL 图像
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
        """放大 Tensor 图像。

        Args:
            input_tensor: 输入 Tensor [N, H, W, C]，范围 [0, 1]

        Returns:
            放大后的 Tensor [N, H*scale, W*scale, C]，范围 [0, 1]
        """
        if input_tensor.numel() == 0:
            return input_tensor

        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscaler_model.upscale_frames(
                input_tensor, device=self.device.type
            )

        return result_tensor
```

## 最佳实践

### 1. 命名规范

- Stage 类名以 `Stage` 结尾：`RealESRGANStage`, `RiftVFIStage`
- 使用描述性名称：`VideoEncodeStage` 而非 `VidEncStage`
- 模型属性使用 `_model` 后缀：`upscale_model`, `vfi_model`

### 2. 输入验证

```python
def process(self, input_images: List[Image.Image]):
    # 检查空输入
    if not input_images:
        return input_images

    # 检查输入类型
    if not all(isinstance(img, Image.Image) for img in input_images):
        raise TypeError("All inputs must be PIL Images")

    # 继续处理...
```

### 3. 类型注解

```python
from typing import List
from PIL import Image

def process(self, input_images: List[Image.Image]) -> List[Image.Image]:
    pass

def process_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
    pass
```

### 4. 文档字符串

```python
def process(self, input_data, param1=10):
    """简短描述。

    详细描述（可选）。

    Args:
        input_data: 输入数据描述
        param1: 参数描述，默认值为 10

    Returns:
        返回值描述

    Raises:
        ValueError: 异常情况描述
    """
    pass
```

### 5. 资源管理

```python
@with_model_offload(["model"])
@torch.inference_mode()
def process(self, input_data):
    # 使用 autocast 进行混合精度
    with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
        output = self.model(input_data)

    # 及时清理中间结果
    del input_data
    return output
```

## 测试 Stage

创建测试脚本验证 Stage 功能：

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
    """创建 ModuleManager 并加载测试模型"""
    manager = ModuleManager(device="cpu", torch_dtype=torch.float32)
    manager.load_model("/path/to/test_model.safetensors")
    return manager


@pytest.fixture
def model_config():
    """创建测试配置"""
    return ModelRuntimeConfig(
        torch_dtype=torch.float32,
        device_type="cpu",
        device_id=0,
    )


def test_stage_initialization(module_manager, model_config):
    """测试 Stage 初始化"""
    stage = MyUpscaleStage("test", module_manager, model_config)
    assert stage.name == "test"
    assert "upscale_model" in stage.model_names


def test_stage_process(module_manager, model_config):
    """测试 Stage 处理"""
    stage = MyUpscaleStage("test", module_manager, model_config)

    # 创建测试图像
    test_images = [Image.new("RGB", (64, 64), color="red")]

    # 执行处理
    result = stage.process(test_images)

    # 验证结果
    assert len(result) == 1
    assert result[0].size == (256, 256)  # 4x 放大
```

## 相关文档

- [添加新模型开发指南](./adding_new_model.md) - 如何添加新的模型支持
- [模型加载用户指南](./model_loading.md) - 模型加载和配置
- [CPU 卸载指南](./offload.md) - 显存优化策略
- [并行推理指南](./parallel.md) - 多 GPU 推理配置
