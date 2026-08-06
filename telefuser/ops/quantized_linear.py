"""FP8 quantized linear layer implementation using tf-kernel."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from telefuser.utils.logging import logger

try:
    import tf_kernel
except ImportError:
    tf_kernel = None
from telefuser.platforms import current_platform


class LinearFP8(nn.Module):
    """FP8 quantized linear layer with per-channel scaling.

    Uses FP8_e4m3 format for weights and tf-kernel per-token activation
    quantization.
    """

    def __init__(self, original_linear: nn.Linear, data_type: torch.dtype) -> None:
        super().__init__()
        out_features = original_linear.out_features
        in_features = original_linear.in_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=data_type))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        device = original_linear.weight.device
        self.weight_scale = nn.Parameter(torch.FloatTensor(out_features, 1).zero_().to(device))

    def act_quant_fp8_perchannel_sym_tf(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize activations using custom tf-kernel."""
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device, requires_grad=False)
        input_tensor_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device, requires_grad=False)
        tf_kernel.tf_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the FP8 linear operation with tf-kernel."""
        if tf_kernel is not None:
            return self.forward_tf(x)
        raise RuntimeError("please install tf-kernel to enable FP8 linear")

    def forward_tf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using tf-kernel backend."""
        x_dim = x.dim()
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        output_dtype = x.dtype
        if torch.is_autocast_enabled():
            output_dtype = torch.get_autocast_dtype(current_platform.device_type)
        input_tensor_quant, input_tensor_scale = self.act_quant_fp8_perchannel_sym_tf(x)
        weight_scale = self.weight_scale
        if self.weight_scale.dtype != torch.float32:
            weight_scale = self.weight_scale.to(torch.float32)
        if self.bias.dtype != output_dtype:
            self.bias.data = self.bias.to(output_dtype)

        output_tensor = tf_kernel.fp8_scaled_mm(
            input_tensor_quant,
            self.weight.transpose(1, 0),
            input_tensor_scale,
            weight_scale,
            output_dtype,
            self.bias if self.bias is not None else None,
        )
        if x_dim == 3:
            output_tensor = output_tensor.reshape(batch_num, -1, output_tensor.shape[1])
        output_tensor = output_tensor.to(output_dtype)
        return output_tensor


def convert_params_to_buffers(model: nn.Module, ignore_dtype: torch.dtype = torch.float8_e4m3fn) -> nn.Module:
    """Convert Parameters to Buffers, except for specified dtype.

    Converts model parameters to buffers (non-trainable) to reduce
    memory overhead during inference. Skips FP8 weights.
    """
    logger.info(f"convert dtype != {ignore_dtype} params to buffer")

    def _process_module(module: nn.Module) -> None:
        params_to_convert = OrderedDict()
        for name, param in list(module.named_parameters(recurse=False)):
            if hasattr(param, "dtype") and param.dtype != ignore_dtype:
                params_to_convert[name] = param.data.clone()
        for name, data in params_to_convert.items():
            delattr(module, name)
            module.register_buffer(name, data)

        for child_name, child_module in module.named_children():
            _process_module(child_module)

    _process_module(model)
    return model


def replace_linear_layers(module: nn.Module, quant_type: torch.dtype) -> None:
    """Recursively replace all Linear layers with FP8 quantized versions."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Directly replace Linear layer
            setattr(module, name, LinearFP8(child, quant_type))
        elif isinstance(child, nn.Sequential):
            # Process Linear layers in Sequential
            for idx, sub_module in enumerate(child):
                if isinstance(sub_module, nn.Linear):
                    child[idx] = LinearFP8(sub_module, quant_type)
        else:
            # Recursively process submodules
            replace_linear_layers(child, quant_type)
