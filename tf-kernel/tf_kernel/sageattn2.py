"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import logging
import re
import subprocess
import warnings
from typing import Any, Optional

import torch

from tf_kernel.capabilities import sage_attention_backend

logger = logging.getLogger(__name__)

# Try to import triton modules (from sageattention package or local)
try:
    from .triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
    from .triton.attn_qk_int8_per_block import forward as attn_false
    from .triton.attn_qk_int8_per_block_causal import forward as attn_true
    from .triton.attn_qk_int8_per_block_causal_varlen import \
        forward as attn_true_varlen
    from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
    from .triton.quant_per_block_varlen import \
        per_block_int8 as per_block_int8_varlen_triton
    from .triton.quant_per_thread import \
        per_thread_int8 as per_thread_int8_triton

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    per_block_int8_triton = None
    per_block_int8_varlen_triton = None
    attn_false = None
    attn_true = None
    attn_false_varlen = None
    attn_true_varlen = None
    per_thread_int8_triton = None


@functools.lru_cache(maxsize=1)
def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major, minor
    except Exception as error:
        logger.debug("Failed to get CUDA version: %s", error)
    return None, None


@functools.lru_cache(maxsize=8)
def get_cuda_arch_versions():
    """Get CUDA architecture versions for all devices.

    This function is cached to avoid repeated device queries during torch.compile.
    The cache is limited to 8 entries to support systems with different GPU configurations.
    """
    if not torch.cuda.is_available():
        return []
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return tuple(cuda_archs)  # Use tuple for hashability


# ==============================================================================
# Quantization functions using torch.ops.tf_kernel
# ==============================================================================


def per_block_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale: Optional[float] = None,
    tensor_layout: str = "HND",
):
    """
    Quantize the query tensor `q` and the key tensor `k` with per block quantization.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[torch.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.

    sm_scale : Optional[float]
        The scale factor for the softmax operation. Default is ``head_dim**-0.5``.
        It will be multiplied by ``1.44269504`` to work together with the triton attention kernel.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // BLKQ]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK]`` with `float32` dtype.

    Note
    ----
    - The tensors `q` and `k` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    """

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32
    )

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    sm_scale *= 1.44269504

    torch.ops.tf_kernel.quant_per_block_int8_cuda_with_sm_scale(
        q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout
    )
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        torch.ops.tf_kernel.quant_per_block_int8_fuse_sub_mean_cuda(
            k, km, k_int8, k_scale, BLKK, _tensor_layout
        )
    else:
        torch.ops.tf_kernel.quant_per_block_int8_cuda(
            k, k_int8, k_scale, BLKK, _tensor_layout
        )

    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
    tensor_layout: str = "HND",
):
    """
    Quantize the query tensor `q` with per warp quantization and the key tensor `k` with per block quantization.
    Warp size of quantizing `q` is 16 or 32, with a block size of 64 or 128.
    Block size of quantizing `k` is 64 or 128.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[torch.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ)]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK]`` with `float32` dtype.

    Note
    ----
    - The tensors `q` and `k` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    """

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty(
        (b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32
    )

    torch.ops.tf_kernel.quant_per_warp_int8_cuda(
        q, q_int8, q_scale, BLKQ, WARPQ, _tensor_layout
    )

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        torch.ops.tf_kernel.quant_per_block_int8_fuse_sub_mean_cuda(
            k, km, k_int8, k_scale, BLKK, _tensor_layout
        )
    else:
        torch.ops.tf_kernel.quant_per_block_int8_cuda(
            k, k_int8, k_scale, BLKK, _tensor_layout
        )

    return q_int8, q_scale, k_int8, k_scale


def sub_mean(v: torch.Tensor, tensor_layout: str = "HND"):
    """
    Calculate the mean of the tensor `v` along the sequence length dimension and subtract it from `v`. Result is stored as fp16.

    Parameters
    ----------
    v : torch.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - The tensor `v_smoothed` with the mean subtracted and stored as fp16. Shape: Same as `v` with `float16` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with dtype same as `v`.

    Note
    ----
    - The tensors `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - The returned tensor `v_smoothed` will have dtype ``torch.float16`` regardless of the input dtype.
    - The returned mean tensor will have the same dtype as the input tensor.
    """

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    vm = v.mean(dim=1 if _tensor_layout == 0 else 2)

    v_smoothed = torch.empty(v.shape, dtype=torch.float16, device=v.device)

    # subtract mean and store the result as fp16
    torch.ops.tf_kernel.sub_mean_cuda(v, vm, v_smoothed, _tensor_layout)

    return v_smoothed, vm


def per_channel_fp8(
    v: torch.Tensor,
    tensor_layout: str = "HND",
    scale_max: float = 448.0,
    smooth_v: bool = True,
):
    """
    Transpose, pad and permute the tensor `v` and quantize it to fp8 with per channel quantization.
    `v` is first transposed along the head dimension and the sequence length dimension, then padded to a multiple of 64.
    After that, the tensor is permuted along the sequence length dimension by ``[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]``.
    The quantization is done per channel, with the scale value and smooth factor calculated per channel.

    Parameters
    ----------
    v : torch.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    scale_max : float
        The maximum scale value for the quantization. Default is 448.0 (upper bound of E4M3 data format).

    smooth_v : bool
        Whether to smooth the quantized tensor. Default is True.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        A tuple containing:
        - The quantized tensor `v_fp8`. Shape:
            - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, head_dim, (kv_len + 63) // 64 * 64]``, with `float8_e4m3fn` dtype.
            - If `tensor_layout` is "NHD": ``[batch_size, head_dim, num_kv_heads, (kv_len + 63) // 64 * 64]``, with `float8_e4m3fn` dtype.
        - The scale tensor of `v`. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.

    Note
    ----
    - The tensors `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - The returned mean tensor will be None if `smooth_v` is False. Otherwise it will have dtype ``torch.float32``.
    """

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty(
            (b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device
        )

    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty(
            (b, head_dim, h_kv, padded_len), dtype=v.dtype, device=v.device
        )

    torch.ops.tf_kernel.transpose_pad_permute_cuda(
        v, v_transposed_permutted, _tensor_layout
    )

    v_fp8 = torch.empty(
        v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device
    )

    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    vm = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)

    if smooth_v:
        torch.ops.tf_kernel.mean_scale_fuse_quant_cuda(
            v_transposed_permutted,
            v_fp8,
            vm,
            v_scale,
            kv_len,
            scale_max,
            _tensor_layout,
        )
        return v_fp8, v_scale, vm
    else:
        torch.ops.tf_kernel.scale_fuse_quant_cuda(
            v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout
        )
        return v_fp8, v_scale, None


# ==============================================================================
# Attention functions using torch.ops.tf_kernel
# ==============================================================================


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """

    major, minor = torch.cuda.get_device_capability(q.device)
    backend = sage_attention_backend(major, minor)
    if backend == "cuda_fp16":
        return sageattn_qk_int8_pv_fp16_cuda(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
            pv_accum_dtype="fp32",
        )
    if backend == "triton_fp16":
        return sageattn_qk_int8_pv_fp16_triton(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
        )
    if backend == "cuda_fp8_thread":
        return sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
            pv_accum_dtype="fp32+fp16",
        )
    if backend == "cuda_fp8_sm90":
        return sageattn_qk_int8_pv_fp8_cuda_sm90(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
        )
    if backend == "cuda_fp8_warp":
        return sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            qk_quant_gran="per_warp",
            sm_scale=sm_scale,
            return_lse=return_lse,
            pv_accum_dtype="fp32+fp16",
        )


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 quantization for Q and K, FP16 PV with FP16 accumulation, implemented using Triton.
    The FP16 accumulator is added to a FP32 buffer immediately after each iteration.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    quantization_backend : str
        The quantization backend, either "triton" or "cuda".
        "cuda" backend offers better performance due to kernel fusion.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    attn_mask : Optional[torch.Tensor]
        The attention mask tensor, of dtype bool or float32.
        Should be able to broadcast to the shape of the matrix qk^T.
        Default: None.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype
        ), "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = (
                    torch.matmul(
                        q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)
                    )
                    .squeeze(-1)
                    .to(torch.float32)
                )
            else:
                lse_correction = (
                    torch.matmul(q, km_broadcast.transpose(2, 3))
                    .squeeze(-1)
                    .to(torch.float32)
                )
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og**0.5)

    if quantization_backend == "triton":
        if not _TRITON_AVAILABLE or per_block_int8_triton is None:
            raise RuntimeError(
                "Triton backend requested but triton modules are not available"
            )
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(
            q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout
        )
    elif quantization_backend == "cuda":
        q_int8, q_scale, k_int8, k_scale = per_block_int8(
            q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout
        )
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")
    if is_causal:
        assert attn_mask is None, "Mask should be None for causal attention."
        if attn_true is None:
            raise RuntimeError("Triton attention modules are not available")
        o, lse = attn_true(
            q_int8,
            k_int8,
            v,
            q_scale,
            k_scale,
            tensor_layout=tensor_layout,
            output_dtype=dtype,
            return_lse=return_lse,
        )
    else:
        if attn_mask is not None:
            if tensor_layout == "HND":
                target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
            elif tensor_layout == "NHD":
                target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
            else:
                raise ValueError(f"tensor_layout {tensor_layout} not supported")
            try:
                attn_mask = attn_mask.expand(target_shape)
            except Exception:
                raise AssertionError(
                    f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}"
                )
        if attn_false is None:
            raise RuntimeError("Triton attention modules are not available")
        o, lse = attn_false(
            q_int8,
            k_int8,
            v,
            q_scale,
            k_scale,
            tensor_layout=tensor_layout,
            output_dtype=dtype,
            attn_mask=attn_mask,
            return_lse=return_lse,
        )

    o = o[..., :head_dim_og]

    if return_lse:
        return o, (
            lse / 1.44269504 + lse_correction * sm_scale
            if smooth_k
            else lse / 1.44269504
        )
    else:
        return o


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`.
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`.
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.

    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."
    assert (
        cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous()
    ), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.mean(dim=0, keepdim=True)
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og**0.5)

    if per_block_int8_varlen_triton is None:
        raise RuntimeError("Triton attention modules are not available")
    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = (
        per_block_int8_varlen_triton(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale=sm_scale,
        )
    )

    if is_causal:
        if attn_true_varlen is None:
            raise RuntimeError("Triton attention modules are not available")
        o = attn_true_varlen(
            q_int8,
            k_int8,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            q_scale,
            k_scale,
            cu_seqlens_q_scale,
            cu_seqlens_k_scale,
            output_dtype=dtype,
        )
    else:
        if attn_false_varlen is None:
            raise RuntimeError("Triton attention modules are not available")
        o = attn_false_varlen(
            q_int8,
            k_int8,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            q_scale,
            k_scale,
            cu_seqlens_q_scale,
            cu_seqlens_k_scale,
            output_dtype=dtype,
        )

    o = o[..., :head_dim_og]

    return o


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP16 PV with FP16/FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp16", "fp16+fp32" or "fp32".
        - "fp16": PV accumulation is done in fully in FP16. This is the fastest option but may lead to numerical instability. `smooth_v` option will increase the accuracy in cases when the value tensor has a large bias (like in CogVideoX-2b).
        - "fp32": PV accumulation is done in FP32. This is the most accurate option but may be slower than "fp16" due to CUDA core overhead.
        - "fp16+fp32": PV accumulation is done in FP16, but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32" or "fp16+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in [
        "per_warp",
        "per_thread",
    ], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = (
                    torch.matmul(
                        q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)
                    )
                    .squeeze(-1)
                    .to(torch.float32)
                )
            else:
                lse_correction = (
                    torch.matmul(q, km_broadcast.transpose(2, 3))
                    .squeeze(-1)
                    .to(torch.float32)
                )
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            q,
            k,
            km,
            tensor_layout=tensor_layout,
            BLKQ=128,
            WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32),
            BLKK=64,
        )
    elif qk_quant_gran == "per_thread":
        if per_thread_int8_triton is None:
            raise RuntimeError("Triton per_thread quantization is not available")
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
            q,
            k,
            km,
            tensor_layout=tensor_layout,
            BLKQ=128,
            WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32),
            BLKK=64,
            WARPK=64,
        )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == "fp32":
        v = v.to(torch.float16)
        lse = torch.ops.tf_kernel.qk_int8_sv_f16_accum_f32_attn(
            q_int8,
            k_int8,
            v,
            o,
            q_scale,
            k_scale,
            _tensor_layout,
            _is_caual,
            _qk_quant_gran,
            sm_scale,
            _return_lse,
        )
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = torch.ops.tf_kernel.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
                q_int8,
                k_int8,
                smoothed_v,
                o,
                q_scale,
                k_scale,
                vm,
                _tensor_layout,
                _is_caual,
                _qk_quant_gran,
                sm_scale,
                _return_lse,
            )
        else:
            v = v.to(torch.float16)
            lse = torch.ops.tf_kernel.qk_int8_sv_f16_accum_f16_attn(
                q_int8,
                k_int8,
                v,
                o,
                q_scale,
                k_scale,
                _tensor_layout,
                _is_caual,
                _qk_quant_gran,
                sm_scale,
                _return_lse,
            )
    elif pv_accum_dtype == "fp16+fp32":
        v = v.to(torch.float16)
        lse = torch.ops.tf_kernel.qk_int8_sv_f16_accum_f16_attn_inst_buf(
            q_int8,
            k_int8,
            v,
            o,
            q_scale,
            k_scale,
            _tensor_layout,
            _is_caual,
            _qk_quant_gran,
            sm_scale,
            _return_lse,
        )
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, (
            lse / 1.44269504 + lse_correction * sm_scale
            if smooth_k
            else lse / 1.44269504
        )
    else:
        return o


def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in [
        "per_warp",
        "per_thread",
    ], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = (
                    torch.matmul(
                        q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)
                    )
                    .squeeze(-1)
                    .to(torch.float32)
                )
            else:
                lse_correction = (
                    torch.matmul(q, km_broadcast.transpose(2, 3))
                    .squeeze(-1)
                    .to(torch.float32)
                )
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64
        )
    elif qk_quant_gran == "per_thread":
        if per_thread_int8_triton is None:
            raise RuntimeError("Triton per_thread quantization is not available")
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
            q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64
        )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype == "fp32+fp32" and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == "fp32+fp16" and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored.")
        smooth_v = False

    quant_v_scale_max = 448.0
    if pv_accum_dtype == "fp32+fp16":
        quant_v_scale_max = 2.25

    v_fp8, v_scale, vm = per_channel_fp8(
        v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v
    )

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = torch.ops.tf_kernel.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(
                q_int8,
                k_int8,
                v_fp8,
                o,
                q_scale,
                k_scale,
                v_scale,
                vm,
                _tensor_layout,
                _is_caual,
                _qk_quant_gran,
                sm_scale,
                _return_lse,
            )
        else:
            lse = torch.ops.tf_kernel.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
                q_int8,
                k_int8,
                v_fp8,
                o,
                q_scale,
                k_scale,
                v_scale,
                _tensor_layout,
                _is_caual,
                _qk_quant_gran,
                sm_scale,
                _return_lse,
            )
    elif pv_accum_dtype == "fp32+fp32":
        lse = torch.ops.tf_kernel.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            o,
            q_scale,
            k_scale,
            v_scale,
            _tensor_layout,
            _is_caual,
            _qk_quant_gran,
            sm_scale,
            _return_lse,
        )
    elif pv_accum_dtype == "fp32+fp16":
        lse = torch.ops.tf_kernel.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            o,
            q_scale,
            k_scale,
            v_scale,
            _tensor_layout,
            _is_caual,
            _qk_quant_gran,
            sm_scale,
            _return_lse,
        )

    o = o[..., :head_dim_og]

    if return_lse:
        return o, (
            lse / 1.44269504 + lse_correction * sm_scale
            if smooth_k
            else lse / 1.44269504
        )
    else:
        return o


def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention implementation using Hopper TMA and WGMMA instructions.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        Default: "fp32+fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling.
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in [
        "per_warp",
        "per_thread",
    ], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = (
                    torch.matmul(
                        q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)
                    )
                    .squeeze(-1)
                    .to(torch.float32)
                )
            else:
                lse_correction = (
                    torch.matmul(q, km_broadcast.transpose(2, 3))
                    .squeeze(-1)
                    .to(torch.float32)
                )
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128
        )
    elif qk_quant_gran == "per_thread":
        if per_thread_int8_triton is None:
            raise RuntimeError("Triton per_thread quantization is not available")
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
            q,
            k,
            km,
            tensor_layout=tensor_layout,
            BLKQ=64,
            WARPQ=16,
            BLKK=128,
            WARPK=128,
        )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    # pad v to multiple of 128
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        v.size(0),
                        v.size(1),
                        v_pad_len,
                        v.size(3),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                ],
                dim=2,
            )
        else:
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        v.size(0),
                        v_pad_len,
                        v.size(2),
                        v.size(3),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                ],
                dim=1,
            )

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    if pv_accum_dtype == "fp32":
        raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
    elif pv_accum_dtype == "fp32+fp32":
        lse = torch.ops.tf_kernel.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            o,
            q_scale,
            k_scale,
            v_scale,
            _tensor_layout,
            _is_caual,
            _qk_quant_gran,
            sm_scale,
            _return_lse,
        )

    o = o[..., :head_dim_og]

    if return_lse:
        return o, (
            lse / 1.44269504 + lse_correction * sm_scale
            if smooth_k
            else lse / 1.44269504
        )
    else:
        return o
