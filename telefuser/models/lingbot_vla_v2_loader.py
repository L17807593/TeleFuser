"""Native utility, alignment, and checkpoint support for LingBot-VLA v2.

Adapted from the Apache-2.0 licensed LingBot-VLA v2 implementation.
"""

import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from packaging.version import Version
# from xformers.ops import memory_efficient_attention


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers greater than each input element
        that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible, otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=device
    )
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.rand((bsize,), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def prefix_query_segments(
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
    """Return prefix segment order after the image block.

    Task-specific query tokens are always placed after language tokens. Current
    task queries precede future task queries; future-depth remains the last
    query segment so the existing suffix-to-future-depth blocking can keep using
    the tail span.
    """
    segments = ["language"]
    if not use_depth_align:
        return tuple(segments)

    segments.append("current_depth")
    if use_future_video:
        if use_future_video_cls:
            segments.append("future_video_cls")
        if use_future_video_patch and not future_video_share_future_depth_query:
            segments.append("future_video")
    if use_future_depth:
        segments.append("future_depth")
    return tuple(segments)


def prefix_query_token_spans(
    prefix_len,
    num_task_tokens,
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
    """Return [start, end) spans for non-language task query segments."""
    counts = {
        "current_depth": num_task_tokens,
        "future_video_cls": 1,
        "future_video": num_task_tokens,
        "future_depth": num_task_tokens,
    }
    ordered = prefix_query_segments(
        use_depth_align=use_depth_align,
        use_future_depth=use_future_depth,
        use_future_video=use_future_video,
        use_future_video_cls=use_future_video_cls,
        use_future_video_patch=use_future_video_patch,
        future_video_share_future_depth_query=future_video_share_future_depth_query,
    )
    query_segments = [name for name in ordered if name != "language"]
    cursor = prefix_len - sum(counts[name] for name in query_segments)
    spans = {}
    for name in query_segments:
        count = counts[name]
        spans[name] = (cursor, cursor + count)
        cursor += count
    return spans


def fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch):
    """Return [start, end) of a tail query block inside the prefix.

    This legacy helper is still used for future-depth tail blocking in V2.
    New prefix layout code should prefer prefix_query_token_spans(), which also
    handles current-depth and separate future-video spans.
    """
    fv_len = (1 if use_cls else 0) + (num_task_tokens if use_patch else 0)
    return prefix_len - fv_len, prefix_len

def block_suffix_to_fv_(att_2d_masks, suffix_row_start, prefix_len,
                        num_task_tokens, use_cls=False, use_patch=True, drop_mask=None):
    """In-place mask out the suffix-to-future-video attention edge.

    `make_att_2d_masks`' cumsum scheme cannot express "a query cannot see a
    segment that precedes it", so we zero the rectangular [suffix rows, FV cols]
    block on the already-built 2D mask instead of touching mask_ar.

    att_2d_masks: bool[B, Q, K], True == visible. `suffix_row_start` is the first
    query row belonging to the suffix: prefix_len in the square training mask,
    0 in the suffix-only inference mask. Leaves FV -> img/lang rows untouched so
    the distillation query still reads the current observation.

    `drop_mask`: optional bool[B], True where this sample's suffix must NOT see
    FV. None == block every sample (hard mask). Used for per-sample stochastic
    masking (FV-attention dropout): keep = visible iff not dropped, applied via
    broadcast multiply so it stays a static graph under torch.compile.
    """
    fv_start, fv_end = fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch)
    if fv_end <= fv_start:
        return att_2d_masks
    if drop_mask is None:
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = False
    else:
        # keep[b] = True where the sample is NOT dropped -> AND keeps those rows
        # visible and zeros the dropped ones, with no data-dependent indexing.
        keep = (~drop_mask).view(-1, 1, 1)
        block = att_2d_masks[:, suffix_row_start:, fv_start:fv_end]
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = block & keep
    return att_2d_masks

def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def our_eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Performs eager attention, optimized with torch.einsum.

    Args:
        query_states: Query tensor of shape [batch_size, seq_len, num_attention_heads, head_dim].
        key_states: Key tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        value_states: Value tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        attention_mask: Attention mask tensor, typically [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len, seq_len].

    Returns:
        Output tensor of shape [batch_size, seq_len, num_attention_heads * head_dim].
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads

    key_states = einops.repeat(
        key_states, "b l h d -> b l (h g) d", g=num_key_value_groups
    )
    value_states = einops.repeat(
        value_states, "b l h d -> b l (h g) d", g=num_key_value_groups
    )

    query_states_permuted = torch.einsum("blhd->bhld", query_states)
    key_states_permuted = torch.einsum("blhd->bhld", key_states)

    att_weights = torch.einsum(
        "bhqd,bhkd->bhqk", query_states_permuted, key_states_permuted
    )
    att_weights *= head_dim**-0.5

    big_neg = -2.3819763e38
    masked_att_weights = torch.where(
        attention_mask[:, None, :, :], att_weights, big_neg
    )

    probs = nn.functional.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    value_states_permuted = torch.einsum("blhd->bhld", value_states)  # [B, H, L_v, D]
    att_output = torch.einsum(
        "bhqk,bhkv->bhqv", probs, value_states_permuted
    )  # [B, H, L_q, D]
    att_output = torch.einsum("bhld->blhd", att_output)  # [B, L, H, D]
    att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)

    return att_output


# @torch.jit.script
def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    original_dtype = x.dtype # bf16
    d = x.shape[-1]
    d_half = d // 2
    device = x.device

    # Cast input to compute_dtype for all internal operations
    x_casted = x.to(dtype)
    positions_casted = positions.to(dtype)

    freq_exponents = (2.0 / d) * torch.arange(d_half, dtype=dtype, device=device)
    timescale = max_wavelength**freq_exponents
    radians = torch.einsum("bl,h->blh", positions_casted, 1.0 / timescale) # fp32 -> bf16

    radians = radians[..., None, :]  # [B, L, 1, D_half]

    sin = torch.sin(radians) # bf16
    cos = torch.cos(radians) # bf16

    x1, x2 = x_casted.split(d_half, dim=-1) # fp32

    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1) # fp32

    return res.to(original_dtype) # bf16



# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F  # noqa: N812
from packaging.version import Version
import einops

FLEX_SPARSE_BLOCK_SIZE = 128
FLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}

if Version(torch.__version__) > Version("2.5.0"):
    # Ffex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )

# @torch.compile(dynamic=False)
def flex_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling=None,
):
    """
    This is defined out of classes to make compile happy.
    """
    batch_size, seq_len, num_att_heads, head_dim = query_states.shape
    original_dtype = query_states.dtype
    num_key_value_heads = key_states.shape[2]
    # num_key_value_groups = num_att_heads // num_key_value_heads # 16 // 2 = 8

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, None, :, : key_states.shape[2]]

        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            return precomputed_mask[b][h][q_idx][kv_idx]

        return mask_mod

    b_mask, h_mask, q_len, kv_len = causal_mask.shape  # The shape of your mask
    # ipdb.set_trace()
    block_size = FLEX_SPARSE_BLOCK_SIZE
    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    # *CRITICAL* we do need to expand here, else we get a CUDA index error

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len

    if pad_q > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)  # [B, H, q_len_rounded, D]
    if pad_k > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)
    padded_causal_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)
    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)

    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )

    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )

    #  mask is applied inside the kernel, ideally more efficiently than score_mod.
    attn_output, attention_weights = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,  # because we shaped query/key states for GQA
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
        kernel_options=FLEX_KERNEL_OPTIONS,
    )
    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_LEN, H, head_dim]
    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],  # merges [H, head_dim]
    )
    return attn_output


@torch.compiler.disable
def build_block_mask(
    attention_mask_3d: torch.Tensor,
    num_heads: int,
    q_len: int,
    kv_len: int,
    block_size: int = FLEX_SPARSE_BLOCK_SIZE,
):
    """
    Build a reusable BlockMask from a 3D attention mask [B, Q, KV].
    This allocates the dense 4D mask once; the returned BlockMask can be reused across layers.
    """
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
    )

    causal_mask = attention_mask_3d[:, None, :, :].expand(-1, num_heads, -1, -1).contiguous()
    b_mask, h_mask = causal_mask.shape[0], causal_mask.shape[1]

    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len
    padded_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor):
        def mask_mod(b, h, q_idx, kv_idx):
            return precomputed_mask[b][h][q_idx][kv_idx]
        return mask_mod

    mask_4d = create_mask(
        mod_fn=precomputed_mask_factory(padded_mask),
        B=b_mask, H=h_mask,
        Q_LEN=q_len_rounded, KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )

    block_mask = create_block_mask(
        mask_mod=precomputed_mask_factory(mask_4d),
        B=b_mask, H=h_mask,
        Q_LEN=q_len_rounded, KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )
    return block_mask


def flex_attention_with_block_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_mask,
    seq_len: int,
    scaling=None,
):
    """
    Run flex_attention with a pre-built BlockMask (no create_mask allocation per call).
    """
    batch_size = query_states.shape[0]
    num_att_heads = query_states.shape[2]
    head_dim = query_states.shape[3]
    original_dtype = query_states.dtype

    query_states = query_states.transpose(1, 2).to(torch.float32)
    key_states = key_states.transpose(1, 2).to(torch.float32)
    value_states = value_states.transpose(1, 2).to(torch.float32)

    q_len_rounded = block_mask.shape[-2] if hasattr(block_mask, 'shape') else query_states.shape[2]
    kv_len_rounded = block_mask.shape[-1] if hasattr(block_mask, 'shape') else key_states.shape[2]

    pad_q = q_len_rounded - query_states.shape[2]
    pad_k = kv_len_rounded - key_states.shape[2]

    if pad_q > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)
    if pad_k > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)

    attn_output, _ = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
        kernel_options=FLEX_KERNEL_OPTIONS,
    )
    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, -1, attn_output.shape[2] * attn_output.shape[3])
    return attn_output



# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class AttentionPool2d(nn.Module):

    def __init__(self, seq_len: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(seq_len + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_all_tokens=False):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x.permute(1, 0, 2)  # (N(HW)C) => (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(query=x,
                                              key=x,
                                              value=x,
                                              embed_dim_to_check=x.shape[-1],
                                              num_heads=self.num_heads,
                                              q_proj_weight=self.q_proj.weight,
                                              k_proj_weight=self.k_proj.weight,
                                              v_proj_weight=self.v_proj.weight,
                                              in_proj_weight=None,
                                              in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                                              bias_k=None,
                                              bias_v=None,
                                              add_zero_attn=False,
                                              dropout_p=0,
                                              out_proj_weight=self.c_proj.weight,
                                              out_proj_bias=self.c_proj.bias,
                                              use_separate_proj_weight=True,
                                              training=self.training,
                                              need_weights=False)
        if return_all_tokens:
            return x
        else:
            return x[0]


class Resampler(nn.Module):

    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(1, num_queries, dim_in) / dim_mid ** 0.5)

        self.proj_in = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        queries = self.queries.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries

class TaskTokenResampler(nn.Module):

    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                    FeedForward(dim=dim_mid, mult=ff_mult),
                ]))

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class ResamplerXL(nn.Module):

    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output1_dim=768,
        output2_dim=1280,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        # self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(dim)

        self.in_dim = dim
        self.out_dim = output1_dim + output2_dim

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

        self.unet_proj_1 = nn.Linear(self.in_dim, output1_dim)
        self.unet_proj_2 = nn.Linear(self.in_dim, output2_dim)
        self.unet_attnpool = AttentionPool2d(num_queries, self.in_dim, heads, output2_dim)

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        hidden_embeds = self.norm_out(latents)

        encoder_hidden_1 = self.unet_proj_1(hidden_embeds)  # [bs, 256, 768]
        encoder_hidden_2 = self.unet_proj_2(hidden_embeds)  # [bs, 256, 1280]
        prompt_embeds = torch.cat([encoder_hidden_1, encoder_hidden_2], dim=-1)  # [bs, 256, 2048]
        pooled_prompt_embeds = self.unet_attnpool(hidden_embeds)  # [bs, 1280]

        return prompt_embeds, pooled_prompt_embeds


class ResamplerXLV2(nn.Module):

    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output1_dim=768,
        output2_dim=1280,
        ff_mult=4,
        normalize=True
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.normalize = normalize
        self.proj_in = nn.Linear(embedding_dim, dim)

        # self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(dim)

        self.in_dim = dim
        self.out_dim = output1_dim + output2_dim

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

        self.unet_proj_1 = nn.Linear(self.in_dim, output1_dim)
        self.unet_proj_2 = nn.Linear(self.in_dim, output2_dim)
        self.unet_attnpool = AttentionPool2d(num_queries, self.in_dim, heads, output2_dim)

    def forward(self, x,pooled_text_embeds=None):

        latents = self.latents.repeat(x.size(0), 1, 1)

        if self.normalize:
            x = F.normalize(x)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        hidden_embeds = self.norm_out(latents)

        encoder_hidden_1 = self.unet_proj_1(hidden_embeds)  # [bs, 256, 768]
        encoder_hidden_2 = self.unet_proj_2(hidden_embeds)  # [bs, 256, 1280]
        prompt_embeds = torch.cat([encoder_hidden_1, encoder_hidden_2], dim=-1)  # [bs, 256, 2048]
        pooled_prompt_embeds = self.unet_attnpool(hidden_embeds)  # [bs, 1280]

        return prompt_embeds, pooled_prompt_embeds

class ResamplerXLIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x, pooled_text_embeds=None):
        return x, pooled_text_embeds


if __name__ == '__main__':
    image_proj_model = Resampler(dim=1024,
                                 depth=4,
                                 dim_head=64,
                                 heads=12,
                                 num_queries=1024,
                                 embedding_dim=1024,
                                 output_dim=1024,
                                 ff_mult=4)
    numel = 0
    for name, param in image_proj_model.named_parameters():
        numel += param.numel()

    print(f'Total params: {numel}')



import torch.nn as nn

def build_mlp(in_hidden_size, hidden_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)

def build_expand_mlp(in_hidden_size, hidden_size, out_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, out_size))
    return nn.Sequential(*modules)

class DepthHead(nn.Module):
    def __init__(
        self, 
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super(DepthHead, self).__init__()
        
        self.projector = Resampler(
                dim_in=llm_hidden_size,
                dim_mid=llm_hidden_size,
                dim_head=proj_config["dim_head"],
                dim_out=proj_config["dim_out"],
                num_layers=proj_config["num_layers"],
                num_heads=proj_config["num_heads"],
                num_queries=proj_config["num_backbone_tokens"],
                ff_mult=proj_config["ff_mult"],
            )

    def forward(self, llm_feats):
        queries = self.projector(llm_feats)
        return  queries

class TaskTokenDepthHead(nn.Module):
    def __init__(
        self, 
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super(TaskTokenDepthHead, self).__init__()

        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats, queries):
        queries = self.projector(llm_feats,  queries)
        return  queries



# Copyright 2025 Ant Group Co., Ltd. All Rights Reserved.
# Developer: xiancun
# Project锛?Lumos VIdeo Generation Foundation Model
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triton-optimized MoE auxiliary loss functions.

Provides numerically equivalent replacements for the functions in loss.py,
with two key optimizations:
  1. Eliminate Python for-loops via vectorized segment-wise operations.
  2. Fuse topK + counting into Triton kernels to avoid huge intermediate tensors.

Usage:
    from telefuser.models.lingbot_vla_v2_loader import (
        triton_load_balancing_loss_func,
        triton_sequence_wise_balance_loss,
    )
    # Drop-in replacement 鈥?same signature and return type as loss.py
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Section 1: Triton availability check + kernel definitions
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    # 鈹€鈹€ Kernel 1: per-segment topK counting (for sequence_wise_balance_loss) 鈹€鈹€
    @triton.jit
    def _topk_segment_count_kernel(
        logits_ptr,         # [N_total, E]
        seg_starts_ptr,     # [S_total]
        seg_lengths_ptr,    # [S_total]
        f_out_ptr,          # [S_total, E]
        stride_logits_n,    # stride of logits along token dim
        E: tl.constexpr,    # actual num_experts
        K: tl.constexpr,    # top_k
        BLOCK_E: tl.constexpr,  # next power of 2 >= E
    ):
        """Each program computes f_i (expert counts) for one segment.

        Iterates over tokens in the segment, performs K rounds of argmax to
        find top-K experts, and accumulates per-expert hit counts.
        No gradient needed 鈥?f_i is always detached in the loss.
        """
        seg_id = tl.program_id(0)
        seg_start = tl.load(seg_starts_ptr + seg_id)
        seg_len = tl.load(seg_lengths_ptr + seg_id)

        expert_offs = tl.arange(0, BLOCK_E)  # [BLOCK_E]
        mask_e = expert_offs < E
        f_acc = tl.zeros((BLOCK_E,), dtype=tl.float32)

        for t in range(0, seg_len):
            row_ptr = logits_ptr + (seg_start + t) * stride_logits_n
            logits_row = tl.load(row_ptr + expert_offs, mask=mask_e, other=float('-inf'))

            # K rounds of argmax to find top-K indices
            row_copy = logits_row
            for _k in range(K):
                max_val = tl.max(row_copy, axis=0)
                is_max = (row_copy == max_val)
                # Distribute count evenly among ties (rare with float32)
                n_ties = tl.sum(is_max.to(tl.float32), axis=0)
                f_acc += tl.where(is_max, 1.0 / n_ties, 0.0)
                row_copy = tl.where(is_max, float('-inf'), row_copy)

        # Write f_count (unnormalized) 鈥?caller normalizes by (E / K) / seg_len
        out_ptr = f_out_ptr + seg_id * E
        tl.store(out_ptr + expert_offs, f_acc, mask=mask_e)

    # 鈹€鈹€ Kernel 2: blocked topK counting (for load_balancing_loss_func) 鈹€鈹€
    @triton.jit
    def _topk_count_with_mask_kernel(
        routing_weights_ptr,  # [N, E] 鈥?softmax probabilities
        mask_ptr,             # [N] 鈥?1.0 for valid, 0.0 for padding
        partial_f_ptr,        # [num_blocks, E] 鈥?partial expert counts
        partial_p_ptr,        # [num_blocks, E] 鈥?partial masked prob sums
        N,
        stride_rw_n,          # stride along token dim
        has_mask: tl.constexpr,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Each program accumulates topK counts + masked probs for a token block.

        Two-phase reduction: writes partial [E] results per block;
        caller sums across blocks in PyTorch.
        """
        pid = tl.program_id(0)
        n_start = pid * BLOCK_N

        expert_offs = tl.arange(0, BLOCK_E)
        mask_e = expert_offs < E
        f_local = tl.zeros((BLOCK_E,), dtype=tl.float32)
        p_local = tl.zeros((BLOCK_E,), dtype=tl.float32)

        for t_offset in range(BLOCK_N):
            t = n_start + t_offset
            # Guard: skip if t >= N (handles last block)
            if t < N:
                row_ptr = routing_weights_ptr + t * stride_rw_n
                rw_row = tl.load(row_ptr + expert_offs, mask=mask_e, other=0.0)

                if has_mask:
                    m = tl.load(mask_ptr + t)
                else:
                    m = 1.0

                # Accumulate masked probs
                p_local += rw_row * m

                # TopK counting
                row_copy = rw_row
                for _k in range(K):
                    max_val = tl.max(row_copy, axis=0)
                    is_max = (row_copy == max_val)
                    n_ties = tl.sum(is_max.to(tl.float32), axis=0)
                    f_local += tl.where(is_max, m / n_ties, 0.0)
                    row_copy = tl.where(is_max, float('-inf'), row_copy)

        # Write partial results (only E valid elements)
        f_ptr = partial_f_ptr + pid * E
        p_ptr = partial_p_ptr + pid * E
        tl.store(f_ptr + expert_offs, f_local, mask=mask_e)
        tl.store(p_ptr + expert_offs, p_local, mask=mask_e)

except ImportError:
    pass


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Section 2: Vectorized PyTorch fallback (no Triton needed)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def _build_segment_info(
    seq_lengths_per_layer: List[int],
    num_layers: int,
    N_per_layer: int,
    device: torch.device,
):
    """Build segment IDs and metadata for all layers combined.

    Returns:
        segment_ids:    [num_layers * N_valid_per_layer] int64
        seg_starts:     [total_segments] int64
        seg_lengths:    [total_segments] int64
        total_segments: int
    """
    S = len(seq_lengths_per_layer)
    total_segments = num_layers * S
    seg_lengths_t = torch.tensor(seq_lengths_per_layer, dtype=torch.int64, device=device)

    # Repeat for all layers
    all_seg_lengths = seg_lengths_t.repeat(num_layers)  # [total_segments]

    # Per-layer segment starts: cumsum of seq_lengths
    per_layer_starts = torch.cumsum(seg_lengths_t, 0) - seg_lengths_t  # [S]
    # Layer offsets in the concatenated tensor
    layer_offsets = torch.arange(num_layers, device=device, dtype=torch.int64) * N_per_layer
    # [L, S] 鈫?[L*S]
    all_seg_starts = (per_layer_starts.unsqueeze(0) + layer_offsets.unsqueeze(1)).reshape(-1)

    # Segment IDs: [num_layers * N_valid]
    segment_ids = torch.repeat_interleave(
        torch.arange(total_segments, device=device), all_seg_lengths,
    )

    return segment_ids, all_seg_starts, all_seg_lengths, total_segments


def _vectorized_segment_f_i(
    logits: torch.Tensor,         # [N_total, E]
    seg_starts: torch.Tensor,     # [S_total]
    seg_lengths: torch.Tensor,    # [S_total]
    top_k: int,
) -> torch.Tensor:
    """Compute f_i per segment without Triton, using vectorized PyTorch ops.

    Returns: f_i [S_total, E]
    """
    N, E = logits.shape
    S_total = seg_starts.shape[0]

    # TopK over all tokens at once
    _, topk_idx = torch.topk(logits, k=top_k, dim=-1)  # [N, K]

    # Build one-hot mask efficiently: scatter into [N, E]
    mask = torch.zeros(N, E, device=logits.device, dtype=torch.float32)
    mask.scatter_(1, topk_idx, 1.0)

    # Segment-wise sum using scatter_add
    segment_ids = torch.repeat_interleave(
        torch.arange(S_total, device=logits.device), seg_lengths,
    )
    seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)  # [N, E]

    f_sum = torch.zeros(S_total, E, device=logits.device, dtype=torch.float32)
    f_sum.scatter_add_(0, seg_ids_exp, mask)

    # Normalize: f_i = (E / K) * f_sum / T_s
    inv_lens = (float(E) / top_k) / seg_lengths.unsqueeze(1).float().clamp(min=1)
    f_i = f_sum * inv_lens

    return f_i


def _vectorized_topk_count(
    routing_weights: torch.Tensor,  # [N, E]
    top_k: int,
    flat_mask: Optional[torch.Tensor] = None,  # [N] float
) -> torch.Tensor:
    """Count per-expert topK selections, optionally masked. Returns [E]."""
    N, E = routing_weights.shape
    _, topk_idx = torch.topk(routing_weights, k=top_k, dim=-1)  # [N, K]

    tokens_per_expert = torch.zeros(E, device=routing_weights.device, dtype=torch.float32)
    weight = flat_mask if flat_mask is not None else torch.ones(N, device=routing_weights.device, dtype=torch.float32)

    # K rounds of scatter_add 鈥?K is small (typically 2-8), no Python overhead concern
    for k in range(top_k):
        tokens_per_expert.scatter_add_(0, topk_idx[:, k], weight)

    return tokens_per_expert


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Section 3: Triton-accelerated wrappers
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def _triton_segment_f_i(
    logits: torch.Tensor,         # [N_total, E]
    seg_starts: torch.Tensor,     # [S_total]
    seg_lengths: torch.Tensor,    # [S_total]
    top_k: int,
) -> torch.Tensor:
    """Compute f_i per segment using the Triton kernel. Returns [S_total, E]."""
    N, E = logits.shape
    S_total = seg_starts.shape[0]
    BLOCK_E = _next_power_of_2(E)

    f_counts = torch.zeros(S_total, E, device=logits.device, dtype=torch.float32)

    _topk_segment_count_kernel[(S_total,)](
        logits,
        seg_starts,
        seg_lengths,
        f_counts,
        logits.stride(0),
        E=E,
        K=top_k,
        BLOCK_E=BLOCK_E,
    )

    # Normalize: f_i = (E / K) * counts / T_s
    inv_lens = (float(E) / top_k) / seg_lengths.unsqueeze(1).float().clamp(min=1)
    return f_counts * inv_lens


def _triton_topk_count(
    routing_weights: torch.Tensor,  # [N, E]
    top_k: int,
    flat_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute tokens_per_expert [E] using the Triton kernel."""
    N, E = routing_weights.shape
    BLOCK_E = _next_power_of_2(E)
    BLOCK_N = 256
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    partial_f = torch.zeros(num_blocks, E, device=routing_weights.device, dtype=torch.float32)
    partial_p = torch.zeros(num_blocks, E, device=routing_weights.device, dtype=torch.float32)

    has_mask = flat_mask is not None
    _topk_count_with_mask_kernel[(num_blocks,)](
        routing_weights,
        flat_mask if has_mask else routing_weights,  # dummy ptr when no mask
        partial_f,
        partial_p,
        N,
        routing_weights.stride(0),
        has_mask=has_mask,
        E=E,
        K=top_k,
        BLOCK_E=BLOCK_E,
        BLOCK_N=BLOCK_N,
    )

    # Phase 2: reduce across blocks
    tokens_per_expert = partial_f.sum(dim=0)  # [E]

    if flat_mask is not None:
        n_valid = flat_mask.sum().clamp(min=1)
    else:
        n_valid = float(N)
    tokens_per_expert = tokens_per_expert / n_valid

    return tokens_per_expert


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Section 4: Main API 鈥?drop-in replacements for loss.py
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def triton_sequence_wise_balance_loss(
    router_logits_list: tuple,
    top_k: int,
    seq_lengths: Optional[List[int]] = None,
    padding_len: int = 0,
    score_func: str = "softmax",
) -> List[torch.Tensor]:
    """Triton-optimized DeepSeek-V3 sequence-wise balance loss.

    Numerically equivalent to sequence_wise_balance_loss() in loss.py,
    but eliminates all Python for-loops by:
      - Processing all layers simultaneously via concatenation
      - Using segment-wise parallel reduction (scatter_add) instead of per-sequence loops
      - Fusing topK + counting in a Triton kernel (with PyTorch vectorized fallback)

    Args / Returns: same as sequence_wise_balance_loss in loss.py.
    """
    if router_logits_list is None or not isinstance(router_logits_list, (tuple, list)):
        return []

    valid_logits = [rl for rl in router_logits_list if rl is not None]
    if len(valid_logits) == 0:
        return []

    num_layers = len(valid_logits)
    device = valid_logits[0].device
    E = valid_logits[0].shape[1]

    # 鈹€鈹€ Step 1: Concatenate all layers, remove padding 鈹€鈹€
    all_logits_list = []
    N_per_layer = None
    for logits in valid_logits:
        logits_f32 = logits.to(dtype=torch.float32)
        N = logits_f32.shape[0]
        if padding_len > 0:
            logits_f32 = logits_f32[:N - padding_len]
        all_logits_list.append(logits_f32)
        if N_per_layer is None:
            N_per_layer = logits_f32.shape[0]

    # Check if all layers have the same valid length (common case)
    same_length = all(l.shape[0] == N_per_layer for l in all_logits_list)

    if not same_length:
        # Rare: different MoE layers have different token counts
        return _fallback_per_layer(valid_logits, top_k, seq_lengths, padding_len, score_func)

    if seq_lengths is None or len(seq_lengths) == 0:
        seq_lengths_effective = [N_per_layer]
    else:
        seq_lengths_effective = seq_lengths

    S = len(seq_lengths_effective)
    all_logits = torch.cat(all_logits_list, dim=0)  # [L * N_valid, E]

    # 鈹€鈹€ Step 2: Build segment metadata 鈹€鈹€
    segment_ids, seg_starts, seg_lengths_t, total_segments = _build_segment_info(
        seq_lengths_effective, num_layers, N_per_layer, device
    )

    # 鈹€鈹€ Step 3: P_i via PyTorch (gradient path) 鈹€鈹€
    if score_func == "sigmoid":
        all_scores = all_logits.sigmoid()
        all_probs = all_scores / all_scores.sum(dim=-1, keepdim=True)
    else:
        all_probs = F.softmax(all_logits, dim=-1)  # [L * N_valid, E]
    seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)  # [L * N_valid, E]

    P_sum = torch.zeros(total_segments, E, device=device, dtype=torch.float32)
    P_sum.scatter_add_(0, seg_ids_exp, all_probs)
    P_i = P_sum / seg_lengths_t.unsqueeze(1).float().clamp(min=1)  # [total_segments, E]

    # 鈹€鈹€ Step 4: f_i (no gradient needed) 鈹€鈹€
    with torch.no_grad():
        if _HAS_TRITON and all_logits.is_cuda:
            f_i = _triton_segment_f_i(all_logits, seg_starts, seg_lengths_t, top_k)
        else:
            f_i = _vectorized_segment_f_i(all_logits, seg_starts, seg_lengths_t, top_k)

    # 鈹€鈹€ Step 5: Per-segment loss 鈫?per-layer mean 鈹€鈹€
    loss_per_seg = (f_i * P_i).sum(dim=-1)  # [total_segments]
    loss_per_seg = loss_per_seg.reshape(num_layers, S)
    layer_losses = loss_per_seg.mean(dim=1)  # [L]

    return list(layer_losses.unbind(0))


def triton_load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """Triton-optimized Switch Transformer load balancing loss.

    Numerically equivalent to load_balancing_loss_func() in loss.py,
    but avoids the huge [N, K, E] one_hot intermediate tensor by
    directly counting expert assignments via Triton or scatter_add.

    Memory reduction: O(N*K*E) 鈫?O(N*E + num_blocks*E)

    Args / Returns: same as load_balancing_loss_func in loss.py.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    gate_logits = tuple(g for g in gate_logits if g is not None)
    if len(gate_logits) == 0:
        return 0

    compute_device = gate_logits[0].device
    concatenated = torch.cat(
        [g.to(device=compute_device, dtype=torch.float32) for g in gate_logits],
        dim=0,
    )  # [L*N, E]

    # 鈹€鈹€ Step 1: softmax (gradient path) 鈹€鈹€
    routing_weights = F.softmax(concatenated, dim=-1)  # [L*N, E]

    # 鈹€鈹€ Step 2: Build flat mask 鈹€鈹€
    N_total = routing_weights.shape[0]
    if attention_mask is not None:
        batch_size, seq_len = attention_mask.shape
        num_layers = N_total // (batch_size * seq_len)
        flat_mask = (
            attention_mask
            .unsqueeze(0)
            .expand(num_layers, -1, -1)
            .reshape(-1)
            .to(device=compute_device, dtype=torch.float32)
        )
    else:
        flat_mask = None

    # 鈹€鈹€ Step 3: tokens_per_expert (no gradient) 鈹€鈹€
    with torch.no_grad():
        if _HAS_TRITON and routing_weights.is_cuda:
            tokens_per_expert = _triton_topk_count(routing_weights, top_k, flat_mask)
        else:
            tokens_per_expert = _vectorized_topk_count(routing_weights, top_k, flat_mask)
            if flat_mask is not None:
                tokens_per_expert = tokens_per_expert / flat_mask.sum().clamp(min=1)
            else:
                tokens_per_expert = tokens_per_expert / float(N_total)

    # 鈹€鈹€ Step 4: router_prob_per_expert (gradient path) 鈹€鈹€
    if flat_mask is not None:
        n_valid = flat_mask.sum().clamp(min=1)
        router_prob_per_expert = (routing_weights * flat_mask.unsqueeze(1)).sum(0) / n_valid
    else:
        router_prob_per_expert = routing_weights.mean(dim=0)

    # 鈹€鈹€ Step 5: loss 鈹€鈹€
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
    return overall_loss * num_experts


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Section 5: Fallback for edge cases
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def _fallback_per_layer(
    valid_logits: List[torch.Tensor],
    top_k: int,
    seq_lengths: Optional[List[int]],
    padding_len: int,
    score_func: str = "softmax",
) -> List[torch.Tensor]:
    """Fallback when layers have different valid token counts.

    Still vectorized within each layer (no per-sequence for-loop).
    """
    layer_loss_list = []
    for logits in valid_logits:
        logits = logits.to(dtype=torch.float32)
        N, E = logits.shape
        if padding_len > 0:
            logits = logits[:N - padding_len]
        if logits.shape[0] == 0:
            continue

        if seq_lengths is not None and len(seq_lengths) > 0:
            S = len(seq_lengths)
            device = logits.device
            seg_lengths_t = torch.tensor(seq_lengths, dtype=torch.int64, device=device)
            seg_starts = torch.cumsum(seg_lengths_t, 0) - seg_lengths_t

            # P_i (gradient path)
            if score_func == "sigmoid":
                scores = logits.sigmoid()
                probs = scores / scores.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
            segment_ids = torch.repeat_interleave(torch.arange(S, device=device), seg_lengths_t)
            seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)
            P_sum = torch.zeros(S, E, device=device, dtype=torch.float32)
            P_sum.scatter_add_(0, seg_ids_exp, probs)
            P_i = P_sum / seg_lengths_t.unsqueeze(1).float().clamp(min=1)

            # f_i (no gradient)
            with torch.no_grad():
                if _HAS_TRITON and logits.is_cuda:
                    f_i = _triton_segment_f_i(logits, seg_starts, seg_lengths_t, top_k)
                else:
                    f_i = _vectorized_segment_f_i(logits, seg_starts, seg_lengths_t, top_k)

            loss_per_seq = (f_i * P_i).sum(dim=-1)
            layer_loss_list.append(loss_per_seq.mean())
        else:
            if score_func == "sigmoid":
                scores = logits.sigmoid()
                probs = scores / scores.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
            P_i = probs.mean(dim=0)

            with torch.no_grad():
                _, topk_idx = torch.topk(logits, k=top_k, dim=-1)
                mask = torch.zeros_like(logits)
                mask.scatter_(1, topk_idx, 1.0)
                f_i = (E / top_k) * mask.mean(dim=0)

            layer_loss_list.append(torch.sum(f_i * P_i))

    return layer_loss_list


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Section 6: Numerical alignment test
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?





import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoConfig


class NativeParallelPlan:
    """Compatibility container for the upstream training-only parallel plan."""

    def __init__(self, ep_plan=None):
        self.ep_plan = ep_plan or {}


class LingBotVLAWeightLoader:
    """Minimal native weight-name mapper retained for model compatibility."""

    def get_vlm_submodule(self, model):
        return model.model.qwenvl_with_expert.qwenvl

    def get_expert_vision_submodule(self, model):
        return getattr(model.model.qwenvl_with_expert, "expert_visual", None)

    def map_ckpt_key(self, key, load_vlm_only=False, post_training=False):
        if key.startswith("expert_visual.") and not post_training:
            return "model.qwenvl_with_expert." + key
        if load_vlm_only:
            return "model.qwenvl_with_expert.qwenvl." + key
        return key


OFFICIAL_6B_MODEL_CONFIG: dict[str, Any] = {
    "post_training": True,
    "adanorm_time": True,
    "moe_implementation": "fused",
    "attention_implementation": "eager",
    "precompute_grid_thw": True,
    "vlm_causal": True,
    "use_moe": True,
    "token_moe_layers": list(range(36)),
    "token_num_experts": 32,
    "token_top_k": 4,
    "token_moe_intermediate_size": 512,
    "token_shared_intermediate_size": 704,
    "bias_update_speed": 0.0,
    "sequence_wise_mode": "per_sequence",
    "sequence_wise_loss_coeff": 1e-3,
    "router_z_loss_coeff": 1e-4,
    "router_activation": "sigmoid",
    "routed_scaling_factor": 4.0,
    "use_shared_expert_gate": False,
    "freeze_vision_encoder": False,
    "tokenizer_max_length": 72,
    "loss_type": "L1_fm",
    "action_dim": 55,
    "max_action_dim": 55,
    "max_state_dim": 55,
    "align_params": {
        "mode": "query",
        "num_task_tokens": 8,
        "depth_loss_weight": 0.004,
        "future_depth_loss_weight": 0.004,
        "use_future_video": True,
        "llm": {
            "dim_out": 2560,
            "image_token_size": 8,
            "image_input_size": 224,
        },
        "depth": {
            "model_type": "MoRGBD",
            "num_layers": 1,
            "num_heads": 4,
            "dim_head": 32,
            "ff_mult": 1,
            "num_backbone_tokens": 256,
            "token_size": 16,
            "dim_out": 1024,
            "input_size": 224,
            "use_future_depth": True,
            "block_future_depth_to_action": True,
            "future_depth_head_type": "resampler",
            "detach_future_image_feats": True,
        },
        "video": {
            "attention_mode": "flex_block_causal",
            "input_size": 256,
            "block_suffix_to_future_video": True,
            "share_future_depth_query": True,
            "use_shared_future_task_proj": True,
            "use_current_shared_task_proj": True,
            "num_future_frames": 1,
            "use_warmup_frame": True,
            "effective_fps": 1.0,
            "n_blocks": 1,
            "cls_pool": "last",
            "detach_image_feats": True,
            "num_layers": 1,
            "num_heads": 4,
            "dim_head": 32,
            "ff_mult": 1,
            "num_backbone_tokens": 256,
            "dim_out": 1024,
            "future_video_loss_weight": 0.004,
            "use_smooth_l1_loss": False,
            "use_mse_loss": True,
            "mse_loss_weight": 1.0,
            "use_patch_loss": True,
            "use_current_patch_loss": True,
            "use_cosine_loss": False,
            "cosine_loss_weight": 0.2,
            "use_cls_loss": False,
            "cls_loss_type": "mse",
            "cls_loss_weight": 0.2,
        },
    },
}


def resolve_lingbot_vla_v2_checkpoint(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.name != "model.safetensors.index.json":
            raise ValueError(f"Expected model.safetensors.index.json, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"LingBot-VLA v2 model path does not exist: {path}")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded checkpoint index: {index_path}")
    return index_path


def resolve_lingbot_vla_v2_shards(model_path: str | Path) -> list[str]:
    index_path = resolve_lingbot_vla_v2_checkpoint(model_path)
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index without weight_map: {index_path}")

    shard_paths = [index_path.parent / name for name in sorted(set(weight_map.values()))]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LingBot-VLA v2 checkpoint shards: {missing}")
    return [str(path) for path in shard_paths]


def build_official_6b_config(qwen3vl_path: str | Path):
    from telefuser.models.lingbot_vla_v2 import LingbotVLAV2Config

    qwen_path = Path(qwen3vl_path).expanduser().resolve()
    qwen_config = AutoConfig.from_pretrained(str(qwen_path), local_files_only=True)
    if not hasattr(qwen_config, "text_config") or not hasattr(qwen_config, "vision_config"):
        raise ValueError(
            "LingBot-VLA v2 requires the local Qwen3-VL-4B-Instruct architecture/tokenizer "
            f"directory; this is not a complete Qwen3-VL directory: {qwen_path}"
        )

    text_config = qwen_config.text_config
    expected_architecture = {"hidden_size": 2560, "num_hidden_layers": 36}
    mismatches = {
        key: (expected, getattr(text_config, key, None))
        for key, expected in expected_architecture.items()
        if getattr(text_config, key, None) != expected
    }
    if mismatches:
        raise ValueError(
            "LingBot-VLA v2 6B was trained with Qwen3-VL-4B-Instruct; "
            f"the supplied architecture is incompatible: {mismatches}"
        )

    values = deepcopy(OFFICIAL_6B_MODEL_CONFIG)
    values["tokenizer_path"] = str(qwen_path)
    config = LingbotVLAV2Config(**values)
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
    ):
        if hasattr(text_config, key):
            setattr(config, key, getattr(text_config, key))
    config.vision_config = qwen_config.vision_config
    config.tokenizer_path = str(qwen_path)
    config.use_cache = True
    config.attention_implementation = "eager"
    return config


def validate_official_6b_checkpoint(state_dict):
    gate = "model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.gate_proj"
    last_gate = "model.qwenvl_with_expert.qwen_expert.model.layers.35.mlp.experts.gate_proj"
    expected = (32, 512, 768)
    for key in (gate, last_gate):
        if key not in state_dict:
            raise ValueError(f"Missing official LingBot-VLA v2 weight: {key}")
        if tuple(state_dict[key].shape) != expected:
            raise ValueError(
                f"Unexpected shape for {key}: expected {expected}, got {tuple(state_dict[key].shape)}"
            )


class LingBotVlaV2StateDictConverter:
    def __init__(self, qwen3vl_path: str | Path):
        self.qwen3vl_path = Path(qwen3vl_path)

    def from_official(self, state_dict):
        validate_official_6b_checkpoint(state_dict)
        config = build_official_6b_config(self.qwen3vl_path)
        return state_dict, {"config": config, "eval": True}

    def from_diffusers(self, state_dict):
        del state_dict
        raise ValueError("LingBot-VLA v2 does not provide a Diffusers checkpoint")


def load_lingbot_vla_v2(
    module_manager,
    model_path: str | Path,
    qwen3vl_path: str | Path,
    *,
    torch_dtype=torch.bfloat16,
    device=None,
):
    from telefuser.models.lingbot_vla_v2 import LingBotVlaV2Model

    shard_paths = resolve_lingbot_vla_v2_shards(model_path)
    module_manager.load_model(
        shard_paths,
        device=device,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        name="lingbot_vla_v2",
        model_class=LingBotVlaV2Model,
        model_resource="official",
        converter_kwargs={"qwen3vl_path": str(qwen3vl_path)},
        strict=True,
    )
    return module_manager.fetch_module("lingbot_vla_v2")
