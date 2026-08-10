"""Sparse attention patterns for video generation.

Implements radial attention where tokens attend to nearby frames densely
and distant frames sparsely, following a radial decay pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange, repeat

from telefuser.core.config import SparseAttentionConfig
from telefuser.ops.attention.backends import FLASHINFER_AVAILABLE, flashinfer, sageattention
from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from torch import Tensor


def get_cuda_arch_versions() -> list[str]:
    """Get CUDA compute capability for all devices."""
    return [
        f"sm{major}{minor}"
        for i in range(torch.cuda.device_count())
        for major, minor in [torch.cuda.get_device_capability(i)]
    ]


def sparge_mask_convert(mask: Tensor, block_size: int = 128, arch: str = "sm") -> Tensor:
    """Convert mask for sparge attention backend.

    Adapts mask dimensions based on block size and GPU architecture
    (SM90 Hopper requires different layout than earlier GPUs).

    Args:
        mask: Input mask [seq, seq].
        block_size: Block size (64 or 128).
        arch: GPU architecture ("sm" or "sm90").

    Returns:
        Converted mask.
    """
    assert block_size in [64, 128], "Block size must be 64 or 128"
    assert mask.shape[0] == mask.shape[1], "Mask must be square"

    if block_size == 128:
        if arch == "sm90":
            return torch.repeat_interleave(mask, 2, dim=0)
        return torch.repeat_interleave(mask, 2, dim=1)

    # block_size == 64
    num_row, num_col = mask.shape
    if arch == "sm90":
        return torch.max(mask.view(num_row, num_col // 2, 2), dim=2).values
    return torch.max(mask.view(num_row // 2, 2, num_col), dim=1).values


def get_indptr_from_mask(mask: Tensor, query: Tensor) -> Tensor:
    """Convert mask to CSR indptr format for FlashInfer."""
    indptr = torch.zeros(mask.shape[0] + 1, device=query.device, dtype=torch.int32)
    row_counts = mask.sum(dim=1).flatten()
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return indptr


def get_indices_from_mask(mask: Tensor, query: Tensor) -> Tensor:
    """Convert mask to CSR indices format for FlashInfer."""
    nonzero = torch.nonzero(mask)
    return nonzero[:, 1].to(dtype=torch.int32, device=query.device)


def shrink_mask_strict(mask: Tensor, block_size: int = 128) -> Tensor:
    """Shrink fine-grained mask to block-level mask.

    A block is considered "active" if >60% of its high-density columns
    have density >1/3.
    """
    seqlen = mask.shape[0]
    block_num = seqlen // block_size

    mask = mask[: block_num * block_size, : block_num * block_size]
    mask = mask.view(block_num, block_size, block_num, block_size)

    col_densities = mask.sum(dim=1) / block_size
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3

    frac_high = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high > 0.6

    # Always include first and last blocks (attention sinks)
    block_mask[0] = True
    block_mask[-1] = True
    return block_mask


def get_diagonal_split_mask(i: int, j: int, token_per_frame: int, sparse_type: str, query: Tensor) -> Tensor:
    """Compute diagonal split mask for frame pair (i, j)."""
    assert sparse_type in ["radial"], f"Unknown sparse type: {sparse_type}"

    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128
    decay_length = 2 ** token_per_frame.bit_length() / 2**group

    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)

    split_factor = int(threshold / decay_length)
    return (
        torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
        if dist % split_factor == 0
        else torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
    )


def get_window_width(
    i: int,
    j: int,
    token_per_frame: int,
    sparse_type: str,
    num_frame: int,
    decay_factor: float = 1.0,
    block_size: int = 128,
    model_type: str | None = None,
) -> float:
    """Compute attention window width for frame pair (i, j)."""
    assert sparse_type in ["radial"], f"Unknown sparse type: {sparse_type}"
    assert model_type == "wan", f"Unknown model type: {model_type}"

    dist = abs(i - j)
    if dist <= 1:
        return token_per_frame

    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2**group * decay_factor
    return max(decay_length, block_size)


def gen_log_mask_shrinked(
    query: Tensor,
    s: int,
    video_token_num: int,
    num_frame: int,
    block_size: int = 128,
    sparse_type: str = "radial",
    decay_factor: float = 0.5,
    model_type: str | None = None,
) -> Tensor:
    """Generate radial attention mask with memory-efficient processing."""
    final_mask = torch.zeros((s // block_size, s // block_size), device=query.device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_blocks = video_token_num // block_size

    # Text-to-all and all-to-text always enabled
    final_mask[video_blocks:] = True
    final_mask[:, video_blocks:] = True

    col_idx = torch.arange(0, token_per_frame, device=query.device).view(1, -1)
    row_idx = torch.arange(0, token_per_frame, device=query.device).view(-1, 1)

    for i in range(num_frame):
        for j in range(num_frame):
            if j == 0 and model_type == "wan":
                local_mask = torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
            else:
                window = get_window_width(
                    i, j, token_per_frame, sparse_type, num_frame, decay_factor, block_size, model_type
                )
                local_mask = torch.abs(col_idx - row_idx) <= window
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query)
                local_mask = torch.logical_and(local_mask, split_mask)

            row_offset = (i * token_per_frame) % block_size
            col_offset = (j * token_per_frame) % block_size
            pad_h = row_offset + ((token_per_frame - 1) // block_size + 1) * block_size
            pad_w = col_offset + ((token_per_frame - 1) // block_size + 1) * block_size

            padded = torch.zeros((pad_h, pad_w), device=query.device, dtype=torch.bool)
            padded[row_offset : row_offset + token_per_frame, col_offset : col_offset + token_per_frame] = local_mask
            block_mask = shrink_mask_strict(padded, block_size=block_size)

            row_start = (i * token_per_frame) // block_size
            col_start = (j * token_per_frame) // block_size
            final_mask[row_start : row_start + block_mask.shape[0], col_start : col_start + block_mask.shape[1]] = (
                torch.logical_or(
                    final_mask[
                        row_start : row_start + block_mask.shape[0], col_start : col_start + block_mask.shape[1]
                    ],
                    block_mask,
                )
            )

    sparsity = 1 - final_mask.sum() / final_mask.numel()
    logger.info(f"Radial attention mask sparsity: {sparsity:.2%}")
    return final_mask


class MaskMap:
    """Global mask and workspace cache for radial attention.

    Avoids regenerating radial masks and workspace buffers for each attention call.
    Workspace buffer is reused to prevent memory fragmentation from repeated allocations.
    """

    _log_mask: Tensor | None = None

    # Per-device workspace cache to avoid repeated 128MB allocations
    _workspace_cache: dict[str, Tensor] = {}

    # Default workspace size: 128MB
    WORKSPACE_SIZE = 128 * 1024 * 1024

    def __init__(self, video_token_num: int = 25440, num_frame: int = 16):
        self.video_token_num = video_token_num
        self.num_frame = num_frame

    def query_log_mask(
        self,
        query: Tensor,
        sparse_type: str,
        block_size: int = 128,
        decay_factor: float = 0.5,
        model_type: str | None = None,
    ) -> Tensor:
        """Get or generate radial mask."""
        if MaskMap._log_mask is None:
            seq_len = query.shape[1]
            MaskMap._log_mask = torch.ones(
                (seq_len // block_size, seq_len // block_size),
                device=query.device,
                dtype=torch.bool,
            )
            MaskMap._log_mask = gen_log_mask_shrinked(
                query,
                seq_len,
                self.video_token_num,
                self.num_frame,
                block_size=block_size,
                sparse_type=sparse_type,
                decay_factor=decay_factor,
                model_type=model_type,
            )
        return MaskMap._log_mask

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the global mask and workspace cache."""
        cls._log_mask = None
        cls._workspace_cache.clear()

    @classmethod
    def get_workspace(cls, device: torch.device) -> Tensor:
        """Get or create workspace buffer for FlashInfer.

        Workspace buffers are cached per device to avoid repeated 128MB allocations
        which cause memory fragmentation and pressure.

        Args:
            device: Target device for the workspace.

        Returns:
            Cached workspace tensor.
        """
        device_key = f"{device.type}_{device.index}"
        if device_key not in cls._workspace_cache:
            cls._workspace_cache[device_key] = torch.empty(cls.WORKSPACE_SIZE, device=device, dtype=torch.uint8)
        return cls._workspace_cache[device_key]


def _get_sageattn_impl():
    """Get sageattention implementation (tf-kernel prioritized)."""
    if sageattention is None:
        return None
    return getattr(sageattention, "sageattn", None)


def _flashinfer_text_attention(
    q_text: Tensor,
    k_text: Tensor,
    v_text: Tensor,
    pre_defined_mask: Tensor | None = None,
    batch_size: int = 1,
) -> Tensor:
    """Compute text attention with FlashInfer."""
    if not FLASHINFER_AVAILABLE:
        raise RuntimeError("FlashInfer required for text attention")

    kwargs = {"q": q_text, "k": k_text, "v": v_text, "causal": False, "return_lse": False}
    if pre_defined_mask is not None:
        kwargs["custom_mask"] = pre_defined_mask

    output = flashinfer.single_prefill_with_kv_cache(**kwargs)
    return rearrange(output, "(b s) h d -> b s (h d)", b=batch_size)


def radial_attention_flashinfer(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask_map: MaskMap,
    video_mask: Tensor,
    pre_defined_mask: Tensor | None,
    block_size: int,
) -> Tensor:
    """Radial attention with FlashInfer backend."""
    if not FLASHINFER_AVAILABLE:
        raise RuntimeError("FlashInfer required")

    batch_size = query.shape[0]
    num_head = query.shape[2]
    hidden_dim = query.shape[3]

    video_mask = video_mask[: mask_map.video_token_num // block_size, : mask_map.video_token_num // block_size]
    workspace = mask_map.get_workspace(query.device)
    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace, backend="fa2")

    indptr = get_indptr_from_mask(video_mask, query)
    indices = get_indices_from_mask(video_mask, query)

    bsr_wrapper.plan(
        indptr=indptr,
        indices=indices,
        M=mask_map.video_token_num,
        N=mask_map.video_token_num,
        R=block_size,
        C=block_size,
        num_qo_heads=num_head,
        num_kv_heads=num_head,
        head_dim=hidden_dim,
        q_data_type=query.dtype,
        kv_data_type=key.dtype,
        o_data_type=query.dtype,
    )

    q_shd = rearrange(query, "b s h d -> (b s) h d")
    k_shd = rearrange(key, "b s h d -> (b s) h d")
    v_shd = rearrange(value, "b s h d -> (b s) h d")

    if pre_defined_mask is not None:
        # Video-video + video-text attention
        video_o, video_lse = bsr_wrapper.run(
            q_shd[: mask_map.video_token_num, :, :],
            k_shd[: mask_map.video_token_num, :, :],
            v_shd[: mask_map.video_token_num, :, :],
            return_lse=True,
        )
        video_text_o, video_text_lse = flashinfer.single_prefill_with_kv_cache(
            q=q_shd[: mask_map.video_token_num, :, :],
            k=k_shd[mask_map.video_token_num :, :, :],
            v=v_shd[mask_map.video_token_num :, :, :],
            causal=False,
            return_lse=True,
            custom_mask=pre_defined_mask[: mask_map.video_token_num, mask_map.video_token_num :],
        )
        o_video, _ = flashinfer.merge_state(v_a=video_o, s_a=video_lse, v_b=video_text_o, s_b=video_text_lse)

        # Text-to-all attention
        o_text = flashinfer.single_prefill_with_kv_cache(
            q=q_shd[mask_map.video_token_num :, :, :],
            k=k_shd,
            v=v_shd,
            causal=False,
            return_lse=False,
            custom_mask=pre_defined_mask[mask_map.video_token_num :, :],
        )

        output = torch.cat([o_video, o_text], dim=0)
        return rearrange(output, "(b s) h d -> b s (h d)", b=batch_size)

    # Pure video case
    o = bsr_wrapper.run(
        q_shd[: mask_map.video_token_num, :, :],
        k_shd[: mask_map.video_token_num, :, :],
        v_shd[: mask_map.video_token_num, :, :],
    )
    return rearrange(o, "(b s) h d -> b s (h d)", b=batch_size)


def radial_attention_sage(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask_map: MaskMap,
    video_mask: Tensor,
    pre_defined_mask: Tensor | None,
    block_size: int,
) -> Tensor:
    """Radial attention with SageAttention backend."""
    # Check for block sparse implementation availability
    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda
    except ImportError:
        try:
            from sparse_sageattn import sparse_sageattn as block_sparse_sage2_attn_cuda
        except ImportError:
            raise RuntimeError("spas_sage_attn or sparse_sageattn required")

    batch_size = query.shape[0]
    sageattn_impl = _get_sageattn_impl()

    # Dense video attention fallback
    if video_mask is not None and video_mask.all():
        kv_border = pre_defined_mask[0].sum() if pre_defined_mask is not None else key.shape[1]
        if sageattn_impl is None:
            raise RuntimeError("sageattn required for dense attention")

        output_video = sageattn_impl(
            query[:, : mask_map.video_token_num, :, :],
            key[:, :kv_border, :, :],
            value[:, :kv_border, :, :],
            tensor_layout="NHD",
        )

        if pre_defined_mask is not None:
            q_text = rearrange(query[:, mask_map.video_token_num :, :, :], "b s h d -> (b s) h d")
            k_text = rearrange(key[:, : pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
            v_text = rearrange(value[:, : pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
            output_text = _flashinfer_text_attention(q_text, k_text, v_text, batch_size=batch_size)
            return torch.cat([output_video.flatten(2, 3), output_text], dim=1)

        return output_video.flatten(2, 3)

    # Sparse video attention
    arch = get_cuda_arch_versions()[query.device.index]
    converted_mask = repeat(
        sparge_mask_convert(mask=video_mask, block_size=block_size, arch=arch),
        "s t -> b h s t",
        b=batch_size,
        h=query.shape[2],
    ).to(torch.int8)

    q_bhsd = rearrange(query, "b s h d -> b h s d")
    k_bhsd = rearrange(key, "b s h d -> b h s d")
    v_bhsd = rearrange(value, "b s h d -> b h s d")

    # Pure video case
    if pre_defined_mask is None:
        output = block_sparse_sage2_attn_cuda(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            mask_id=converted_mask[:, :, :, : k_bhsd.shape[2] // block_size].contiguous(),
            tensor_layout="HND",
        )
        return rearrange(output, "b h s d -> b s (h d)")

    # Mixed video-text case
    kv_border = (pre_defined_mask[0].sum() + 63) // 64
    converted_mask[:, :, :, kv_border:] = False

    output_video = block_sparse_sage2_attn_cuda(
        q_bhsd[:, :, : mask_map.video_token_num, :],
        k_bhsd,
        v_bhsd,
        mask_id=converted_mask[:, :, : mask_map.video_token_num // block_size, :].contiguous(),
        tensor_layout="HND",
    )
    output_video = rearrange(output_video, "b h s d -> b s (h d)")

    # Text attention
    q_text = rearrange(query[:, mask_map.video_token_num :, :, :], "b s h d -> (b s) h d")
    k_text = rearrange(key[:, : pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
    v_text = rearrange(value[:, : pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
    output_text = flashinfer.single_prefill_with_kv_cache(
        q=q_text,
        k=k_text,
        v=v_text,
        causal=False,
        return_lse=False,
        custom_mask=pre_defined_mask[mask_map.video_token_num :, :],
    )
    output_text = rearrange(output_text, "(b s) h d -> b s (h d)", b=batch_size)

    return torch.cat([output_video, output_text], dim=1)


def radial_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask_map: MaskMap | None = None,
    sparsity_type: str = "radial",
    block_size: int = 128,
    decay_factor: float = 1.0,
    model_type: str | None = None,
    pre_defined_mask: Tensor | None = None,
    use_sage_attention: bool = False,
) -> Tensor:
    """Radial attention entry point.

    Args:
        query: Query [B, S, H, D].
        key: Key [B, S, H, D].
        value: Value [B, S, H, D].
        mask_map: MaskMap for mask caching.
        sparsity_type: "dense" or "radial".
        block_size: Block size (128 or 64).
        decay_factor: Window decay factor.
        model_type: Model architecture. Only `"wan"` is supported.
        pre_defined_mask: Text attention mask.
        use_sage_attention: Use SageAttention backend.

    Returns:
        Attention output [B, S, H*D].
    """
    if sparsity_type == "dense":
        video_mask = torch.ones(
            (mask_map.video_token_num // block_size, mask_map.video_token_num // block_size),
            device=query.device,
            dtype=torch.bool,
        )
    else:
        video_mask = (
            mask_map.query_log_mask(
                query,
                sparsity_type,
                block_size=block_size,
                decay_factor=decay_factor,
                model_type=model_type,
            )
            if mask_map
            else None
        )

    if use_sage_attention:
        return radial_attention_sage(query, key, value, mask_map, video_mask, pre_defined_mask, block_size)
    else:
        return radial_attention_flashinfer(query, key, value, mask_map, video_mask, pre_defined_mask, block_size)


__all__ = [
    "MaskMap",
    "radial_attention",
    "sparse_attention",
    "create_radial_mask_map",
    "clear_radial_mask_cache",
    "gen_log_mask_shrinked",
    "shrink_mask_strict",
    "get_window_width",
    "get_diagonal_split_mask",
    "sparge_mask_convert",
    "get_indptr_from_mask",
    "get_indices_from_mask",
    "get_cuda_arch_versions",
]


# Convenience functions for sparse attention (previously in sparse_attention.py)


def sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sparse_config: SparseAttentionConfig,
    mask_map: MaskMap,
    numeral_timestep: int = 0,
    layer_idx: int = 0,
    model_type: str = "wan",
) -> Tensor:
    """Compute sparse attention with dynamic density selection.

    Automatically selects dense or sparse attention based on timestep/layer.

    Args:
        q: Query tensor [B, S, H, D].
        k: Key tensor [B, S, H, D].
        v: Value tensor [B, S, H, D].
        sparse_config: SparseAttentionConfig instance.
        mask_map: Cached mask map.
        numeral_timestep: Current denoising timestep.
        layer_idx: Current transformer layer.
        model_type: Model architecture. Only `"wan"` is supported.

    Returns:
        Attention output tensor [B, S, H*D].
    """
    sparsity_type = (
        "dense" if sparse_config.should_use_dense(numeral_timestep, layer_idx) else sparse_config.sparse_impl
    )

    return radial_attention(
        query=q,
        key=k,
        value=v,
        mask_map=mask_map,
        sparsity_type=sparsity_type,
        block_size=sparse_config.block_size,
        decay_factor=sparse_config.decay_factor,
        model_type=model_type,
        pre_defined_mask=None,
        use_sage_attention=sparse_config.use_sage_attention,
    )


def create_radial_mask_map(
    video_token_num: int,
    num_frame: int,
) -> MaskMap:
    """Create a MaskMap for radial attention.

    Args:
        video_token_num: Total video tokens (frames * tokens_per_frame).
        num_frame: Number of frames in video.

    Returns:
        MaskMap instance for caching radial masks.
    """
    return MaskMap(video_token_num=video_token_num, num_frame=num_frame)


def clear_radial_mask_cache() -> None:
    """Clear the global radial attention mask and workspace cache.

    Call this when processing videos with different sizes to invalidate
    cached masks and free workspace memory.
    """
    MaskMap.clear_cache()
    logger.debug("Cleared radial attention mask and workspace cache")
