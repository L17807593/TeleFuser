# SPDX-License-Identifier: Apache-2.0
# ViT runtime helpers for the MiniMax H3 visual VAE.
from collections.abc import Sequence
from typing import Tuple

import torch


def create_token_ids(patch_dims, device, dtype, id_type="length_normalized", flatten=True):
    coords_list = []

    if isinstance(id_type, str):
        id_type_list = [id_type] * len(patch_dims)
    elif isinstance(id_type, list):
        id_type_list = id_type
        if len(id_type_list) != len(patch_dims):
            raise ValueError("id_type list must match patch_dims")
    else:
        raise ValueError("id_type must be a string or a list")

    if "area_normalized" in id_type_list or id_type == "area_normalized":
        raise NotImplementedError("area_normalized id_type is not supported in this inference-only bundle")

    for _dim_size, _id_type in zip(patch_dims, id_type_list):
        if isinstance(_dim_size, torch.Tensor):
            coords_list.append(_dim_size.to(device=device, dtype=dtype))
            continue

        if _id_type == "length_normalized":
            coords = torch.arange(0.5, _dim_size, dtype=dtype, device=device)
            coords = coords / _dim_size
            coords = 2.0 * coords - 1.0
        else:
            coords = torch.arange(_dim_size, dtype=dtype, device=device)

        coords_list.append(coords)

    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    if flatten:
        coords = coords.flatten(0, len(patch_dims) - 1)

    return coords.unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_impl(t: torch.Tensor, rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rotary_pos_emb[:2]

    if cos.dim() != 4:
        raise ValueError(f"cos must be [B, N, 1, D], got {cos.shape}")

    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]

    if rot_dim < t_dim:
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        scaled = t_rot * cos
        scaled.add_(_rotate_half(t_rot) * sin)
        t_rot = scaled
        t = torch.cat((t_rot, t_pass), dim=-1)
    else:
        scaled = t * cos
        scaled.add_(_rotate_half(t) * sin)
        t = scaled

    return t


def prepare_rotary_pos_emb(
    rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Cast-free eager rotary cache used by the parity path."""
    cos, sin = rotary_pos_emb
    del dtype
    return cos, sin


def apply_rotary_pos_emb(t: torch.Tensor, rotary_pos_emb: Sequence[torch.Tensor]) -> torch.Tensor:
    return _apply_rotary_pos_emb_impl(t, rotary_pos_emb)


def apply_rotary_pos_emb_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the released NeoX rotary recipe to Q and K."""
    return (
        apply_rotary_pos_emb(query, rotary_pos_emb),
        apply_rotary_pos_emb(key, rotary_pos_emb),
    )
