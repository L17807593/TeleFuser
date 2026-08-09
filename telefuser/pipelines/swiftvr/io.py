"""Tensor preprocessing and output conversion for SwiftVR."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_INTERP_NEEDS_ALIGN = ("linear", "bilinear", "bicubic", "trilinear")


def preprocess_clip_uint8(
    frames_uint8: torch.Tensor,
    out_h: int,
    out_w: int,
    mode: str,
    pad_h: int,
    pad_w: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert CUDA uint8 THWC frames to padded target-dtype NTCHW frames in [0, 1]."""
    frames = frames_uint8.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)
    _, _, height, width = frames.shape
    if (height, width) != (out_h, out_w):
        if mode in _INTERP_NEEDS_ALIGN:
            frames = F.interpolate(frames, size=(out_h, out_w), mode=mode, align_corners=False)
        else:
            frames = F.interpolate(frames, size=(out_h, out_w), mode=mode)
    frames = frames / 255.0
    if pad_h > 0 or pad_w > 0:
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return frames.unsqueeze(0)


def crop_spatial_padding_ntchw(video: torch.Tensor | None, pad_h: int = 0, pad_w: int = 0) -> torch.Tensor | None:
    """Remove bottom/right spatial padding from an NTCHW tensor."""
    if video is None:
        return None
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


def ntchw_to_uint8_frames(video: torch.Tensor | None) -> np.ndarray | None:
    """Convert [0, 1] NTCHW output to uint8 THWC frames on the host."""
    if video is None or video.numel() == 0 or video.shape[1] == 0:
        return None
    frames = (video[0].permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8)
    if frames.device.type != "cuda":
        return frames.cpu().numpy()
    cpu_frames = torch.empty_like(frames, device="cpu", pin_memory=True)
    cpu_frames.copy_(frames, non_blocking=True)
    torch.cuda.current_stream(frames.device).synchronize()
    return cpu_frames.numpy()
