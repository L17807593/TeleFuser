# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 keyframe target-canvas preparation.

Geometry behavior:
- auto-aspect canvases delegate to the shared adaptive v2 shape resolver;
- cover-crop: aspect-preserving max-scale LANCZOS resize + center crop,
  upscaling refused unless explicitly allowed.

Both Qwen presentation and the visual-condition tokenizer consume the same
prepared canvas image supplied by the pipeline.
"""

from __future__ import annotations

from typing import Any


def minimax_h3_cover_crop_plan(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    allow_upscale: bool,
) -> dict[str, Any]:
    """Deterministic aspect-preserving cover-crop transform."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("cover_crop requires positive source width/height")
    scale = max(target_width / float(source_width), target_height / float(source_height))
    if scale > 1.0 and not allow_upscale:
        raise ValueError(
            "target_canvas cover_crop would upscale the source; set "
            f"allow_upscale=true (source={source_width}x{source_height}, "
            f"target={target_width}x{target_height})"
        )
    resized_width = max(target_width, int(round(source_width * scale)))
    resized_height = max(target_height, int(round(source_height * scale)))
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return {
        "scale": scale,
        "resized_size": (resized_width, resized_height),
        "crop_box": (left, top, left + target_width, top + target_height),
    }


def minimax_h3_prepare_keyframe_canvas(
    image: Any,
    *,
    target_width: int,
    target_height: int,
    allow_upscale: bool = False,
) -> Any:
    """Prepare a PIL image onto the target canvas.

    Identity (no resample) when the image already IS the canvas.
    """
    from PIL import Image

    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    plan = minimax_h3_cover_crop_plan(
        source_width=image.size[0],
        source_height=image.size[1],
        target_width=target_width,
        target_height=target_height,
        allow_upscale=allow_upscale,
    )
    resized = image.resize(plan["resized_size"], Image.Resampling.LANCZOS)
    return resized.crop(plan["crop_box"])


def minimax_h3_stretch_keyframe_canvas(
    image: Any,
    *,
    target_width: int,
    target_height: int,
) -> Any:
    """Stretch the FL first frame directly onto the resolved target canvas."""

    from PIL import Image

    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


__all__ = [
    "minimax_h3_cover_crop_plan",
    "minimax_h3_prepare_keyframe_canvas",
    "minimax_h3_stretch_keyframe_canvas",
]
