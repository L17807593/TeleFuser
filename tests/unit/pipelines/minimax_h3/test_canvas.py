import pytest
from PIL import Image

from telefuser.pipelines.minimax_h3.canvas import (
    minimax_h3_cover_crop_plan,
    minimax_h3_prepare_keyframe_canvas,
)


def test_cover_crop_is_centered_and_refuses_implicit_upscale() -> None:
    plan = minimax_h3_cover_crop_plan(
        source_width=1600,
        source_height=1200,
        target_width=1344,
        target_height=768,
        allow_upscale=False,
    )
    assert plan["resized_size"] == (1344, 1008)
    assert plan["crop_box"] == (0, 120, 1344, 888)
    with pytest.raises(ValueError, match="allow_upscale=true"):
        minimax_h3_cover_crop_plan(
            source_width=320,
            source_height=240,
            target_width=1344,
            target_height=768,
            allow_upscale=False,
        )


def test_prepare_keyframe_canvas_returns_target_rgb_image() -> None:
    source = Image.new("RGBA", (1600, 1200), color=(255, 0, 0, 128))
    actual = minimax_h3_prepare_keyframe_canvas(source, target_width=1344, target_height=768, allow_upscale=False)
    assert actual.mode == "RGB"
    assert actual.size == (1344, 768)
