from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from examples.abot_world._loader import get_pipeline
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline


@pytest.mark.gpu
@pytest.mark.slow
def test_abot_world_thirty_control_blocks_preserve_session_frame_count() -> None:
    """Run the requested long-forcing smoke against a local checkpoint.

    Set ``ABOT_WORLD_MODEL_ROOT`` and ``ABOT_WORLD_TEST_IMAGE`` to run this
    test. It is skipped on ordinary CPU CI because loading the release
    checkpoint is intentionally expensive.
    """
    model_root = os.environ.get("ABOT_WORLD_MODEL_ROOT")
    image_path = os.environ.get("ABOT_WORLD_TEST_IMAGE")
    if not model_root or not image_path:
        pytest.skip("set ABOT_WORLD_MODEL_ROOT and ABOT_WORLD_TEST_IMAGE to run the ABot GPU smoke test")
    if not Path(model_root).is_dir() or not Path(image_path).is_file():
        pytest.skip("ABot checkpoint or test image is unavailable")

    pipeline = get_pipeline(
        model_root,
        height=480,
        width=832,
        latent_frames=31,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    try:
        pipeline.preload_models()
        with Image.open(image_path) as source:
            session = pipeline.create_interactive_session(
                source.convert("RGB"),
                "A smooth first-person exploration through a vivid natural landscape.",
                seed=42,
            )
        total_frames = 0
        try:
            for _ in range(30):
                frames = pipeline.generate_next_block(session, {"W": True}, control_latent_frames=3)
                assert frames
                total_frames += len(frames)
            assert total_frames > 0
            assert session.emitted_frames == total_frames
            assert not session.closed
        finally:
            pipeline.close_interactive_session(session)
    finally:
        pipeline.close()
