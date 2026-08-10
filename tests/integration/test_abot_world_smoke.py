from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from examples.abot_world._loader import get_pipeline
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService


@pytest.mark.gpu
@pytest.mark.slow
def test_abot_world_thirty_control_blocks_preserve_session_frame_count() -> None:
    """Run 30 ordered blocks through the LiveKit model-service contract.

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
    service = ABotWorldLiveKitService(
        pipeline,
        default_fps=12,
        output_queue_size=2,
        control_idle_timeout=3600.0,
        close_timeout=600.0,
    )
    session_id: str | None = None
    try:
        service.start()
        session_id = service.create_session(
            {
                "image_path": image_path,
                "prompt": "A smooth first-person exploration through a vivid natural landscape.",
                "seed": 42,
                "fps": 12,
                "control_latent_frames": 3,
            }
        )
        state = service._session(session_id)
        assert state is not None

        async def collect_blocks() -> list[dict]:
            chunks: list[dict] = []
            async for payload in service.pull_chunks(session_id):
                if payload["type"] == "preview":
                    service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                    continue
                assert payload["type"] == "chunk"
                chunks.append(payload)
                if len(chunks) == 30:
                    service.push_chunk(session_id, {"type": "stop"})
                    break
            return chunks

        chunks = asyncio.run(collect_blocks())
        assert [chunk["index"] for chunk in chunks] == list(range(30))
        assert all(chunk["frames"] for chunk in chunks)
        total_frames = sum(len(chunk["frames"]) for chunk in chunks)
        assert total_frames > 0
        assert state.pipeline_session.emitted_frames == total_frames
        assert not state.pipeline_session.closed
    finally:
        if session_id is not None:
            service.close_session(session_id)
        service.stop()
