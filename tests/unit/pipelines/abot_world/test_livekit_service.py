from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from telefuser.pipelines.abot_world.service import (
    ABotWorldLiveKitService,
    _ABotWorldLiveKitSession,
)
from telefuser.service.core.stream_pipeline_service import BidirectionalService


class _FakePipeline:
    def __init__(self) -> None:
        self.config = SimpleNamespace(width=8, height=8)
        self.generate_calls: list[dict[str, bool]] = []
        self.closed_sessions: list[object] = []
        self.closed = False

    def preload_models(self) -> None:
        return None

    def create_interactive_session(self, image: Image.Image, prompt: str, *, seed: int) -> object:
        assert image.mode == "RGB"
        assert prompt
        return object()

    def generate_next_block(self, session: object, controls: dict[str, bool], *, control_latent_frames: int) -> list:
        assert control_latent_frames == 3
        self.generate_calls.append(controls)
        return [Image.new("RGB", (8, 8), color=(20, len(self.generate_calls), 40))]

    def close_interactive_session(self, session: object) -> None:
        self.closed_sessions.append(session)

    def close(self) -> None:
        self.closed = True


def _service(**kwargs: object) -> tuple[ABotWorldLiveKitService, _FakePipeline]:
    pipeline = _FakePipeline()
    service = ABotWorldLiveKitService(
        pipeline,
        default_session_config={"prompt": "test prompt"},
        **kwargs,
    )
    return service, pipeline


def test_service_matches_shared_bidirectional_contract() -> None:
    service, _ = _service()
    assert isinstance(service, BidirectionalService)
    assert service.configure_session_capacity(1)["effective_capacity"] == 1
    with pytest.raises(ValueError, match="one retained"):
        service.configure_session_capacity(2)


def test_session_is_preview_only_until_control_and_preserves_chunk_order() -> None:
    service, pipeline = _service(output_queue_size=2)
    session_id = service.create_session({"image": Image.new("RGB", (10, 10)), "prompt": "test"})
    state = service._session(session_id)
    assert state is not None
    preview = state.output_queue.get(timeout=1)
    assert preview["type"] == "preview"
    assert pipeline.generate_calls == []

    service.push_chunk(session_id, {"type": "control_state", "controls": ["ArrowUp"]})
    generated = state.output_queue.get(timeout=1)
    assert generated["type"] == "chunk"
    assert generated["index"] == 0
    assert generated["controls"] == ["W"]
    assert pipeline.generate_calls == [{"W": True}]
    service.close_session(session_id)
    assert len(pipeline.closed_sessions) == 1


def test_control_aliases_and_release_stop_generation() -> None:
    service, pipeline = _service(output_queue_size=4)
    session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    state = service._session(session_id)
    assert state is not None
    state.output_queue.get(timeout=1)
    service.push_chunk(session_id, {"type": "control", "control": "KeyJ", "event": "press"})
    assert state.output_queue.get(timeout=1)["controls"] == ["J"]
    service.push_chunk(session_id, {"type": "control", "control": "KeyJ", "event": "release"})
    calls_after_release = len(pipeline.generate_calls)
    time.sleep(0.15)
    assert len(pipeline.generate_calls) == calls_after_release
    service.close_session(session_id)


def test_bounded_output_queue_applies_backpressure_without_dropping() -> None:
    service, _ = _service(output_queue_size=1)
    state = _ABotWorldLiveKitSession(
        session_id="test",
        pipeline_session=object(),
        output_queue=queue.Queue(maxsize=1),
        control_event=threading.Event(),
        config={"fps": 12, "control_latent_frames": 3},
    )
    first = {"type": "chunk", "index": 0}
    second = {"type": "chunk", "index": 1}
    state.output_queue.put(first)
    completed = threading.Event()

    def produce() -> None:
        assert service._put_output(state, second)
        completed.set()

    producer = threading.Thread(target=produce)
    producer.start()
    time.sleep(0.1)
    assert not completed.is_set()
    assert state.output_queue.get(timeout=1) is first
    producer.join(timeout=1)
    assert completed.is_set()
    assert state.output_queue.get(timeout=1) is second


def test_public_pull_stream_yields_thirty_complete_blocks_in_order() -> None:
    service, pipeline = _service(output_queue_size=1, control_idle_timeout=30.0)
    session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    state = service._session(session_id)
    assert state is not None

    async def collect() -> list[dict]:
        chunks: list[dict] = []
        preview_seen = False
        async for payload in service.pull_chunks(session_id):
            if payload["type"] == "preview":
                preview_seen = True
                service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                continue
            assert preview_seen
            assert payload["type"] == "chunk"
            chunks.append(payload)
            if len(chunks) == 30:
                service.push_chunk(session_id, {"type": "control", "control": "KeyW", "event": "release"})
                break
            state.control_event.set()
        return chunks

    try:
        chunks = asyncio.run(collect())
        assert [chunk["index"] for chunk in chunks] == list(range(30))
        assert all(len(chunk["frames"]) == 1 for chunk in chunks)
        assert pipeline.generate_calls == [{"W": True}] * 30
    finally:
        service.close_session(session_id)


def test_public_pull_stream_drains_preview_then_finishes_after_stop() -> None:
    service, pipeline = _service(output_queue_size=1)
    session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    service.push_chunk(session_id, {"type": "stop"})

    async def collect() -> list[dict]:
        return [payload async for payload in service.pull_chunks(session_id)]

    try:
        payloads = asyncio.run(collect())
        assert [payload["type"] for payload in payloads] == ["preview"]
        assert pipeline.generate_calls == []
    finally:
        service.close_session(session_id)


def test_second_livekit_session_is_rejected_until_first_is_closed() -> None:
    service, pipeline = _service()
    first_session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    try:
        with pytest.raises(RuntimeError, match="one retained session"):
            service.create_session({"image": Image.new("RGB", (8, 8))})
    finally:
        service.close_session(first_session_id)

    second_session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    service.close_session(second_session_id)
    assert len(pipeline.closed_sessions) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "control_state", "controls": "KeyW"},
        {"type": "control", "control": "unsupported", "event": "press"},
        {"type": "unsupported"},
    ],
)
def test_invalid_livekit_control_payloads_are_rejected(payload: dict) -> None:
    service, _ = _service()
    session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    try:
        with pytest.raises(ValueError):
            service.push_chunk(session_id, payload)
    finally:
        service.close_session(session_id)
