from __future__ import annotations

import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from fastapi.testclient import TestClient

from telefuser.pipelines.lingbot_vla_v2.pipeline import LingBotVlaV2CanonicalActionChunk
from telefuser.pipelines.lingbot_vla_v2.robot_profile import ROBOTWIN_CAMERA_KEYS
from telefuser.pipelines.lingbot_vla_v2.service import (
    LingBotVlaV2ActionRequest,
    LingBotVlaV2Service,
    LingBotVlaV2ServiceConfig,
    create_lingbot_vla_v2_app,
)


def _encoded_image(*, data_url: bool = False) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}" if data_url else encoded


def _payload() -> dict:
    image = _encoded_image()
    return {
        "task": "pick up the red block",
        "state": [0.0] * 14,
        "camera_high": image,
        "camera_left_wrist": image,
        "camera_right_wrist": image,
        "seed": 7,
    }


class _Pipeline:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.closed = False
        self.observations = []
        self.seeds = []
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def __call__(self, observation, seed=None) -> LingBotVlaV2CanonicalActionChunk:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            self.observations.append(observation)
            self.seeds.append(seed)
            return LingBotVlaV2CanonicalActionChunk(
                canonical_normalized_actions=torch.zeros(2, 55),
                horizon=2,
                action_dim=55,
            )
        finally:
            with self._counter_lock:
                self.active -= 1

    def close(self) -> None:
        self.closed = True


def _config(**kwargs) -> LingBotVlaV2ServiceConfig:
    return LingBotVlaV2ServiceConfig(
        model_root="/models/lingbot-vla-v2-6b",
        qwen3vl_root="/models/Qwen3-VL-4B-Instruct",
        **kwargs,
    )


def test_app_serves_health_and_normalized_action_contract() -> None:
    pipeline = _Pipeline()
    config = _config(device="cuda:3")
    app = create_lingbot_vla_v2_app(config, pipeline_factory=lambda received: pipeline)

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post("/v1/vla/actions", json=_payload())

    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "model": "lingbot-vla-v2-6b-base",
        "device": "cuda:3",
        "policy_verified": False,
    }
    assert response.status_code == 200
    body = response.json()
    assert body["horizon"] == 2
    assert body["action_dim"] == 55
    assert body["checkpoint_variant"] == "base"
    assert body["policy_verified"] is False
    assert body["verification_status"] == "unverified_official_6b_base"
    assert len(body["canonical_normalized_actions"]) == 2
    assert len(body["canonical_normalized_actions"][0]) == 55
    assert pipeline.seeds == [7]
    assert tuple(pipeline.observations[0].images) == ROBOTWIN_CAMERA_KEYS
    assert all(image.mode == "RGB" for image in pipeline.observations[0].images.values())
    assert pipeline.closed is True


def test_app_accepts_image_data_urls() -> None:
    pipeline = _Pipeline()
    payload = _payload()
    payload["camera_high"] = _encoded_image(data_url=True)
    app = create_lingbot_vla_v2_app(_config(), pipeline_factory=lambda received: pipeline)

    with TestClient(app) as client:
        response = client.post("/v1/vla/actions", json=payload)

    assert response.status_code == 200


def test_app_rejects_invalid_observations_without_running_policy() -> None:
    pipeline = _Pipeline()
    app = create_lingbot_vla_v2_app(_config(), pipeline_factory=lambda received: pipeline)
    payload = _payload()
    payload["camera_high"] = "not-base64"

    with TestClient(app) as client:
        invalid_image = client.post("/v1/vla/actions", json=payload)
        invalid_state = client.post("/v1/vla/actions", json={**_payload(), "state": [0.0] * 13})
        extra_field = client.post("/v1/vla/actions", json={**_payload(), "output_path": "/tmp/action"})

    assert invalid_image.status_code == 422
    assert invalid_image.json()["detail"] == "image must be valid base64"
    assert invalid_state.status_code == 422
    assert extra_field.status_code == 422
    assert pipeline.observations == []


def test_service_serializes_policy_calls() -> None:
    pipeline = _Pipeline(delay=0.02)
    service = LingBotVlaV2Service(pipeline, _config())
    request = LingBotVlaV2ActionRequest.model_validate(_payload())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(service.predict, request) for _ in range(2)]
        responses = [future.result() for future in futures]

    assert [response.horizon for response in responses] == [2, 2]
    assert pipeline.max_active == 1


def test_service_config_rejects_non_positive_image_limit() -> None:
    try:
        _config(max_image_bytes=0)
    except ValueError as error:
        assert str(error) == "max_image_bytes must be positive"
    else:
        raise AssertionError("expected an invalid image size limit to be rejected")
