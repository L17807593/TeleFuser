from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import pytest

from tools.validation import validate_lingbot_vla_v2_structured_service as validator


def _action_result(value: float = 0.25) -> dict[str, Any]:
    return {
        "canonical_normalized_actions": [[value] * 55 for _ in range(50)],
        "horizon": 50,
        "action_dim": 55,
        "checkpoint_variant": "base",
        "policy_verified": False,
        "verification_status": "unverified_official_6b_base",
    }


def _metadata() -> dict[str, Any]:
    parameters = {
        "instruction": {"type": "string", "required": True},
        "state": {"type": "array", "required": True},
        "camera_high": {"type": "string", "required": True},
        "camera_left_wrist": {"type": "string", "required": True},
        "camera_right_wrist": {"type": "string", "required": True},
        "seed": {"type": "integer", "required": False},
    }
    return {
        "declared_pipeline_contract": True,
        "supported_tasks": ["vla_action"],
        "supported_media_types": ["structured"],
        "task_contracts": {
            "vla_action": {
                "media_type": "structured",
                "required_inputs": ["camera_high", "camera_left_wrist", "camera_right_wrist"],
                "optional_inputs": [],
                "parameters": parameters,
            }
        },
    }


def test_parse_state_json_requires_fourteen_finite_numbers() -> None:
    assert validator.parse_state_json("[0,1,2,3,4,5,6,7,8,9,10,11,12,13]") == [float(index) for index in range(14)]

    with pytest.raises(argparse.ArgumentTypeError, match="exactly 14"):
        validator.parse_state_json("[0, 1]")
    with pytest.raises(argparse.ArgumentTypeError, match="finite numbers"):
        validator.parse_state_json("[0,1,2,3,4,5,6,7,8,9,10,11,12,true]")


def test_validate_service_metadata_requires_native_structured_contract() -> None:
    validator.validate_service_metadata(_metadata())

    metadata = _metadata()
    metadata["task_contracts"]["vla_action"]["media_type"] = "video"
    with pytest.raises(validator.ValidationFailure, match="structured task contract"):
        validator.validate_service_metadata(metadata)

    metadata = _metadata()
    del metadata["task_contracts"]["vla_action"]["parameters"]["seed"]
    with pytest.raises(validator.ValidationFailure, match="parameter fields changed"):
        validator.validate_service_metadata(metadata)


def test_validate_action_result_reports_shape_stats_and_fingerprint() -> None:
    summary = validator.validate_action_result(_action_result(), expected_horizon=50, expected_action_dim=55)

    assert summary["shape"] == [50, 55]
    assert summary["value_count"] == 2750
    assert summary["minimum"] == 0.25
    assert summary["maximum"] == 0.25
    assert summary["policy_verified"] is False
    assert len(summary["sha256_float64_le"]) == 64


def test_validate_action_result_rejects_additive_result_fields() -> None:
    result = _action_result()
    result["debug"] = "unstable"

    with pytest.raises(validator.ValidationFailure, match="result fields changed"):
        validator.validate_action_result(result, expected_horizon=50, expected_action_dim=55)


def test_validate_task_status_rejects_sensitive_echo_and_missing_fields() -> None:
    status = {
        "task_id": "task-1",
        "status": "completed",
        "inference_time_s": 0.25,
        "peak_memory_mb": None,
        "result": _action_result(),
    }
    validator.validate_task_status(status, task_id="task-1")

    leaked = dict(status, camera_high="base64")
    with pytest.raises(validator.ValidationFailure, match="sensitive image"):
        validator.validate_task_status(leaked, task_id="task-1")


def test_compare_windows_reports_first_and_last_measurement_change() -> None:
    result = validator.compare_windows([1.0, 1.0, 1.0, 1.2, 1.2, 1.2])

    assert result is not None
    assert result["window_count"] == 1
    assert result["change_percent"] == pytest.approx(20.0)


def test_parse_gpu_process_memory_filters_process_tree_and_physical_gpu() -> None:
    output = "\n".join(
        [
            "101, GPU-a, 1024",
            "102, GPU-a, 512",
            "101, GPU-b, 2048",
            "999, GPU-a, 4096",
            "malformed",
        ]
    )

    result = validator._parse_gpu_process_memory(
        output,
        process_ids={101, 102},
        uuid_to_index={"GPU-a": "0", "GPU-b": "1"},
        gpu_indexes={"0"},
    )

    assert result == {"0": 1536.0}


def test_parse_gpu_indexes_requires_integer_indexes() -> None:
    assert validator.parse_gpu_indexes("0, 2") == {"0", "2"}

    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        validator.parse_gpu_indexes("0,GPU-a")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda result: result.update(horizon=49), "horizon field"),
        (lambda result: result["canonical_normalized_actions"][0].pop(), "row 0"),
        (lambda result: result["canonical_normalized_actions"][0].__setitem__(0, float("nan")), "non-finite"),
    ],
)
def test_validate_action_result_rejects_invalid_contract(mutation, match: str) -> None:
    result = _action_result()
    mutation(result)

    with pytest.raises(validator.ValidationFailure, match=match):
        validator.validate_action_result(result, expected_horizon=50, expected_action_dim=55)


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _Session:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.trust_env = False

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def request(self, method: str, url: str, *, json=None, timeout=None) -> _Response:
        if method == "POST":
            assert url.endswith("/v1/tasks/structured")
            assert json["task"] == "vla_action"
            with self.state["lock"]:
                self.state["next_id"] += 1
                task_id = f"task-{self.state['next_id']}"
            return _Response({"task_id": task_id, "task_status": "pending"})
        task_id = url.rsplit("/", maxsplit=2)[-2]
        return _Response(
            {
                "task_id": task_id,
                "status": "completed",
                "inference_time_s": 0.25,
                "peak_memory_mb": 128.0,
                "result": _action_result(),
            }
        )


def test_run_workload_exercises_concurrent_structured_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"lock": threading.Lock(), "next_id": 0}
    monkeypatch.setattr(validator, "_new_session", lambda: _Session(state))
    config = validator.RequestConfig(
        base_url="http://127.0.0.1:18080",
        payload={"task": "vla_action"},
        http_timeout_seconds=1.0,
        task_timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        expected_horizon=50,
        expected_action_dim=55,
    )

    report = validator.run_workload(
        config,
        request_count=4,
        duration_seconds=None,
        concurrency=2,
        max_records=10,
    )

    assert report["requests"]["total"] == 4
    assert report["requests"]["succeeded"] == 4
    assert report["requests"]["failed"] == 0
    assert report["requests"]["unique_task_ids"] == 4
    assert report["latency_seconds"]["target_inference"]["mean"] == 0.25
    assert len(report["retained_records"]["successful"]) == 4


def test_resource_sampler_bounds_samples_and_reports_trend() -> None:
    state = {"value": 100.0}
    ready = threading.Event()

    def sample() -> dict[str, Any]:
        state["value"] += 10.0
        ready.set()
        return {"process_ids": [123], "cpu_rss_mib": state["value"], "gpu_memory_mib": {"0": 2048.0}}

    sampler = validator.ResourceSampler(
        123,
        interval_seconds=0.001,
        max_samples=4,
        sample_function=sample,
    )
    sampler.start()
    assert ready.wait(1.0)
    time.sleep(0.01)
    report = sampler.stop()

    assert report["sample_count"] >= 2
    assert len(report["retained_samples"]) <= 4
    assert report["cpu_rss_mib"]["trend"]["last_mean"] > report["cpu_rss_mib"]["trend"]["first_mean"]
    assert report["gpu_memory_mib"]["0"]["distribution"]["mean"] == 2048.0
