from __future__ import annotations

from typing import Any

import pytest

from tools.validation import validate_lingbot_vla_v2_service_faults as validator


class _Response:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses

    def request(self, *args: object, **kwargs: object) -> _Response:
        return self.responses.pop(0)


def test_expect_invalid_request_accepts_synchronous_rejection() -> None:
    result = validator._expect_rejected_or_failed(
        _Session([_Response(422, {"detail": "invalid"})]),
        "http://127.0.0.1:18080",
        {"task": "vla_action"},
        case_name="invalid",
        http_timeout_seconds=1.0,
        task_timeout_seconds=1.0,
        poll_interval_seconds=0.001,
    )

    assert result == {"name": "invalid", "passed": True, "handling": "rejected", "http_status": 422}


def test_expect_invalid_request_accepts_asynchronous_failure() -> None:
    session = _Session(
        [
            _Response(200, {"task_id": "task-1", "task_status": "pending"}),
            _Response(200, {"task_id": "task-1", "status": "failed", "error": "bad input"}),
        ]
    )

    result = validator._expect_rejected_or_failed(
        session,
        "http://127.0.0.1:18080",
        {"task": "vla_action"},
        case_name="invalid",
        http_timeout_seconds=1.0,
        task_timeout_seconds=1.0,
        poll_interval_seconds=0.001,
    )

    assert result["handling"] == "asynchronous_failure"
    assert result["terminal_status"] == "failed"
    assert result["error"] == "bad input"


def test_select_replica_process_filters_unrelated_and_root_processes() -> None:
    rows = "\n".join(
        [
            "100, GPU-a",
            "101, GPU-a",
            "102, GPU-b",
            "999, GPU-a",
        ]
    )

    selected = validator.select_replica_process(
        rows,
        service_process_ids={101, 102},
        gpu_uuid_to_index={"GPU-a": "0", "GPU-b": "1"},
        gpu_index="0",
        service_pid=100,
    )

    assert selected == 101


def test_select_replica_process_requires_unambiguous_target() -> None:
    with pytest.raises(validator.FaultValidationFailure, match="exactly one"):
        validator.select_replica_process(
            "101, GPU-a\n102, GPU-a",
            service_process_ids={101, 102},
            gpu_uuid_to_index={"GPU-a": "0"},
            gpu_index="0",
            service_pid=100,
        )


def test_gpu_compute_process_ids_ignores_malformed_rows() -> None:
    rows = "101, GPU-a\ninvalid, GPU-b\n102, GPU-c\n"

    assert validator.gpu_compute_process_ids(rows) == {101, 102}


def test_validate_pool_degradation_requires_one_dead_replica_and_reduced_capacity() -> None:
    before = {
        "effective_max_concurrent_tasks": 2,
        "pool": [{"id": 0, "status": "idle"}, {"id": 1, "status": "idle"}],
    }
    after = {
        "effective_max_concurrent_tasks": 1,
        "pool": [{"id": 0, "status": "dead"}, {"id": 1, "status": "idle"}],
    }

    result = validator.validate_pool_degradation(before, after)

    assert result["before_capacity"] == 2
    assert result["after_capacity"] == 1
    assert result["dead_replica_ids"] == [0]
    assert result["recovery_semantics"] == "graceful_capacity_degradation_without_automatic_restart"


def test_validate_pool_degradation_rejects_unchanged_capacity() -> None:
    before = {
        "effective_max_concurrent_tasks": 2,
        "pool": [{"id": 0, "status": "idle"}, {"id": 1, "status": "idle"}],
    }
    after = {
        "effective_max_concurrent_tasks": 2,
        "pool": [{"id": 0, "status": "dead"}, {"id": 1, "status": "idle"}],
    }

    with pytest.raises(validator.FaultValidationFailure, match="capacity 1"):
        validator.validate_pool_degradation(before, after)
