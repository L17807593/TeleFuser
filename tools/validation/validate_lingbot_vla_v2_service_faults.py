"""Validate fault handling of a real LingBot-VLA v2 structured service."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import requests

try:
    from tools.validation import validate_lingbot_vla_v2_structured_service as structured_validator
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    import validate_lingbot_vla_v2_structured_service as structured_validator

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class FaultValidationFailure(RuntimeError):
    """Raised when the service violates an expected fault-handling behavior."""


def _new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _response_body(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as error:
        raise FaultValidationFailure(f"HTTP {response.status_code} response is not JSON") from error
    if not isinstance(body, dict):
        raise FaultValidationFailure(f"HTTP {response.status_code} response is not a JSON object")
    return body


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    response = session.request(method, url, json=payload, timeout=timeout)
    return response.status_code, _response_body(response)


def _wait_terminal(
    session: requests.Session,
    base_url: str,
    task_id: str,
    *,
    http_timeout_seconds: float,
    task_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + task_timeout_seconds
    while time.monotonic() < deadline:
        status_code, body = _request_json(
            session,
            "GET",
            f"{base_url}/v1/tasks/{task_id}/status",
            timeout=http_timeout_seconds,
        )
        if status_code != 200:
            raise FaultValidationFailure(f"task status returned HTTP {status_code}: {body}")
        status = body.get("status") or body.get("task_status")
        if status in _TERMINAL_STATUSES:
            return body
        time.sleep(poll_interval_seconds)
    raise FaultValidationFailure(f"task {task_id} did not become terminal within {task_timeout_seconds:g}s")


def _expect_rejected_or_failed(
    session: requests.Session,
    base_url: str,
    payload: dict[str, Any],
    *,
    case_name: str,
    http_timeout_seconds: float,
    task_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    status_code, created = _request_json(
        session,
        "POST",
        f"{base_url}/v1/tasks/structured",
        timeout=http_timeout_seconds,
        payload=payload,
    )
    if 400 <= status_code < 500:
        return {"name": case_name, "passed": True, "handling": "rejected", "http_status": status_code}
    if status_code != 200:
        raise FaultValidationFailure(f"{case_name} returned unexpected HTTP {status_code}: {created}")
    task_id = created.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise FaultValidationFailure(f"{case_name} accepted without a task_id")
    terminal = _wait_terminal(
        session,
        base_url,
        task_id,
        http_timeout_seconds=http_timeout_seconds,
        task_timeout_seconds=task_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    if terminal.get("status") != "failed":
        raise FaultValidationFailure(f"{case_name} reached unexpected terminal status {terminal.get('status')}")
    return {
        "name": case_name,
        "passed": True,
        "handling": "asynchronous_failure",
        "http_status": status_code,
        "task_id": task_id,
        "terminal_status": "failed",
        "error": str(terminal.get("error") or "")[:1000],
    }


def validate_request_faults(
    session: requests.Session,
    base_url: str,
    payload: dict[str, Any],
    *,
    http_timeout_seconds: float,
    task_timeout_seconds: float,
    poll_interval_seconds: float,
) -> list[dict[str, Any]]:
    """Validate required fields, payload validation, and request cancellation."""
    cases: list[dict[str, Any]] = []

    missing_camera = dict(payload)
    missing_camera.pop("camera_high", None)
    cases.append(
        _expect_rejected_or_failed(
            session,
            base_url,
            missing_camera,
            case_name="missing_required_camera",
            http_timeout_seconds=http_timeout_seconds,
            task_timeout_seconds=task_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    )

    invalid_state = dict(payload)
    invalid_state["state"] = [0.0] * 13
    cases.append(
        _expect_rejected_or_failed(
            session,
            base_url,
            invalid_state,
            case_name="invalid_state_dimension",
            http_timeout_seconds=http_timeout_seconds,
            task_timeout_seconds=task_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    )

    invalid_image = dict(payload)
    invalid_image["camera_high"] = "not-valid-base64"
    cases.append(
        _expect_rejected_or_failed(
            session,
            base_url,
            invalid_image,
            case_name="invalid_camera_base64",
            http_timeout_seconds=http_timeout_seconds,
            task_timeout_seconds=task_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    )

    status_code, created = _request_json(
        session,
        "POST",
        f"{base_url}/v1/tasks/structured",
        timeout=http_timeout_seconds,
        payload=payload,
    )
    if status_code != 200 or not isinstance(created.get("task_id"), str):
        raise FaultValidationFailure(f"cancellation case was not accepted: HTTP {status_code}: {created}")
    task_id = created["task_id"]
    cancel_status, cancellation = _request_json(
        session,
        "DELETE",
        f"{base_url}/v1/tasks/{task_id}",
        timeout=http_timeout_seconds,
    )
    if cancel_status != 200 or cancellation.get("stop_status") not in {"success", "do_nothing"}:
        raise FaultValidationFailure(f"task cancellation failed: HTTP {cancel_status}: {cancellation}")
    terminal = _wait_terminal(
        session,
        base_url,
        task_id,
        http_timeout_seconds=http_timeout_seconds,
        task_timeout_seconds=task_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    terminal_status = terminal.get("status")
    if terminal_status not in {"cancelled", "completed"}:
        raise FaultValidationFailure(f"cancelled task reached unexpected terminal status {terminal_status}")
    cases.append(
        {
            "name": "client_cancellation",
            "passed": True,
            "task_id": task_id,
            "stop_status": cancellation.get("stop_status"),
            "terminal_status": terminal_status,
            "race_with_completion": terminal_status == "completed",
        }
    )
    return cases


def _gpu_uuid_to_index() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {
        fields[1]: fields[0]
        for line in result.stdout.splitlines()
        if len(fields := [field.strip() for field in line.split(",")]) == 2
    }


def select_replica_process(
    compute_rows: str,
    *,
    service_process_ids: set[int],
    gpu_uuid_to_index: dict[str, str],
    gpu_index: str,
    service_pid: int,
) -> int:
    """Select exactly one descendant compute process on a physical GPU."""
    candidates: set[int] = set()
    for line in compute_rows.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        observed_index = gpu_uuid_to_index.get(fields[1], fields[1])
        if pid != service_pid and pid in service_process_ids and observed_index == gpu_index:
            candidates.add(pid)
    if len(candidates) != 1:
        raise FaultValidationFailure(
            f"expected exactly one service descendant compute process on GPU {gpu_index}, observed {sorted(candidates)}"
        )
    return candidates.pop()


def discover_replica_process(service_pid: int, gpu_index: str) -> int:
    """Discover one replica process without considering unrelated system processes."""
    root = psutil.Process(service_pid)
    service_process_ids = {process.pid for process in root.children(recursive=True)}
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return select_replica_process(
        result.stdout,
        service_process_ids=service_process_ids,
        gpu_uuid_to_index=_gpu_uuid_to_index(),
        gpu_index=gpu_index,
        service_pid=service_pid,
    )


def gpu_compute_process_ids(compute_rows: str) -> set[int]:
    """Parse compute PIDs from nvidia-smi rows."""
    process_ids: set[int] = set()
    for line in compute_rows.splitlines():
        try:
            process_ids.add(int(line.split(",", 1)[0].strip()))
        except (ValueError, IndexError):
            continue
    return process_ids


def wait_for_replica_exit(replica_pid: int, *, timeout_seconds: float = 10.0) -> None:
    """Wait until a replica is exited or zombie and no longer owns GPU memory."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            process_exited = psutil.Process(replica_pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            process_exited = True
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if process_exited and replica_pid not in gpu_compute_process_ids(result.stdout):
            return
        time.sleep(0.1)
    raise FaultValidationFailure(f"replica process {replica_pid} did not exit and release GPU memory after SIGTERM")


def validate_pool_degradation(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Validate one-replica capacity reduction without requiring automatic restart."""
    before_pool = before.get("pool")
    after_pool = after.get("pool")
    if not isinstance(before_pool, list) or len(before_pool) < 2:
        raise FaultValidationFailure("replica termination requires service status with at least two pool entries")
    if not isinstance(after_pool, list) or len(after_pool) != len(before_pool):
        raise FaultValidationFailure("pool status disappeared or changed size after replica termination")
    before_capacity = before.get("effective_max_concurrent_tasks")
    after_capacity = after.get("effective_max_concurrent_tasks")
    if not isinstance(before_capacity, int) or not isinstance(after_capacity, int):
        raise FaultValidationFailure("service status has no integer effective capacity")
    dead = [replica for replica in after_pool if replica.get("status") == "dead"]
    live = [replica for replica in after_pool if replica.get("status") != "dead"]
    if len(dead) != 1 or not live or after_capacity != before_capacity - 1:
        raise FaultValidationFailure(
            f"expected one dead replica and capacity {before_capacity - 1}, "
            f"observed dead={len(dead)}, capacity={after_capacity}"
        )
    return {
        "before_capacity": before_capacity,
        "after_capacity": after_capacity,
        "dead_replica_ids": [replica.get("id") for replica in dead],
        "live_replica_ids": [replica.get("id") for replica in live],
        "recovery_semantics": "graceful_capacity_degradation_without_automatic_restart",
    }


def validate_replica_exit(
    session: requests.Session,
    base_url: str,
    payload: dict[str, Any],
    *,
    service_pid: int,
    gpu_index: str,
    http_timeout_seconds: float,
    task_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Terminate one explicitly selected replica and validate graceful degradation."""
    status_code, before = _request_json(session, "GET", f"{base_url}/v1/service/status", timeout=http_timeout_seconds)
    if status_code != 200 or before.get("execution_mode") != "concurrent_pipeline_pool":
        raise FaultValidationFailure("replica termination requires a ready concurrent pipeline pool")
    replica_pid = discover_replica_process(service_pid, gpu_index)
    os.kill(replica_pid, signal.SIGTERM)
    wait_for_replica_exit(replica_pid)

    failed_attempts: list[str] = []
    successful_request: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    config = structured_validator.RequestConfig(
        base_url=base_url,
        payload=payload,
        http_timeout_seconds=http_timeout_seconds,
        task_timeout_seconds=task_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        expected_horizon=50,
        expected_action_dim=55,
    )
    for attempt in range(3):
        record = structured_validator.execute_request(
            session,
            config,
            request_index=attempt,
            worker_index=0,
            run_started_at=time.perf_counter(),
        )
        if record.get("outcome") == "succeeded":
            successful_request = record
        else:
            failed_attempts.append(str(record.get("error") or "unknown request failure"))
        _, observed = _request_json(session, "GET", f"{base_url}/v1/service/status", timeout=http_timeout_seconds)
        if any(replica.get("status") == "dead" for replica in observed.get("pool", [])):
            after = observed
        if successful_request is not None and after is not None:
            break
    if successful_request is None or after is None:
        raise FaultValidationFailure(
            f"service did not degrade cleanly after replica exit; request_errors={failed_attempts}"
        )
    degradation = validate_pool_degradation(before, after)
    return {
        "name": "replica_exit",
        "passed": True,
        "terminated_pid": replica_pid,
        "physical_gpu_index": gpu_index,
        "failed_attempts": failed_attempts,
        "successful_action": successful_request["action"],
        **degradation,
    }


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--image", type=Path, required=True, help="Image reused for all three camera inputs.")
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--state", type=structured_validator.parse_state_json, default=[0.0] * 14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--http-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--task-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.05)
    parser.add_argument("--service-pid", type=int)
    parser.add_argument("--kill-replica-gpu-index")
    parser.add_argument("--output", type=Path, default=Path("work_dirs/vla_service_fault_validation.json"))
    args = parser.parse_args()
    if (args.service_pid is None) != (args.kill_replica_gpu_index is None):
        parser.error("--service-pid and --kill-replica-gpu-index must be provided together")
    if args.http_timeout_seconds <= 0 or args.task_timeout_seconds <= 0 or args.poll_interval_seconds <= 0:
        parser.error("timeout and polling values must be positive")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    return args


def main() -> int:
    args = parse_args()
    encoded_image = _image_base64(args.image)
    payload = {
        "task": "vla_action",
        "instruction": args.instruction,
        "state": args.state,
        "camera_high": encoded_image,
        "camera_left_wrist": encoded_image,
        "camera_right_wrist": encoded_image,
        "seed": args.seed,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "validation": "lingbot_vla_v2_structured_service_faults",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.base_url.rstrip("/"),
        "checks": [],
        "passed": False,
    }
    try:
        with _new_session() as session:
            structured_validator.inspect_service(report["target"], timeout_seconds=args.http_timeout_seconds)
            report["checks"].extend(
                validate_request_faults(
                    session,
                    report["target"],
                    payload,
                    http_timeout_seconds=args.http_timeout_seconds,
                    task_timeout_seconds=args.task_timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                )
            )
            if args.service_pid is not None:
                report["checks"].append(
                    validate_replica_exit(
                        session,
                        report["target"],
                        payload,
                        service_pid=args.service_pid,
                        gpu_index=args.kill_replica_gpu_index,
                        http_timeout_seconds=args.http_timeout_seconds,
                        task_timeout_seconds=args.task_timeout_seconds,
                        poll_interval_seconds=args.poll_interval_seconds,
                    )
                )
        report["passed"] = all(check.get("passed") is True for check in report["checks"])
    except (
        FaultValidationFailure,
        structured_validator.ValidationFailure,
        requests.RequestException,
        OSError,
    ) as error:
        report["error"] = str(error)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
