"""Validate a real LingBot-VLA v2 native structured API service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import requests

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_REQUIRED_PARAMETERS = frozenset(
    {
        "instruction",
        "state",
        "camera_high",
        "camera_left_wrist",
        "camera_right_wrist",
    }
)


class ValidationFailure(RuntimeError):
    """Raised when the target violates the VLA structured API contract."""


@dataclass(frozen=True)
class RequestConfig:
    """Immutable settings shared by validation workers."""

    base_url: str
    payload: dict[str, Any]
    http_timeout_seconds: float
    task_timeout_seconds: float
    poll_interval_seconds: float
    expected_horizon: int
    expected_action_dim: int


def parse_state_json(value: str) -> list[float]:
    """Parse and validate a finite 14-dimensional RobotWin state."""
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("state must be valid JSON") from error
    if not isinstance(raw, list) or len(raw) != 14:
        raise argparse.ArgumentTypeError("state must be a JSON array containing exactly 14 values")
    state: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item)):
            raise argparse.ArgumentTypeError("state values must be finite numbers")
        state.append(float(item))
    return state


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: Sequence[float]) -> dict[str, float | int] | None:
    """Summarize a possibly empty sample in seconds."""
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def validate_service_metadata(metadata: Any) -> None:
    """Validate that the target exposes the native VLA structured contract."""
    if not isinstance(metadata, dict):
        raise ValidationFailure("service metadata must be a JSON object")
    if metadata.get("declared_pipeline_contract") is not True:
        raise ValidationFailure("service does not expose a declared pipeline contract")
    if "vla_action" not in metadata.get("supported_tasks", []):
        raise ValidationFailure("service metadata does not declare the vla_action task")
    if "structured" not in metadata.get("supported_media_types", []):
        raise ValidationFailure("service metadata does not declare structured output")
    task_contract = metadata.get("task_contracts", {}).get("vla_action")
    if not isinstance(task_contract, dict) or task_contract.get("media_type") != "structured":
        raise ValidationFailure("vla_action does not have a structured task contract")
    parameters = task_contract.get("parameters")
    if not isinstance(parameters, dict):
        raise ValidationFailure("vla_action parameters are missing from service metadata")
    missing = sorted(_REQUIRED_PARAMETERS.difference(parameters))
    if missing:
        raise ValidationFailure(f"vla_action metadata is missing parameters: {', '.join(missing)}")
    invalid = sorted(name for name in _REQUIRED_PARAMETERS if not isinstance(parameters[name], dict))
    if invalid:
        raise ValidationFailure(f"vla_action parameter contracts are invalid: {', '.join(invalid)}")
    not_required = sorted(name for name in _REQUIRED_PARAMETERS if parameters[name].get("required") is not True)
    if not_required:
        raise ValidationFailure(f"vla_action parameters are not required: {', '.join(not_required)}")


def validate_action_result(result: Any, *, expected_horizon: int, expected_action_dim: int) -> dict[str, Any]:
    """Validate and summarize one canonical normalized action chunk."""
    if expected_horizon < 1 or expected_action_dim < 1:
        raise ValueError("expected action dimensions must be positive")
    if not isinstance(result, dict):
        raise ValidationFailure("completed task result must be a JSON object")
    actions = result.get("canonical_normalized_actions")
    if not isinstance(actions, list) or len(actions) != expected_horizon:
        observed = len(actions) if isinstance(actions, list) else type(actions).__name__
        raise ValidationFailure(f"expected action horizon {expected_horizon}, observed {observed}")
    if result.get("horizon") != expected_horizon:
        raise ValidationFailure(f"result horizon field is not {expected_horizon}")
    if result.get("action_dim") != expected_action_dim:
        raise ValidationFailure(f"result action_dim field is not {expected_action_dim}")

    flat: list[float] = []
    digest = hashlib.sha256()
    for row_index, row in enumerate(actions):
        if not isinstance(row, list) or len(row) != expected_action_dim:
            observed = len(row) if isinstance(row, list) else type(row).__name__
            raise ValidationFailure(f"action row {row_index} has dimension {observed}, expected {expected_action_dim}")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValidationFailure("action chunk contains a non-finite or non-numeric value")
            number = float(value)
            flat.append(number)
            digest.update(struct.pack("<d", number))

    policy_verified = result.get("policy_verified")
    verification_status = result.get("verification_status")
    if not isinstance(policy_verified, bool):
        raise ValidationFailure("result policy_verified field must be boolean")
    if not isinstance(verification_status, str) or not verification_status:
        raise ValidationFailure("result verification_status field must be a non-empty string")
    checkpoint_variant = result.get("checkpoint_variant")
    if not isinstance(checkpoint_variant, str) or not checkpoint_variant:
        raise ValidationFailure("result checkpoint_variant field must be a non-empty string")

    return {
        "shape": [expected_horizon, expected_action_dim],
        "value_count": len(flat),
        "minimum": min(flat),
        "maximum": max(flat),
        "mean": statistics.fmean(flat),
        "l2_norm": math.sqrt(sum(value * value for value in flat)),
        "sha256_float64_le": digest.hexdigest(),
        "checkpoint_variant": checkpoint_variant,
        "policy_verified": policy_verified,
        "verification_status": verification_status,
    }


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.request(method, url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = response.text[:1000]
        raise ValidationFailure(f"{method} {url} returned HTTP {response.status_code}: {body}") from error
    try:
        body = response.json()
    except ValueError as error:
        raise ValidationFailure(f"{method} {url} did not return JSON") from error
    if not isinstance(body, dict):
        raise ValidationFailure(f"{method} {url} did not return a JSON object")
    return body


def _new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def inspect_service(base_url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Read and validate native service readiness and metadata."""
    with _new_session() as session:
        ready = _request_json(session, "GET", f"{base_url}/v1/service/ready", timeout=timeout_seconds)
        if ready.get("ready") is not True:
            raise ValidationFailure("service readiness endpoint reports not ready")
        metadata = _request_json(session, "GET", f"{base_url}/v1/service/metadata", timeout=timeout_seconds)
        validate_service_metadata(metadata)
        status = _request_json(session, "GET", f"{base_url}/v1/service/status", timeout=timeout_seconds)
        metrics = _request_json(session, "GET", f"{base_url}/v1/service/metrics/json", timeout=timeout_seconds)
    return {"ready": ready, "metadata": metadata, "status": status, "metrics": metrics}


def execute_request(
    session: requests.Session,
    config: RequestConfig,
    *,
    request_index: int,
    worker_index: int,
    run_started_at: float,
) -> dict[str, Any]:
    """Submit, poll, validate, and summarize one real structured request."""
    record: dict[str, Any] = {
        "request_index": request_index,
        "worker_index": worker_index,
        "start_offset_seconds": time.perf_counter() - run_started_at,
    }
    request_started_at = time.perf_counter()
    try:
        submit_started_at = time.perf_counter()
        created = _request_json(
            session,
            "POST",
            f"{config.base_url}/v1/tasks/structured",
            timeout=config.http_timeout_seconds,
            payload=config.payload,
        )
        accepted_at = time.perf_counter()
        record["submit_seconds"] = accepted_at - submit_started_at
        task_id = created.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValidationFailure("structured task creation response has no task_id")
        record["task_id"] = task_id

        deadline = accepted_at + config.task_timeout_seconds
        transitions: list[str] = []
        poll_count = 0
        while True:
            if time.perf_counter() >= deadline:
                raise ValidationFailure(f"task {task_id} exceeded {config.task_timeout_seconds:g}s timeout")
            status = _request_json(
                session,
                "GET",
                f"{config.base_url}/v1/tasks/{task_id}/status",
                timeout=config.http_timeout_seconds,
            )
            poll_count += 1
            task_status = status.get("status") or status.get("task_status")
            if not isinstance(task_status, str):
                raise ValidationFailure(f"task {task_id} status response has no status")
            if not transitions or transitions[-1] != task_status:
                transitions.append(task_status)
            if task_status in _TERMINAL_STATUSES:
                break
            time.sleep(config.poll_interval_seconds)

        completed_at = time.perf_counter()
        record.update(
            end_to_end_seconds=completed_at - request_started_at,
            accepted_to_terminal_seconds=completed_at - accepted_at,
            poll_count=poll_count,
            status_transitions=transitions,
            terminal_status=task_status,
        )
        inference_time = status.get("inference_time_s")
        if inference_time is not None:
            if isinstance(inference_time, bool) or not isinstance(inference_time, int | float):
                raise ValidationFailure("inference_time_s must be numeric or null")
            inference_time = float(inference_time)
            if not math.isfinite(inference_time) or inference_time < 0:
                raise ValidationFailure("inference_time_s must be finite and non-negative")
        record["inference_time_seconds"] = inference_time
        peak_memory = status.get("peak_memory_mb")
        if peak_memory is not None:
            if isinstance(peak_memory, bool) or not isinstance(peak_memory, int | float):
                raise ValidationFailure("peak_memory_mb must be numeric or null")
            peak_memory = float(peak_memory)
            if not math.isfinite(peak_memory) or peak_memory < 0:
                raise ValidationFailure("peak_memory_mb must be finite and non-negative")
        record["peak_memory_mb"] = peak_memory

        if task_status != "completed":
            raise ValidationFailure(f"task {task_id} reached terminal status {task_status}: {status.get('error')}")
        record["action"] = validate_action_result(
            status.get("result"),
            expected_horizon=config.expected_horizon,
            expected_action_dim=config.expected_action_dim,
        )
        record["outcome"] = "succeeded"
    except (requests.RequestException, ValidationFailure, ValueError) as error:
        record["outcome"] = "failed"
        record["error"] = str(error)
        record.setdefault("end_to_end_seconds", time.perf_counter() - request_started_at)
    return record


class RunAccumulator:
    """Collect aggregate measurements while bounding retained request records."""

    def __init__(self, max_records: int) -> None:
        self._lock = threading.Lock()
        self.max_records = max_records
        self.total = 0
        self.succeeded = 0
        self.failed = 0
        self.task_ids: set[str] = set()
        self.duplicate_task_ids: set[str] = set()
        self.end_to_end: list[float] = []
        self.submit: list[float] = []
        self.accepted_to_terminal: list[float] = []
        self.inference: list[float] = []
        self.peak_memory: list[float] = []
        self.poll_counts: list[float] = []
        self.terminal_statuses: Counter[str] = Counter()
        self.policy_statuses: Counter[str] = Counter()
        self.failures: deque[dict[str, Any]] = deque(maxlen=max_records)
        self.first_successes: list[dict[str, Any]] = []
        self.recent_successes: deque[dict[str, Any]] = deque(maxlen=max_records // 2)

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.total += 1
            task_id = record.get("task_id")
            if isinstance(task_id, str):
                if task_id in self.task_ids:
                    self.duplicate_task_ids.add(task_id)
                self.task_ids.add(task_id)
            terminal_status = record.get("terminal_status")
            if isinstance(terminal_status, str):
                self.terminal_statuses[terminal_status] += 1
            if record["outcome"] == "failed":
                self.failed += 1
                self.failures.append(record)
                return

            self.succeeded += 1
            self.end_to_end.append(float(record["end_to_end_seconds"]))
            self.submit.append(float(record["submit_seconds"]))
            self.accepted_to_terminal.append(float(record["accepted_to_terminal_seconds"]))
            self.poll_counts.append(float(record["poll_count"]))
            if record.get("inference_time_seconds") is not None:
                self.inference.append(float(record["inference_time_seconds"]))
            if record.get("peak_memory_mb") is not None:
                self.peak_memory.append(float(record["peak_memory_mb"]))
            self.policy_statuses[str(record["action"]["verification_status"])] += 1
            first_capacity = self.max_records - self.recent_successes.maxlen
            if len(self.first_successes) < first_capacity:
                self.first_successes.append(record)
            else:
                self.recent_successes.append(record)

    def report(self, elapsed_seconds: float) -> dict[str, Any]:
        retained_successes = self.first_successes + list(self.recent_successes)
        retained_successes.sort(key=lambda record: int(record["request_index"]))
        failures = sorted(self.failures, key=lambda record: int(record["request_index"]))
        return {
            "requests": {
                "total": self.total,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "success_rate": self.succeeded / self.total if self.total else 0.0,
                "unique_task_ids": len(self.task_ids),
                "duplicate_task_ids": sorted(self.duplicate_task_ids),
                "terminal_statuses": dict(sorted(self.terminal_statuses.items())),
                "policy_statuses": dict(sorted(self.policy_statuses.items())),
            },
            "elapsed_seconds": elapsed_seconds,
            "throughput_requests_per_second": self.succeeded / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "latency_seconds": {
                "end_to_end": summarize(self.end_to_end),
                "submission": summarize(self.submit),
                "accepted_to_terminal": summarize(self.accepted_to_terminal),
                "target_inference": summarize(self.inference),
            },
            "poll_count": summarize(self.poll_counts),
            "peak_memory_mb": summarize(self.peak_memory),
            "retained_records": {
                "limit_per_outcome": self.max_records,
                "successful": retained_successes,
                "failed": failures,
            },
        }


def run_workload(
    config: RequestConfig,
    *,
    request_count: int | None,
    duration_seconds: float | None,
    concurrency: int,
    max_records: int,
) -> dict[str, Any]:
    """Run a closed-loop fixed-count or duration workload."""
    accumulator = RunAccumulator(max_records)
    counter = 0
    counter_lock = threading.Lock()
    run_started_at = time.perf_counter()
    stop_claiming_at = None if duration_seconds is None else run_started_at + duration_seconds
    workers = concurrency if request_count is None else min(concurrency, request_count)
    barrier = threading.Barrier(workers)

    def claim_request() -> int | None:
        nonlocal counter
        with counter_lock:
            if request_count is not None and counter >= request_count:
                return None
            if stop_claiming_at is not None and time.perf_counter() >= stop_claiming_at:
                return None
            index = counter
            counter += 1
            return index

    def worker(worker_index: int) -> None:
        with _new_session() as session:
            barrier.wait()
            while (request_index := claim_request()) is not None:
                accumulator.add(
                    execute_request(
                        session,
                        config,
                        request_index=request_index,
                        worker_index=worker_index,
                        run_started_at=run_started_at,
                    )
                )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vla-structured-validator") as executor:
        futures = [executor.submit(worker, worker_index) for worker_index in range(workers)]
        for future in futures:
            future.result()
    return accumulator.report(time.perf_counter() - run_started_at)


def _encode_image(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"camera image does not exist: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _resolve_camera_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    fallback = args.image
    paths = tuple(path or fallback for path in (args.camera_high, args.camera_left_wrist, args.camera_right_wrist))
    if any(path is None for path in paths):
        raise ValueError("provide --image or all three --camera-* paths")
    return paths  # type: ignore[return-value]


def _package_version() -> str:
    try:
        return version("telefuser")
    except PackageNotFoundError:
        return "source"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for section in ("tasks",):
        before_section = before.get(section, {})
        after_section = after.get(section, {})
        if isinstance(before_section, dict) and isinstance(after_section, dict):
            delta[section] = {
                key: after_section[key] - before_section.get(key, 0)
                for key in after_section
                if isinstance(after_section[key], int | float) and not isinstance(after_section[key], bool)
            }
    return delta


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the service and return a reproducible JSON report."""
    if args.concurrency < 1 or args.warmup < 0 or args.max_records < 2:
        raise ValueError("concurrency must be positive, warmup non-negative, and max-records at least 2")
    if args.requests is not None and args.requests < 1:
        raise ValueError("requests must be positive")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive")
    if args.poll_interval_seconds <= 0 or args.http_timeout_seconds <= 0 or args.task_timeout_seconds <= 0:
        raise ValueError("poll interval and HTTP/task timeouts must be positive")
    if args.expected_horizon < 1 or args.expected_action_dim < 1:
        raise ValueError("expected action dimensions must be positive")
    base_url = args.base_url.rstrip("/")
    camera_high, camera_left, camera_right = _resolve_camera_paths(args)
    payload = {
        "task": "vla_action",
        "instruction": args.instruction,
        "state": args.state_json,
        "camera_high": _encode_image(camera_high),
        "camera_left_wrist": _encode_image(camera_left),
        "camera_right_wrist": _encode_image(camera_right),
        "seed": args.seed,
    }
    config = RequestConfig(
        base_url=base_url,
        payload=payload,
        http_timeout_seconds=args.http_timeout_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        expected_horizon=args.expected_horizon,
        expected_action_dim=args.expected_action_dim,
    )
    before = inspect_service(base_url, timeout_seconds=args.http_timeout_seconds)
    warmup_records: list[dict[str, Any]] = []
    warmup_started_at = time.perf_counter()
    with _new_session() as session:
        for index in range(args.warmup):
            warmup_records.append(
                execute_request(
                    session,
                    config,
                    request_index=index,
                    worker_index=0,
                    run_started_at=warmup_started_at,
                )
            )
    if any(record["outcome"] != "succeeded" for record in warmup_records):
        raise ValidationFailure("at least one warmup request failed")

    request_count = args.requests
    if request_count is None and args.duration_seconds is None:
        request_count = 1
    workload = run_workload(
        config,
        request_count=request_count,
        duration_seconds=args.duration_seconds,
        concurrency=args.concurrency,
        max_records=args.max_records,
    )
    after = inspect_service(base_url, timeout_seconds=args.http_timeout_seconds)
    requests_report = workload["requests"]
    checks = {
        "service_ready_before": before["ready"].get("ready") is True,
        "service_ready_after": after["ready"].get("ready") is True,
        "warmup_succeeded": all(record["outcome"] == "succeeded" for record in warmup_records),
        "all_measured_requests_succeeded": requests_report["failed"] == 0 and requests_report["total"] > 0,
        "task_ids_unique": not requests_report["duplicate_task_ids"],
        "queue_drained": (
            after["metrics"].get("queue", {}).get("pending") == 0
            and after["metrics"].get("queue", {}).get("processing") == 0
        ),
    }
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "validation": "lingbot_vla_v2_native_structured_api",
        "passed": all(checks.values()),
        "checks": checks,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "telefuser_version": _package_version(),
            "telefuser_commit": _git_commit(repo_root),
        },
        "target": {
            "base_url": base_url,
            "transport": "HTTP native TeleFuser asynchronous structured task API",
            "metadata": before["metadata"],
            "status_before": before["status"],
            "status_after": after["status"],
            "health_before": before["ready"],
            "health_after": after["ready"],
            "metrics_before": before["metrics"],
            "metrics_after": after["metrics"],
            "metrics_delta": _metric_delta(before["metrics"], after["metrics"]),
        },
        "workload": {
            "mode": "duration" if args.duration_seconds is not None else "fixed_requests",
            "requested_requests": request_count,
            "requested_duration_seconds": args.duration_seconds,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup,
            "instruction": args.instruction,
            "state_dimension": len(args.state_json),
            "seed": args.seed,
            "camera_files": {
                "high": str(camera_high.resolve()),
                "left_wrist": str(camera_left.resolve()),
                "right_wrist": str(camera_right.resolve()),
            },
            "expected_action_shape": [args.expected_horizon, args.expected_action_dim],
            "poll_interval_seconds": args.poll_interval_seconds,
            "task_timeout_seconds": args.task_timeout_seconds,
        },
        "warmup_records": warmup_records,
        "result": workload,
        "interpretation": (
            "This validates service transport, scheduling, and normalized canonical action structure. "
            "It does not establish embodiment-specific robot control semantics."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    camera_group = parser.add_argument_group("camera inputs")
    camera_group.add_argument("--image", type=Path, help="Fallback image reused for camera inputs not set explicitly.")
    camera_group.add_argument("--camera-high", type=Path)
    camera_group.add_argument("--camera-left-wrist", type=Path)
    camera_group.add_argument("--camera-right-wrist", type=Path)
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument(
        "--state-json", type=parse_state_json, default=parse_state_json("[0,0,0,0,0,0,0,0,0,0,0,0,0,0]")
    )
    parser.add_argument("--seed", type=int, default=7)
    workload_group = parser.add_mutually_exclusive_group()
    workload_group.add_argument("--requests", type=int)
    workload_group.add_argument("--duration-seconds", type=float)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--task-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--expected-horizon", type=int, default=50)
    parser.add_argument("--expected-action-dim", type=int, default=55)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report: dict[str, Any]
    exit_code = 0
    try:
        report = run_validation(args)
        if not report["passed"]:
            exit_code = 1
    except Exception as error:
        report = {
            "schema_version": 1,
            "validation": "lingbot_vla_v2_native_structured_api",
            "passed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fatal_error": str(error),
        }
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "passed": report["passed"],
        "checks": report.get("checks"),
        "requests": report.get("result", {}).get("requests"),
        "latency_seconds": report.get("result", {}).get("latency_seconds"),
        "fatal_error": report.get("fatal_error"),
        "artifact": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
