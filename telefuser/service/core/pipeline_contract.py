"""Pipeline service contract for external pipeline entrypoints and capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

VIDEO_TASKS = frozenset({"t2v", "i2v", "fl2v", "vc", "s2v", "vsr"})
IMAGE_TASKS = frozenset({"t2i", "i2i", "edit"})
STRUCTURED_TASKS = frozenset({"vla_action"})
TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


@dataclass(frozen=True)
class PipelineEntrypoints:
    """Named entrypoints required by the service runtime."""

    get_pipeline: str = "get_pipeline"
    run_with_file: str = "run_with_file"


@dataclass(frozen=True)
class ParameterContract:
    """Service-facing contract for a single task parameter."""

    type: str
    required: bool = False
    default: Any = None
    description: str = ""
    enum: tuple[str, ...] = tuple()
    exposed: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ParameterContract:
        """Build a parameter contract from a manifest mapping."""
        default = raw.get("default")
        return cls(
            type=str(raw.get("type") or _infer_parameter_type(default)),
            required=bool(raw.get("required", False)),
            default=default,
            description=str(raw.get("description") or ""),
            enum=_normalize_string_list(raw.get("enum")),
            exposed=bool(raw.get("exposed", True)),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the parameter contract into API-facing metadata."""
        return {
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
            "enum": list(self.enum),
            "exposed": self.exposed,
        }


@dataclass(frozen=True)
class TaskContract:
    """Service-facing task contract for a single task variant."""

    task: str
    media_type: str
    required_inputs: tuple[str, ...] = tuple()
    optional_inputs: tuple[str, ...] = tuple()
    parameters: dict[str, ParameterContract] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, task: str, raw: dict[str, Any]) -> TaskContract:
        """Build a task contract from a manifest mapping."""
        normalized_task = validate_task_name_format(task)
        media_type = str(raw.get("media_type") or infer_media_type_for_task(normalized_task))
        return cls(
            task=normalized_task,
            media_type=media_type,
            required_inputs=_normalize_string_list(raw.get("required_inputs")),
            optional_inputs=_normalize_string_list(raw.get("optional_inputs")),
            parameters=_normalize_parameter_contracts(raw.get("parameters")),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the task contract into API-facing metadata."""
        return {
            "media_type": self.media_type,
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "parameters": {
                name: contract.to_metadata() for name, contract in self.parameters.items() if contract.exposed
            },
        }


@dataclass(frozen=True)
class PipelineContract:
    """Explicit service-facing contract for an external pipeline module."""

    contract_version: str
    pipeline_name: str
    supported_tasks: tuple[str, ...]
    supported_media_types: tuple[str, ...]
    execution_mode: str
    effective_max_concurrent_tasks: int
    entrypoints: PipelineEntrypoints
    task_contracts: dict[str, TaskContract]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, fallback_name: str) -> PipelineContract:
        """Build a contract from a manifest-style mapping."""
        pipeline_name = str(raw.get("pipeline_name") or fallback_name).strip()
        if not pipeline_name:
            raise ValueError("Pipeline contract requires a non-empty pipeline_name")

        supported_tasks_raw = raw.get("supported_tasks")
        supported_tasks = _normalize_string_list(supported_tasks_raw)
        if not supported_tasks:
            raise ValueError("Pipeline contract requires at least one supported task")

        supported_media_types_raw = raw.get("supported_media_types")
        supported_media_types = _normalize_string_list(supported_media_types_raw)
        if not supported_media_types:
            supported_media_types = tuple(_derive_media_types_from_tasks(supported_tasks))

        effective_max_concurrent_tasks = int(raw.get("effective_max_concurrent_tasks") or 1)
        if effective_max_concurrent_tasks < 1:
            raise ValueError("Pipeline contract effective_max_concurrent_tasks must be >= 1")

        entrypoints_raw = raw.get("entrypoints") or {}
        entrypoints = PipelineEntrypoints(
            get_pipeline=str(entrypoints_raw.get("get_pipeline") or "get_pipeline"),
            run_with_file=str(entrypoints_raw.get("run_with_file") or "run_with_file"),
        )

        task_contracts = _build_task_contracts(raw, supported_tasks)

        return cls(
            contract_version=str(raw.get("contract_version") or "v1"),
            pipeline_name=pipeline_name,
            supported_tasks=supported_tasks,
            supported_media_types=supported_media_types,
            execution_mode=str(raw.get("execution_mode") or "serial_single_pipeline"),
            effective_max_concurrent_tasks=effective_max_concurrent_tasks,
            entrypoints=entrypoints,
            task_contracts=task_contracts,
        )

    @classmethod
    def legacy_default(cls, *, ppl_file: str, default_task: str) -> PipelineContract:
        """Build a compatibility contract for legacy service scripts."""
        name = Path(ppl_file).stem
        supported_tasks = (default_task,)
        return cls(
            contract_version="legacy",
            pipeline_name=name,
            supported_tasks=supported_tasks,
            supported_media_types=tuple(_derive_media_types_from_tasks(supported_tasks)),
            execution_mode="serial_single_pipeline",
            effective_max_concurrent_tasks=1,
            entrypoints=PipelineEntrypoints(),
            task_contracts=_default_task_contracts(supported_tasks),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the contract into API-facing metadata."""
        return {
            "contract_version": self.contract_version,
            "pipeline_name": self.pipeline_name,
            "supported_tasks": list(self.supported_tasks),
            "supported_media_types": list(self.supported_media_types),
            "execution_mode": self.execution_mode,
            "effective_max_concurrent_tasks": self.effective_max_concurrent_tasks,
            "entrypoints": {
                "get_pipeline": self.entrypoints.get_pipeline,
                "run_with_file": self.entrypoints.run_with_file,
            },
            "task_contracts": {task: contract.to_metadata() for task, contract in self.task_contracts.items()},
        }

    def get_task_contract(self, task: str) -> TaskContract | None:
        """Get task-level contract for a declared task."""
        return self.task_contracts.get(task)


def load_pipeline_contract(module: ModuleType, *, ppl_file: str, default_task: str) -> tuple[PipelineContract, bool]:
    """Load an explicit contract from the module or synthesize a legacy fallback."""
    for attr_name in ("get_pipeline_contract", "get_pipeline_manifest"):
        factory = getattr(module, attr_name, None)
        if callable(factory):
            raw = factory()
            return _coerce_contract(raw, fallback_name=Path(ppl_file).stem), True

    for attr_name in ("PIPELINE_CONTRACT", "PIPELINE_MANIFEST"):
        if hasattr(module, attr_name):
            raw = getattr(module, attr_name)
            return _coerce_contract(raw, fallback_name=Path(ppl_file).stem), True

    return PipelineContract.legacy_default(ppl_file=ppl_file, default_task=default_task), False


def _coerce_contract(raw: Any, *, fallback_name: str) -> PipelineContract:
    if isinstance(raw, PipelineContract):
        return raw
    if isinstance(raw, dict):
        return PipelineContract.from_mapping(raw, fallback_name=fallback_name)
    raise TypeError("Pipeline contract must be a dict, PipelineContract, or returned from a callable")


def _normalize_string_list(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw)

    normalized = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _normalize_parameter_contracts(raw: Any) -> dict[str, ParameterContract]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("Task contract parameters must be a mapping")

    return {
        str(name).strip(): ParameterContract.from_mapping(dict(value))
        for name, value in raw.items()
        if str(name).strip()
    }


def _infer_parameter_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "string"
    return "string"


def _derive_media_types_from_tasks(tasks: tuple[str, ...]) -> list[str]:
    media_types: list[str] = []
    if any(task in VIDEO_TASKS for task in tasks):
        media_types.append("video")
    if any(task in IMAGE_TASKS for task in tasks):
        media_types.append("image")
    if any(task in STRUCTURED_TASKS for task in tasks):
        media_types.append("structured")
    if not media_types:
        media_types.append("unknown")
    return media_types


def _default_required_inputs(task: str) -> tuple[str, ...]:
    if task in {"i2i", "i2v"}:
        return ("first_image_path",)
    if task == "fl2v":
        return ("first_image_path", "last_image_path")
    if task in {"vc", "vsr"}:
        return ("ref_video_path",)
    return tuple()


def _default_optional_inputs(task: str) -> tuple[str, ...]:
    if task == "i2v":
        return ("last_image_path",)
    return tuple()


def _default_task_contracts(tasks: tuple[str, ...]) -> dict[str, TaskContract]:
    return {
        task: TaskContract(
            task=task,
            media_type=infer_media_type_for_task(task),
            required_inputs=_default_required_inputs(task),
            optional_inputs=_default_optional_inputs(task),
            parameters={},
        )
        for task in tasks
    }


def default_task_contract(task: str) -> TaskContract:
    """Build the default task contract for a task identifier."""
    normalized_task = validate_task_name_format(task)
    return _default_task_contracts((normalized_task,))[normalized_task]


def _build_task_contracts(raw: dict[str, Any], supported_tasks: tuple[str, ...]) -> dict[str, TaskContract]:
    task_contracts = _default_task_contracts(supported_tasks)
    raw_task_contracts = raw.get("task_contracts") or {}
    for task_name, task_raw in raw_task_contracts.items():
        task_contracts[validate_task_name_format(task_name)] = TaskContract.from_mapping(task_name, dict(task_raw))

    for task in supported_tasks:
        task_contracts.setdefault(task, _default_task_contracts((task,))[task])

    return {task: task_contracts[task] for task in supported_tasks}


def validate_task_name_format(task: str) -> str:
    """Validate a task identifier format without enforcing pipeline-specific support."""
    normalized = task.strip().lower()
    if not TASK_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Invalid task type format. Expected a lowercase identifier starting with a letter and containing only "
            "letters, digits, or underscores."
        )
    return normalized


def is_video_task(task: str) -> bool:
    """Whether the task is a known video task."""
    return task in VIDEO_TASKS


def is_image_task(task: str) -> bool:
    """Whether the task is a known image task."""
    return task in IMAGE_TASKS


def infer_media_type_for_task(task: str) -> str:
    """Infer output media type from a task identifier."""
    if is_image_task(task):
        return "image"
    if is_video_task(task):
        return "video"
    if task in STRUCTURED_TASKS:
        return "structured"
    return "video"
