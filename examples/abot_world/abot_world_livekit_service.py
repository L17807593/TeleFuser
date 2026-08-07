"""ABot-World LiveKit pipeline file for ``telefuser stream-serve``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_IMAGE_PATH = (
    _PROJECT_ROOT.parent / "ABot-World" / "web_client" / "datasets" / "images" / "84b90ad568b693d2.png"
)
_LOADER_PATH = Path(__file__).with_name("_loader.py")
_LOADER_SPEC = importlib.util.spec_from_file_location("abot_world_example_loader", _LOADER_PATH)
if _LOADER_SPEC is None or _LOADER_SPEC.loader is None:
    raise RuntimeError(f"Could not load ABot example loader: {_LOADER_PATH}")
_LOADER = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_LOADER)
DEFAULT_PROMPT = _LOADER.DEFAULT_PROMPT
get_pipeline = _LOADER.get_pipeline


def get_service(gpu_num: int = 1) -> ABotWorldLiveKitService:
    """Load one ABot model copy for the shared TeleFuser LiveKit worker."""
    if gpu_num != 1:
        raise ValueError("ABot-World-0-5B-LF currently supports exactly one GPU")
    pipeline = get_pipeline(pipeline_class=ABotWorldInteractivePipeline)
    return ABotWorldLiveKitService(
        pipeline,
        default_fps=12,
        default_session_config={
            "image_path": str(_DEFAULT_IMAGE_PATH),
            "prompt": DEFAULT_PROMPT,
            "fps": 12,
            "control_latent_frames": 3,
            "seed": 42,
        },
    )
