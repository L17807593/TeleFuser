"""Serve the shared LingBot-style LiveKit browser control UI for ABot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEMO_PATH = _PROJECT_ROOT / "examples" / "stream_server" / "livekit_bidirectional_demo.py"
_DEMO_SPEC = importlib.util.spec_from_file_location("abot_world_livekit_demo", _DEMO_PATH)
if _DEMO_SPEC is None or _DEMO_SPEC.loader is None:
    raise RuntimeError(f"Could not load shared LiveKit demo: {_DEMO_PATH}")
livekit_bidirectional_demo = importlib.util.module_from_spec(_DEMO_SPEC)
_DEMO_SPEC.loader.exec_module(livekit_bidirectional_demo)

_DEFAULT_IMAGE_PATH = (
    _PROJECT_ROOT.parent / "ABot-World" / "web_client" / "datasets" / "images" / "84b90ad568b693d2.png"
)
DEFAULT_PROMPT = "A smooth first-person exploration through a vivid natural landscape."


def main() -> None:
    """Reuse the shared page with ABot's image and prompt as defaults."""
    livekit_bidirectional_demo.DEFAULT_IMAGE_PATH = str(_DEFAULT_IMAGE_PATH)
    livekit_bidirectional_demo.DEFAULT_PROMPT = DEFAULT_PROMPT
    livekit_bidirectional_demo.main()


if __name__ == "__main__":
    main()
