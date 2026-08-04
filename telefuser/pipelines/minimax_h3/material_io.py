# SPDX-License-Identifier: Apache-2.0
"""Bounded material localization and probing for MiniMax H3 requests."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

MINIMAX_H3_MAX_MATERIAL_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class MiniMaxH3MaterialFacts:
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    has_audio: bool = False


@contextmanager
def minimax_h3_localize_material(uri: str) -> Iterator[Path]:
    """Yield a local path for a file path, file URI, or bounded HTTP(S) URI."""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else uri).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax H3 material not found: {path}")
        yield path
        return
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported MiniMax H3 material URI scheme {parsed.scheme!r}")

    suffix = Path(parsed.path).suffix
    temp_dir = Path(tempfile.mkdtemp(prefix="telefuser-minimax-h3-"))
    target = temp_dir / f"material{suffix}"
    try:
        request = urllib.request.Request(uri, headers={"User-Agent": "TeleFuser/1"})
        with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MINIMAX_H3_MAX_MATERIAL_BYTES:
                raise ValueError("MiniMax H3 material exceeds the 2 GiB download limit")
            copied = 0
            while chunk := response.read(1024 * 1024):
                copied += len(chunk)
                if copied > MINIMAX_H3_MAX_MATERIAL_BYTES:
                    raise ValueError("MiniMax H3 material exceeds the 2 GiB download limit")
                output.write(chunk)
        yield target
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _probe_av(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height,sample_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def minimax_h3_probe_material(path: Path, condition_type: str) -> MiniMaxH3MaterialFacts:
    if condition_type == "image":
        from PIL import Image, ImageOps

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            return MiniMaxH3MaterialFacts(width=int(image.width), height=int(image.height))
    if condition_type not in {"audio", "video", "video_audio"}:
        raise ValueError(f"unsupported MiniMax H3 condition type {condition_type!r}")
    payload = _probe_av(path)
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if condition_type in {"video", "video_audio"} and video is None:
        raise ValueError(f"{condition_type} material has no video stream: {path}")
    if condition_type in {"audio", "video_audio"} and audio is None:
        raise ValueError(f"{condition_type} material has no audio stream: {path}")
    duration_value = (payload.get("format") or {}).get("duration")
    if duration_value is None:
        duration_value = (audio or video or {}).get("duration")
    if duration_value is None:
        raise ValueError(f"media duration is unavailable: {path}")
    sample_rate = None if audio is None or audio.get("sample_rate") is None else int(audio["sample_rate"])
    return MiniMaxH3MaterialFacts(
        width=None if video is None else int(video["width"]),
        height=None if video is None else int(video["height"]),
        duration_seconds=float(duration_value),
        sample_rate=sample_rate,
        has_audio=audio is not None,
    )


__all__ = [
    "MINIMAX_H3_MAX_MATERIAL_BYTES",
    "MiniMaxH3MaterialFacts",
    "minimax_h3_localize_material",
    "minimax_h3_probe_material",
]
