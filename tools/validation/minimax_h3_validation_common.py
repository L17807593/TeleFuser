# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic hashing helpers for MiniMax H3 validation tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_config_hashes(component_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(component_root)): sha256(path)
        for path in sorted(component_root.rglob("*.json"))
        if path.is_file()
    }


__all__ = ["json_sha256", "model_config_hashes", "sha256"]
