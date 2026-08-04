# SPDX-License-Identifier: Apache-2.0
"""Freeze the official MiniMax H3 requests and their remote input bytes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tools.validation.minimax_h3_validation_common import sha256 as _sha256

_CASES = ("t2va", "fl2va", "ref2va")
_MAX_DOWNLOAD_BYTES = 2 * 1024**3


def _extract_request(script: Path) -> dict[str, object]:
    text = script.read_text(encoding="utf-8")
    match = re.search(r"<<'JSON'\n(?P<payload>.*?)\nJSON", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"could not find the JSON heredoc in {script}")
    payload = json.loads(match.group("payload"))
    if not isinstance(payload, dict):
        raise ValueError(f"request in {script} is not a JSON object")
    return payload


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "TeleFuser-MiniMax-H3-parity/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > _MAX_DOWNLOAD_BYTES:
            raise ValueError(f"remote fixture exceeds the {_MAX_DOWNLOAD_BYTES}-byte limit: {url}")
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            copied = 0
            while chunk := response.read(1024 * 1024):
                copied += len(chunk)
                if copied > _MAX_DOWNLOAD_BYTES:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"remote fixture exceeds the {_MAX_DOWNLOAD_BYTES}-byte limit: {url}")
                output.write(chunk)
    temporary.replace(destination)


def _fixture_name(case: str, index: int, url: str) -> str:
    suffix = Path(unquote(urlsplit(url).path)).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".bin"
    return f"{case}-{index:02d}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument("--output-root", default="work_dirs/minimax_h3_parity")
    parser.add_argument("--refresh", action="store_true", help="Download existing remote fixtures again")
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    output_root = Path(args.output_root).resolve()
    fixture_root = output_root / "fixtures"
    request_root = output_root / "requests"
    fixture_root.mkdir(parents=True, exist_ok=True)
    request_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for case in _CASES:
        script = model_root / "scripts" / "readme" / f"reproducible-768p-{case}-request.sh"
        request = _extract_request(script)
        conditions = request.get("conditions", [])
        if not isinstance(conditions, list):
            raise ValueError(f"conditions in {script} must be a list")
        localized: list[dict[str, object]] = []
        for index, raw_condition in enumerate(conditions):
            if not isinstance(raw_condition, dict):
                raise ValueError(f"condition {index} in {script} must be an object")
            condition = dict(raw_condition)
            uri = condition.get("uri")
            if not isinstance(uri, str):
                raise ValueError(f"condition {index} in {script} has no string URI")
            if urlsplit(uri).scheme not in {"http", "https"}:
                raise ValueError(f"official condition URI is not HTTP(S): {uri}")
            destination = fixture_root / _fixture_name(case, index, uri)
            if args.refresh or not destination.is_file():
                _download(uri, destination)
            condition["uri"] = str(destination)
            localized.append(condition)
            records.append(
                {
                    "case": case,
                    "condition_index": index,
                    "original_uri": uri,
                    "local_path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        request["conditions"] = localized
        request_path = request_root / f"reproducible-768p-{case}.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        records.append(
            {
                "case": case,
                "source_script": str(script),
                "source_script_sha256": _sha256(script),
                "localized_request": str(request_path),
                "localized_request_sha256": _sha256(request_path),
            }
        )

    provenance = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_root": str(model_root),
        "records": records,
    }
    provenance_path = fixture_root / "provenance.json"
    with tempfile.NamedTemporaryFile("w", dir=fixture_root, suffix=".json", encoding="utf-8", delete=False) as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
        temporary = Path(stream.name)
    shutil.copymode(provenance_path, temporary) if provenance_path.exists() else None
    temporary.replace(provenance_path)
    print(json.dumps({"provenance": str(provenance_path), "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
