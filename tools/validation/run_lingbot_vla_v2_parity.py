"""Compare LingBot-VLA v2 upstream and TeleFuser parity artifacts.

The artifact contract is intentionally file-based so the upstream checkout and
TeleFuser model do not need to live in one Python process. Capture both sides at
the fixed upstream commit and save ``.npz`` files with matching array keys.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

UPSTREAM_REPOSITORY = "https://github.com/Robbyant/lingbot-vla-v2"
UPSTREAM_COMMIT = "be27333c9b5f2663b0ec33f069dd7dfd67fa32b5"
PREPROCESSING_KEYS = (
    "images",
    "img_masks",
    "image_grid_thw",
    "lang_tokens",
    "lang_masks",
    "state",
)
FINAL_ACTION_KEYS = ("actions", "canonical_normalized_actions")


@dataclass(frozen=True)
class ArrayParity:
    layer: str
    key: str
    shape: tuple[int, ...]
    max_abs: float
    mean_abs: float
    rtol: float
    atol: float
    passed: bool


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _velocity_keys(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> list[str]:
    prefixes = ("velocity_step_", "v_t_step_")
    keys = sorted(key for key in left if key in right and key.startswith(prefixes))
    if not keys and "velocity" in left and "velocity" in right:
        keys = ["velocity"]
    return keys


def _first_present(keys: Iterable[str], left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> str | None:
    for key in keys:
        if key in left and key in right:
            return key
    return None


def _compare_array(
    layer: str,
    key: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> ArrayParity:
    if expected.shape != actual.shape:
        return ArrayParity(layer, key, tuple(actual.shape), float("inf"), float("inf"), rtol, atol, False)
    diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    passed = bool(np.allclose(expected, actual, rtol=rtol, atol=atol))
    return ArrayParity(layer, key, tuple(actual.shape), max_abs, mean_abs, rtol, atol, passed)


def compare_artifacts(reference: Path, candidate: Path, *, rtol: float, atol: float) -> dict[str, object]:
    expected = _load_npz(reference)
    actual = _load_npz(candidate)
    results: list[ArrayParity] = []

    missing_reference = sorted(set(actual) - set(expected))
    missing_candidate = sorted(set(expected) - set(actual))

    for key in PREPROCESSING_KEYS:
        if key in expected and key in actual:
            results.append(_compare_array("preprocessing", key, expected[key], actual[key], rtol=0.0, atol=0.0))

    for key in _velocity_keys(expected, actual):
        results.append(_compare_array("velocity", key, expected[key], actual[key], rtol=rtol, atol=atol))

    action_key = _first_present(FINAL_ACTION_KEYS, expected, actual)
    if action_key is not None:
        results.append(
            _compare_array("action", action_key, expected[action_key], actual[action_key], rtol=rtol, atol=atol)
        )

    if not any(item.layer == "preprocessing" for item in results):
        raise ValueError("No shared preprocessing keys were found in the parity artifacts")
    if not any(item.layer == "velocity" for item in results):
        raise ValueError("No shared velocity keys were found; capture intermediate flow-matching velocity tensors")
    if not any(item.layer == "action" for item in results):
        raise ValueError("No shared final action key was found; expected actions or canonical_normalized_actions")

    return {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "reference": str(reference),
        "candidate": str(candidate),
        "passed": all(item.passed for item in results) and not missing_candidate,
        "missing_reference_keys": missing_reference,
        "missing_candidate_keys": missing_candidate,
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Upstream .npz artifact captured at the pinned commit",
    )
    parser.add_argument("--candidate", type=Path, required=True, help="TeleFuser .npz artifact from the same inputs")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    report = compare_artifacts(args.reference, args.candidate, rtol=args.rtol, atol=args.atol)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
