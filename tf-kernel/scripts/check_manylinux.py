"""Verify that wheel binaries do not exceed a claimed manylinux GLIBC policy."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

POLICY_MAX_GLIBC = {"manylinux_2_28": (2, 28)}
GLIBC_PATTERN = re.compile(r"\bGLIBC_(\d+)\.(\d+)\b")


def _required_glibc_versions(shared_object: Path) -> set[tuple[int, int]]:
    result = subprocess.run(
        ["readelf", "--version-info", str(shared_object)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"readelf failed for {shared_object}: {result.stderr.strip()}")
    return {(int(major), int(minor)) for major, minor in GLIBC_PATTERN.findall(result.stdout)}


def check_wheel(wheel: Path, policy: str) -> list[str]:
    maximum = POLICY_MAX_GLIBC[policy]
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(wheel) as archive:
            members = [name for name in archive.namelist() if name.endswith(".so")]
            archive.extractall(tmpdir, members)

        if not members:
            errors.append(f"{wheel}: no shared libraries found")
        for member in members:
            shared_object = Path(tmpdir, member)
            too_new = sorted(version for version in _required_glibc_versions(shared_object) if version > maximum)
            if too_new:
                versions = ", ".join(f"GLIBC_{major}.{minor}" for major, minor in too_new)
                errors.append(f"{wheel}:{member} exceeds {policy}: {versions}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICY_MAX_GLIBC), required=True)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    errors = [error for wheel in args.wheels for error in check_wheel(wheel, args.policy)]
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1

    print(f"Validated {len(args.wheels)} wheel(s) against {args.policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
