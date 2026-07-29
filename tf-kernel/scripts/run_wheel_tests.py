"""Install a tf-kernel wheel into an isolated target and run its tests."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _resolve_wheel(pattern: str) -> Path:
    matches = sorted(
        (Path(path).resolve() for path in glob.glob(pattern) if Path(path).is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No wheel found for {pattern!r}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one wheel for {pattern!r}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    wheel = _resolve_wheel(args.wheel)
    project_dir = Path(__file__).resolve().parents[1]
    tests_dir = project_dir / "tests"
    config = project_dir / "pyproject.toml"

    with tempfile.TemporaryDirectory(prefix="tf-kernel-wheel-test-") as tmpdir:
        work_dir = Path(tmpdir)
        target = work_dir / "site-packages"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                str(wheel),
                "--target",
                str(target),
                "--no-deps",
                "--force-reinstall",
            ],
            check=True,
        )

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target)
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import tf_kernel; "
                    f"target=Path({str(target)!r}).resolve(); "
                    "loaded=Path(tf_kernel.__file__).resolve(); "
                    "assert loaded.is_relative_to(target), (loaded, target); "
                    "print(f'testing installed package: {loaded}')"
                ),
            ],
            cwd=work_dir,
            env=environment,
            check=True,
        )

        pytest_args = args.pytest_args
        if pytest_args[:1] == ["--"]:
            pytest_args = pytest_args[1:]
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(tests_dir),
            "-c",
            str(config),
            "--import-mode=importlib",
            *pytest_args,
        ]
        return subprocess.run(command, cwd=work_dir, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
