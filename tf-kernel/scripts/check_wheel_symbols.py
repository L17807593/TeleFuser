"""Validate architecture contents and exported symbols in tf-kernel wheels."""

from __future__ import annotations

import argparse
import ast
import glob
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

ARCHITECTURES = ("sm80", "sm90", "sm100")
EXPECTED_ARCHITECTURES = {
    "SM80": {"sm80"},
    "SM90": {"sm90"},
    "SM100": {"sm100"},
    "ALL": set(ARCHITECTURES),
}
FP4_PATTERNS = (
    re.compile("sageattn3_", re.IGNORECASE),
    re.compile("scaled_fp4_", re.IGNORECASE),
    re.compile("cutlass_scaled_fp4", re.IGNORECASE),
    re.compile("nvfp4", re.IGNORECASE),
)
def _run_nm(shared_object: Path, flag: str) -> list[str]:
    result = subprocess.run(
        ["nm", "-D", flag, str(shared_object)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nm failed for {shared_object}: {result.stderr.strip()}")
    symbols: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            symbols.append(parts[-1])
    return symbols


def _is_fp4_symbol(symbol: str) -> bool:
    return any(pattern.search(symbol) for pattern in FP4_PATTERNS)


def _read_build_target(archive: zipfile.ZipFile) -> str:
    try:
        source = archive.read("tf_kernel/_build_info.py").decode("utf-8")
    except KeyError as error:
        raise ValueError("wheel does not contain tf_kernel/_build_info.py") from error

    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "TARGET_SM":
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("wheel build info does not define TARGET_SM")


def check_wheel(wheel: Path) -> list[str]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(wheel) as archive:
            try:
                target_sm = _read_build_target(archive)
            except ValueError as error:
                return [f"{wheel}: {error}"]

            expected = EXPECTED_ARCHITECTURES.get(target_sm)
            if expected is None:
                return [f"{wheel}: unsupported TARGET_SM={target_sm!r}"]

            extension_members = {
                arch: [
                    name
                    for name in archive.namelist()
                    if name.startswith(f"tf_kernel/{arch}/common_ops") and name.endswith(".so")
                ]
                for arch in ARCHITECTURES
            }
            members = [name for names in extension_members.values() for name in names]
            archive.extractall(tmpdir, members)

        actual = {arch for arch, members in extension_members.items() if members}
        if actual != expected:
            errors.append(
                f"{wheel}: TARGET_SM={target_sm} expects {sorted(expected)}, found {sorted(actual)}"
            )

        for arch, members in extension_members.items():
            if len(members) > 1:
                errors.append(f"{wheel}: multiple common_ops extensions found for {arch}")
                continue
            if not members:
                continue

            shared_object = Path(tmpdir, members[0])
            try:
                defined = _run_nm(shared_object, "--defined-only")
            except RuntimeError as error:
                errors.append(str(error))
                continue

            fp4_symbols = [symbol for symbol in defined if _is_fp4_symbol(symbol)]
            if arch != "sm100" and fp4_symbols:
                errors.append(f"{wheel}:{arch} unexpectedly exports FP4 symbols")

    return errors


def _resolve_wheels(patterns: list[str]) -> list[Path]:
    resolved = {
        Path(match)
        for pattern in patterns
        for match in glob.glob(str(Path(pattern).expanduser()))
    }
    return sorted(path for path in resolved if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="*", default=["dist/*.whl"])
    args = parser.parse_args()

    wheels = _resolve_wheels(args.wheels)
    if not wheels:
        print("error: no wheel files found")
        return 1

    errors = [error for wheel in wheels for error in check_wheel(wheel)]
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1

    print(f"Validated architecture contents and symbols for {len(wheels)} wheel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
