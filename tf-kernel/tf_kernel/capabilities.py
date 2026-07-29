"""CUDA architecture and kernel capability mappings."""

from typing import Literal

ArchitectureFamily = Literal["sm80", "sm90", "sm100"]
SageAttentionBackend = Literal[
    "cuda_fp16",
    "triton_fp16",
    "cuda_fp8_thread",
    "cuda_fp8_sm90",
    "cuda_fp8_warp",
]


class UnsupportedArchitectureError(RuntimeError):
    """Raised when tf-kernel does not support a CUDA architecture."""


def capability_label(major: int, minor: int) -> str:
    """Return a conventional CUDA compute-capability label."""
    return f"sm{major}{minor}"


def architecture_family(major: int, minor: int) -> ArchitectureFamily:
    """Map a CUDA compute capability to a packaged extension family."""
    label = capability_label(major, minor)
    if major == 8:
        return "sm80"
    if major == 9 and minor == 0:
        return "sm90"
    if major >= 10:
        return "sm100"
    raise UnsupportedArchitectureError(
        f"Unsupported CUDA architecture {label}; tf-kernel requires SM80 or newer"
    )


def build_target_for_family(family: ArchitectureFamily) -> str:
    """Return the CMake/Make target name for a packaged architecture family."""
    return family.upper()


def sage_attention_backend(major: int, minor: int) -> SageAttentionBackend:
    """Return the validated SageAttention v2 backend for a capability."""
    capability = (major, minor)
    backends: dict[tuple[int, int], SageAttentionBackend] = {
        (8, 0): "cuda_fp16",
        (8, 6): "triton_fp16",
        (8, 9): "cuda_fp8_thread",
        (9, 0): "cuda_fp8_sm90",
        (12, 0): "cuda_fp8_warp",
        (12, 1): "cuda_fp8_warp",
    }
    try:
        return backends[capability]
    except KeyError as error:
        label = capability_label(major, minor)
        raise UnsupportedArchitectureError(
            f"SageAttention v2 has no validated backend for {label}"
        ) from error
