"""Simplified LoRA (Low-Rank Adaptation) loader with support for multiple format patterns."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from telefuser.utils.logging import logger
from telefuser.utils.model_weight import load_state_dict

# Simplified pattern definitions as tuples (up_suffix, down_suffix, mid_suffix)
LORA_PATTERNS = {
    "standard": (".lora_up.weight", ".lora_down.weight", ".lora_mid.weight"),
    "diffusers": ("_lora.up.weight", "_lora.down.weight", None),
    "diffusers_v2": (".lora_B.weight", ".lora_A.weight", None),
    "diffusers_v3": (".lora.up.weight", ".lora.down.weight", None),
    "mochi": (".lora_B", ".lora_A", None),
    "transformers": (
        ".lora_linear_layer.up.weight",
        ".lora_linear_layer.down.weight",
        None,
    ),
    "qwen": (".lora_B.default.weight", ".lora_A.default.weight", None),
}

# Diff patterns for direct addition style LoRA
DIFF_PATTERNS = [
    (".diff", ".weight"),
    (".diff_b", ".bias"),
    (".diff_m", ".modulation"),
]

# Common prefixes to remove from model keys
COMMON_PREFIXES = ["diffusion_model.", "model.", "unet."]


@dataclass(frozen=True)
class LoRATarget:
    """Resolved model tensor that receives one LoRA delta."""

    key: str
    tensor: torch.Tensor


LoRATargetResolver = Callable[[str, Mapping[str, torch.Tensor]], LoRATarget | None]


class _SafetensorMapping(Mapping[str, torch.Tensor]):
    """Mapping facade that reads safetensors lazily while its file is open."""

    def __init__(self, source: object) -> None:
        self.source = source
        self._keys = tuple(source.keys())
        self._key_set = frozenset(self._keys)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._key_set:
            raise KeyError(key)
        return self.source.get_tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def metadata(self) -> dict[str, str]:
        """Return string metadata stored in the open safetensors file."""
        return self.source.metadata() or {}


class LoRALoader:
    """Simplified LoRA loader that applies weights to model weights."""

    def __init__(
        self,
        key_mapping_rules: list[tuple[str, str]] | None = None,
        *,
        target_resolver: LoRATargetResolver | None = None,
        strict: bool = False,
        default_alpha: float | None = None,
        stream_safetensors: bool = False,
        merge_dtype: torch.dtype | None = None,
    ) -> None:
        """
        Args:
            key_mapping_rules: Optional list of (pattern, replacement) regex rules for key mapping.
            target_resolver: Optional resolver for fused parameters or parameter slices.
            strict: Raise for unused keys, missing targets, invalid shapes, and merge failures.
            default_alpha: Alpha used when neither the checkpoint nor apply_lora supplies one.
            stream_safetensors: Read safetensors lazily instead of materializing the whole file.
            merge_dtype: Optional dtype for the low-rank matrix multiplication.
        """
        self.key_mapping_rules = key_mapping_rules or []
        self.target_resolver = target_resolver
        self.strict = strict
        self.default_alpha = default_alpha
        self.stream_safetensors = stream_safetensors
        self.merge_dtype = merge_dtype
        self._compile_rules()

    def _compile_rules(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_rules = [(re.compile(pattern), replacement) for pattern, replacement in self.key_mapping_rules]

    def _apply_key_mapping(self, key: str) -> str:
        """Apply key mapping rules to a key."""
        for pattern, replacement in self.compiled_rules:
            key = pattern.sub(replacement, key)
        return key

    def _detect_lora_format(self, key: str) -> tuple[str, str] | None:
        """Detect LoRA format and return (format_name, up_suffix) if found."""
        for format_name, (up_suffix, down_suffix, _) in LORA_PATTERNS.items():
            if key.endswith(up_suffix):
                return format_name, up_suffix
        return None

    def _extract_base_key(self, key: str, suffix: str) -> str | None:
        """Extract base key by removing the detected suffix."""
        if key.endswith(suffix):
            return key[: -len(suffix)]
        return None

    def _get_model_key(self, base_key: str, suffix_to_add: str = ".weight") -> str | None:
        """Extract the model weight key from LoRA key."""
        # For Qwen models, keep transformer_blocks prefix
        if base_key.startswith("transformer_blocks.") and len(base_key.split(".")) > 1:
            if base_key.split(".")[1].isdigit():
                model_key = base_key + suffix_to_add
            else:
                model_key = self._remove_prefixes(base_key) + suffix_to_add
        else:
            model_key = self._remove_prefixes(base_key) + suffix_to_add

        # Apply key mapping rules if provided
        if self.compiled_rules:
            model_key = self._apply_key_mapping(model_key)

        return model_key

    @staticmethod
    def _remove_prefixes(key: str) -> str:
        """Remove common model prefixes from a key."""
        for prefix in COMMON_PREFIXES:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def extract_lora_alphas(self, lora_weights: Mapping[str, torch.Tensor]) -> dict:
        """Extract LoRA alpha values from the state dict."""
        lora_alphas = {}
        for key in lora_weights.keys():
            if key.endswith(".alpha"):
                base_key = key[:-6]  # Remove .alpha
                lora_alphas[base_key] = lora_weights[key].item()
        return lora_alphas

    def extract_lora_pairs(self, lora_weights: Mapping[str, torch.Tensor]) -> list[dict]:
        """Extract all LoRA pairs from the state dict."""
        lora_alphas = self.extract_lora_alphas(lora_weights)
        lora_pairs = []

        for key in lora_weights.keys():
            # Skip alpha parameters
            if key.endswith(".alpha"):
                continue

            # Detect format
            format_detected = self._detect_lora_format(key)
            if format_detected is None:
                continue

            format_name, up_suffix = format_detected
            up_suffix, down_suffix, mid_suffix = LORA_PATTERNS[format_name]

            # Extract base key
            base_key = self._extract_base_key(key, up_suffix)
            if base_key is None:
                continue

            # Check if down weight exists
            down_key = base_key + down_suffix
            if down_key not in lora_weights:
                continue

            # Check for mid weight
            mid_key = None
            if mid_suffix:
                mid_key = base_key + mid_suffix
                if mid_key not in lora_weights:
                    mid_key = None

            # Get alpha value
            alpha = lora_alphas.get(base_key, None)

            # Get model key
            model_key = self._get_model_key(base_key, ".weight")
            if model_key is None:
                logger.warning(f"Failed to extract model key from LoRA key: {key}")
                continue

            lora_pairs.append(
                {
                    "format": format_name,
                    "model_key": model_key,
                    "base_key": base_key,
                    "up_key": key,
                    "down_key": down_key,
                    "mid_key": mid_key,
                    "alpha": alpha,
                    "alpha_key": base_key + ".alpha" if base_key in lora_alphas else None,
                }
            )

        return lora_pairs

    def extract_lora_diffs(self, lora_weights: Mapping[str, torch.Tensor]) -> dict[str, dict]:
        """Extract diff-style LoRA weights."""
        lora_diffs = {}

        for key in lora_weights.keys():
            for check_suffix, add_suffix in DIFF_PATTERNS:
                if key.endswith(check_suffix):
                    base_key = key[: -len(check_suffix)]
                    model_key = self._get_model_key(base_key, add_suffix)

                    if model_key:
                        lora_diffs[model_key] = {
                            "diff_key": key,
                            "type": check_suffix,
                        }
                    break

        return lora_diffs

    def apply_lora(
        self,
        model_weights: dict[str, torch.Tensor] | nn.Module,
        lora_weights: Mapping[str, torch.Tensor] | str | Path,
        alpha: float | None = None,
        strength: float = 1.0,
    ) -> int:
        """Apply LoRA weights to model weights.

        Args:
            model_weights: The model weights dictionary or module
            lora_weights: The LoRA weights dictionary or file path
            alpha: Global alpha scaling factor
            strength: Additional strength factor for LoRA deltas

        Returns:
            Number of LoRA weights successfully applied
        """
        if isinstance(lora_weights, (str, Path)):
            path = str(lora_weights)
            if self.stream_safetensors and path.endswith(".safetensors"):
                with safe_open(path, framework="pt", device="cpu") as source:
                    metadata = source.metadata() or {}
                    metadata_alpha = metadata.get("alpha")
                    resolved_alpha = float(metadata_alpha) if alpha is None and metadata_alpha is not None else alpha
                    return self._apply_lora_mapping(
                        model_weights,
                        _SafetensorMapping(source),
                        alpha=resolved_alpha,
                        strength=strength,
                    )
            lora_weights = load_state_dict(path)
        return self._apply_lora_mapping(model_weights, lora_weights, alpha=alpha, strength=strength)

    def _resolve_target(
        self,
        model_key: str,
        weight_dict: Mapping[str, torch.Tensor],
    ) -> LoRATarget | None:
        if self.target_resolver is not None:
            return self.target_resolver(model_key, weight_dict)
        tensor = weight_dict.get(model_key)
        return None if tensor is None else LoRATarget(model_key, tensor)

    def _fail_or_warn(self, message: str, error: Exception | None = None) -> None:
        if self.strict:
            if error is None:
                raise ValueError(message)
            raise ValueError(message) from error
        logger.warning(message)

    @torch.no_grad()
    def _apply_lora_mapping(
        self,
        model_weights: dict[str, torch.Tensor] | nn.Module,
        lora_weights: Mapping[str, torch.Tensor],
        *,
        alpha: float | None,
        strength: float,
    ) -> int:
        if isinstance(model_weights, nn.Module):
            weight_dict = dict(model_weights.named_parameters())
            weight_dict.update(model_weights.named_buffers())
        else:
            weight_dict = model_weights

        # Extract LoRA pairs and diffs
        lora_pairs = self.extract_lora_pairs(lora_weights)
        lora_diffs = self.extract_lora_diffs(lora_weights)
        paired_keys = {
            key
            for pair_info in lora_pairs
            for key in (pair_info["up_key"], pair_info["down_key"], pair_info["alpha_key"])
            if key is not None
        }
        incomplete_keys = {
            key
            for key in lora_weights
            if (
                self._detect_lora_format(key) is not None
                or any(key.endswith(pattern[1]) for pattern in LORA_PATTERNS.values())
            )
            and key not in paired_keys
        }
        if incomplete_keys:
            self._fail_or_warn(f"Found incomplete LoRA pairs: {sorted(incomplete_keys)[:10]}")

        applied_count = 0
        used_lora_keys: set[str] = set()

        # Apply LoRA pairs (matrix multiplication)
        for pair_info in lora_pairs:
            model_key = pair_info["model_key"]
            target = self._resolve_target(model_key, weight_dict)
            if target is None:
                self._fail_or_warn(f"Model key not found: {model_key}")
                continue

            param = target.tensor
            up_key = pair_info["up_key"]
            down_key = pair_info["down_key"]

            # Track used keys
            used_lora_keys.add(up_key)
            used_lora_keys.add(down_key)
            if pair_info["mid_key"]:
                used_lora_keys.add(pair_info["mid_key"])
            if pair_info["alpha_key"]:
                used_lora_keys.add(pair_info["alpha_key"])

            try:
                lora_up = lora_weights[up_key]
                lora_down = lora_weights[down_key]
                if lora_up.ndim != 2 or lora_down.ndim != 2:
                    raise ValueError(f"down={tuple(lora_down.shape)}, up={tuple(lora_up.shape)}")

                # Calculate LoRA scale
                if pair_info["alpha"] is not None:
                    lora_scale = pair_info["alpha"] / lora_down.shape[0]
                elif alpha is not None:
                    lora_scale = alpha / lora_down.shape[0]
                elif self.default_alpha is not None:
                    lora_scale = self.default_alpha / lora_down.shape[0]
                else:
                    lora_scale = 1

                compute_device = param.device
                compute_dtype = self.merge_dtype or param.dtype
                lora_up = lora_up.to(device=compute_device, dtype=compute_dtype)
                lora_down = lora_down.to(device=compute_device, dtype=compute_dtype)
                lora_delta = torch.mm(lora_up, lora_down)
                if lora_delta.shape != param.shape:
                    raise ValueError(f"delta={tuple(lora_delta.shape)}, target={tuple(param.shape)} ({target.key})")
                param.add_(
                    lora_delta.to(dtype=param.dtype),
                    alpha=float(lora_scale) * float(strength),
                )
                applied_count += 1

            except Exception as e:
                self._fail_or_warn(f"Failed to apply LoRA pair for {model_key}: {e}", e)

        # Apply diff weights (direct addition)
        for model_key, diff_info in lora_diffs.items():
            target = self._resolve_target(model_key, weight_dict)
            if target is None:
                self._fail_or_warn(f"Model key not found for diff: {model_key}")
                continue

            param = target.tensor
            diff_key = diff_info["diff_key"]

            # Track used keys
            used_lora_keys.add(diff_key)

            try:
                lora_diff = lora_weights[diff_key].to(param.device, param.dtype)
                resolved_alpha = alpha if alpha is not None else self.default_alpha
                scale_factor = (resolved_alpha if resolved_alpha is not None else 1) * float(strength)
                if lora_diff.shape != param.shape:
                    raise ValueError(f"diff={tuple(lora_diff.shape)}, target={tuple(param.shape)} ({target.key})")
                param.add_(lora_diff, alpha=float(scale_factor))
                applied_count += 1
            except Exception as e:
                self._fail_or_warn(f"Failed to apply LoRA diff for {model_key}: {e}", e)

        # Warn about unused keys
        all_lora_keys = set(lora_weights)
        unused_lora_keys = all_lora_keys - used_lora_keys

        if unused_lora_keys:
            preview = sorted(unused_lora_keys)[:10]
            self._fail_or_warn(f"Found {len(unused_lora_keys)} unused LoRA weights: {preview}")

        logger.info(f"Applied {applied_count} LoRA weight adjustments")

        if applied_count == 0 and (lora_pairs or lora_diffs):
            self._fail_or_warn("No LoRA weights were applied! Check for key name mismatches.")

        return applied_count


__all__ = ["LoRALoader", "LoRATarget", "LoRATargetResolver"]
