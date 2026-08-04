# SPDX-License-Identifier: Apache-2.0
"""Capture MiniMax H3 text-refiner boundaries without loading the full DiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from telefuser.models.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MiniMaxH3TokenRefiner,
    _reorder_grouped_qkv_to_qkv,
)


class _TextRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, dtype=torch.bfloat16)
        self.token_refiner = MiniMaxH3TokenRefiner(config)


def _load_refiner(checkpoint_dir: Path, device: torch.device) -> _TextRefiner:
    config = MiniMaxH3DiTConfig.from_json(checkpoint_dir / "config.json")
    model = _TextRefiner(config)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    selected = [name for name in weight_map if name.startswith("condition_proj.") or name.startswith("token_refiner.")]
    shard_names = {weight_map[name] for name in selected}
    if len(shard_names) != 1:
        raise ValueError(f"text-refiner weights must occupy one shard, got {sorted(shard_names)}")
    shard_path = checkpoint_dir / next(iter(shard_names))
    state: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as stream:
        for name in selected:
            value = stream.get_tensor(name)
            if name.endswith(".attn.qkv_proj.weight"):
                value = _reorder_grouped_qkv_to_qkv(
                    value,
                    num_query_groups=config.num_attention_heads,
                    heads_per_group=1,
                    head_dim=config.attention_head_dim,
                )
            state[name] = value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    hidden = source["hidden_states"].to(device=device, dtype=torch.bfloat16)
    model = _load_refiner(args.checkpoint_dir, device)
    captured: dict[str, torch.Tensor] = {"input": hidden.detach().cpu()}
    with torch.inference_mode():
        hidden = model.condition_proj(hidden)
        captured["condition_proj"] = hidden.detach().cpu()
        for index, block in enumerate(model.token_refiner.blocks):
            hidden = block(
                hidden,
                sequence_lengths=[int(hidden.shape[0])],
                attention_config=None,
            )
            captured[f"block_{index}"] = hidden.detach().cpu()
        hidden = model.token_refiner.final_norm(hidden)
        captured["final"] = hidden.detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, args.output)


if __name__ == "__main__":
    main()
