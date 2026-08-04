# SPDX-License-Identifier: Apache-2.0
"""Replay MiniMax H3 DiT block 0 and compare stable operator boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from compare_minimax_h3_trajectories import _tensor_metrics
from safetensors import safe_open

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models.minimax_h3_dit import (
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTConfig,
    _modulate,
    _reorder_grouped_qkv_to_qkv,
    _rotate_half,
)
from telefuser.ops.attention import attention


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_block(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MiniMaxH3DiTBlock, torch.Tensor]:
    config = MiniMaxH3DiTConfig.from_json(config_path)
    block = MiniMaxH3DiTBlock(config)
    state = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        rope_inv_freq = checkpoint.get_tensor("rope.inv_freq")
        for name in block.state_dict():
            tensor = checkpoint.get_tensor(f"blocks.0.{name}")
            if name == "attn.qkv_proj.weight":
                tensor = _reorder_grouped_qkv_to_qkv(
                    tensor,
                    num_query_groups=config.num_attention_heads,
                    heads_per_group=1,
                    head_dim=config.attention_head_dim,
                )
            state[name] = tensor
    block.load_state_dict(state, strict=True)
    return block.to(device).eval(), rope_inv_freq.to(device=device, dtype=torch.float32)


def _record(comparisons: dict[str, Any], name: str, reference: torch.Tensor, candidate: torch.Tensor) -> None:
    comparisons[name] = _tensor_metrics(reference, candidate.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-layers", type=Path, required=True)
    parser.add_argument("--reference-block-details", type=Path, required=True)
    parser.add_argument("--candidate-layers", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rope-source", choices=("checkpoint", "captured-candidate"), default="checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    reference_layers = torch.load(args.reference_layers, map_location="cpu", weights_only=True, mmap=True)
    reference_details = torch.load(args.reference_block_details, map_location="cpu", weights_only=True, mmap=True)
    candidate_layers = torch.load(args.candidate_layers, map_location="cpu", weights_only=True, mmap=True)
    candidate_trajectory = torch.load(args.candidate_trajectory, map_location="cpu", weights_only=True, mmap=True)

    block_input = reference_layers["block_0_input"]
    hidden = block_input["hidden"].to(device)
    adaln_input = block_input["adaln_input"].to(device)
    combined_indices = block_input["combined_indices"].to(device)
    block, rope_inv_freq = _load_block(args.config, args.checkpoint, device)
    if args.rope_source == "checkpoint":
        position_ids = candidate_trajectory["transformer_layout"]["img_position_ids"][0].to(device).float()
        per_axis = position_ids.unsqueeze(-1) * rope_inv_freq.view(1, 1, -1)
        half = torch.cat(tuple(per_axis.unbind(dim=1)), dim=-1)
        rope_frequencies = torch.cat((half, half), dim=-1)
    else:
        rope_frequencies = candidate_layers["block_0_input"]["rope_frequencies"].to(device)
    cu_seqlens = candidate_trajectory["packed"]["cu_seqlens"].tolist()
    sequence_lengths = [
        stop - start for start, stop in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True) if stop > start
    ]

    comparisons: dict[str, Any] = {}
    with torch.inference_mode():
        adaln_params = block.adaln_proj(adaln_input)
        adaln_pairs = zip(reference_details["adaln_params"], adaln_params, strict=True)
        for index, (reference, candidate) in enumerate(adaln_pairs):
            _record(comparisons, f"adaln_params.{index}", reference, candidate)

        norm1 = block.norm1(hidden)
        _record(comparisons, "norm1", reference_details["norm1"], norm1)
        attention_input = _modulate(norm1, adaln_params[0], adaln_params[1], combined_indices)
        _record(comparisons, "attention_input", reference_details["attention_input"], attention_input)

        sequence = attention_input.shape[0]
        qkv = block.attn.qkv_proj(attention_input).reshape(
            sequence,
            3,
            block.attn.num_heads,
            block.attn.head_dim,
        )
        query, key, value = qkv.unbind(dim=1)
        query = block.attn.q_norm(query)
        key = block.attn.k_norm(key)
        rotary_dim = rope_frequencies.shape[-1]
        cosine = rope_frequencies.cos().unsqueeze(1).to(query.dtype)
        sine = rope_frequencies.sin().unsqueeze(1).to(query.dtype)
        query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        query = torch.cat((query_rotary * cosine + _rotate_half(query_rotary) * sine, query_pass), dim=-1)
        key = torch.cat((key_rotary * cosine + _rotate_half(key_rotary) * sine, key_pass), dim=-1)
        attended = attention(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attention_config=AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
            scale=block.attn.head_dim**-0.5,
            sequence_lengths=sequence_lengths,
        )
        attention_output = block.attn.out_proj(attended[0].reshape(sequence, block.attn.inner_dim))
        _record(comparisons, "attention_output", reference_details["attention_output"], attention_output)
        del attended, query, key, value, qkv

        post_attention = hidden + adaln_params[2].index_select(0, combined_indices) * attention_output
        _record(comparisons, "post_attention", reference_details["post_attention"], post_attention)
        norm2 = block.norm2(post_attention)
        _record(comparisons, "norm2", reference_details["norm2"], norm2)
        mlp_input = _modulate(norm2, adaln_params[3], adaln_params[4], combined_indices)
        _record(comparisons, "mlp_input", reference_details["mlp_input"], mlp_input)
        mlp_output = block.mlp(mlp_input)
        _record(comparisons, "mlp_output", reference_details["mlp_output"], mlp_output)
        block_output = post_attention + adaln_params[5].index_select(0, combined_indices) * mlp_output
        _record(comparisons, "block_output", reference_details["block_output"], block_output)

    report = {
        "schema_version": 1,
        "device": str(device),
        "rope_source": args.rope_source,
        "sequence_lengths": sequence_lengths,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "reference_layers": args.reference_layers,
                "reference_block_details": args.reference_block_details,
                "candidate_layers": args.candidate_layers,
                "candidate_trajectory": args.candidate_trajectory,
            }.items()
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
