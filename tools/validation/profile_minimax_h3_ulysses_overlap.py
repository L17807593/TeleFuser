"""Export a CUDA trace for the MiniMax H3 Ulysses scatter overlap.

Run with:
    PYTHONPATH=/tmp/tf-kernel-ulysses torchrun --standalone --nproc-per-node=4 \
        tools/validation/profile_minimax_h3_ulysses_overlap.py --output /tmp/h3-ulysses-trace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models import minimax_h3_dit
from telefuser.models.minimax_h3_dit import MiniMaxH3Attention, MiniMaxH3DiTConfig


def _config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=5376,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=28,
        attention_head_dim=128,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=16,
    )


def _ulysses_pair(rank: int) -> dist.ProcessGroup:
    pairs = ((0, 1), (2, 3))
    groups = [dist.new_group(ranks) for ranks in pairs]
    return groups[rank // 2]


def _run_attention(
    module: MiniMaxH3Attention,
    hidden: torch.Tensor,
    rope_cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    return module(
        hidden,
        sequence_lengths=[hidden.shape[0]],
        rope_cos_sin_cache=rope_cos_sin_cache,
        attention_config=AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=9472)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("this profiler requires four ranks for the Ulysses2 x TP2 shape")

    torch.manual_seed(17 + rank)
    module = MiniMaxH3Attention(_config()).eval().to(device)
    module.set_ulysses_group(_ulysses_pair(rank))
    hidden = torch.randn(args.sequence_length, module.qkv_proj.in_features, device=device, dtype=torch.bfloat16)
    angles = torch.randn(args.sequence_length, 16, device=device, dtype=torch.float32)
    rope_cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

    # The trace isolates pre-attention communication from the much larger attention kernel.
    original_attention = minimax_h3_dit.attention
    minimax_h3_dit.attention = lambda query, *_args, **_kwargs: query
    try:
        _run_attention(module, hidden, rope_cos_sin_cache)
        torch.cuda.synchronize(device)
        dist.barrier()

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            with torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                _run_attention(module, hidden, rope_cos_sin_cache)
                torch.cuda.synchronize(device)
            profiler.export_chrome_trace(str(args.output / "minimax_h3_ulysses_rank0.json.gz"))
        else:
            _run_attention(module, hidden, rope_cos_sin_cache)
            torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        minimax_h3_dit.attention = original_attention
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
