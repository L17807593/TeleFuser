"""Export a CUDA trace for LingBot World V2's Ulysses Copy Engine overlap.

Run with:
    PYTHONPATH=/tmp/tf-kernel-ulysses torchrun --standalone --nproc-per-node=4 \
        tools/validation/profile_lingbot_world_v2_ulysses_overlap.py --output /tmp/lingbot-v2-trace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from telefuser.models import lingbot_world_fast_dit
from telefuser.models.lingbot_world_fast_dit import CausalSelfAttention
from telefuser.models.wan_video_dit import precompute_freqs_cis_3d


def _frequencies(head_dim: int, sequence_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = precompute_freqs_cis_3d(head_dim, end=sequence_length)
    cosine = torch.cat([frequency.real for frequency in frequencies], dim=-1).to(device)
    sine = torch.cat([frequency.imag for frequency in frequencies], dim=-1).to(device)
    return cosine, sine


def _run_attention(
    module: CausalSelfAttention,
    hidden: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    cache: dict[str, torch.Tensor | int],
    grid_size: tuple[int, int, int],
) -> torch.Tensor:
    output = module(
        hidden,
        freqs_cos,
        freqs_sin,
        grid_size,
        cache,
        current_start=0,
        max_attention_size=grid_size[-1],
        device_mesh=object(),
    )
    if output is None:
        raise RuntimeError("LingBot World V2 attention unexpectedly returned no output")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-sequence-length", type=int, default=1560)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError("this profiler requires four Ulysses ranks")

    torch.manual_seed(17)
    module = CausalSelfAttention(dim=2048, num_heads=16).eval().to(device, dtype=torch.bfloat16)
    torch.manual_seed(31 + rank)
    hidden = torch.randn(args.local_sequence_length, 2048, device=device, dtype=torch.bfloat16).unsqueeze(0)
    global_sequence_length = args.local_sequence_length * world_size
    grid_size = (1, 1, global_sequence_length)
    freqs_cos, freqs_sin = _frequencies(module.head_dim, global_sequence_length, device)
    cache = {
        "k": torch.zeros(
            1,
            global_sequence_length,
            module.num_heads // world_size,
            module.head_dim,
            device=device,
            dtype=torch.bfloat16,
        ),
        "v": torch.zeros(
            1,
            global_sequence_length,
            module.num_heads // world_size,
            module.head_dim,
            device=device,
            dtype=torch.bfloat16,
        ),
        "global_end_index": 0,
        "local_end_index": 0,
    }

    original_group = lingbot_world_fast_dit.get_ulysses_group
    original_world_size = lingbot_world_fast_dit.get_ulysses_world_size
    original_attention = lingbot_world_fast_dit.attn_func
    lingbot_world_fast_dit.get_ulysses_group = lambda _mesh: dist.group.WORLD
    lingbot_world_fast_dit.get_ulysses_world_size = lambda _mesh: world_size
    lingbot_world_fast_dit.attn_func = lambda query, _key, _value, **_kwargs: query
    try:
        _run_attention(module, hidden, freqs_cos, freqs_sin, cache, grid_size)
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
                _run_attention(module, hidden, freqs_cos, freqs_sin, cache, grid_size)
                torch.cuda.synchronize(device)
            profiler.export_chrome_trace(str(args.output / "lingbot_world_v2_ulysses_rank0.json.gz"))
        else:
            _run_attention(module, hidden, freqs_cos, freqs_sin, cache, grid_size)
            torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        lingbot_world_fast_dit.get_ulysses_group = original_group
        lingbot_world_fast_dit.get_ulysses_world_size = original_world_size
        lingbot_world_fast_dit.attn_func = original_attention
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
