"""Compile-aware LingBot-VLA v2 MoE operation dispatch."""

from __future__ import annotations

import torch


def robby_moe_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run the optional Triton grouped-MoE path for VLA eager inference."""
    if torch.compiler.is_compiling():
        raise RuntimeError("LingBot-VLA v2 Triton MoE is disabled during torch.compile")
    if hidden_states.device.type != "cuda":
        raise RuntimeError("LingBot-VLA v2 Triton MoE requires CUDA tensors")

    from telefuser.kernel.triton.lingbot_vla_v2_moe import robby_moe_forward as _triton_robby_moe_forward

    return _triton_robby_moe_forward(
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
        workspace=workspace,
    )
