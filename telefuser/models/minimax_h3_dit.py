# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed multimodal DiT."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig
from telefuser.distributed.device_mesh import get_ulysses_group, get_ulysses_world_size
from telefuser.distributed.parallel_shard import sequence_parallel_shard, sequence_parallel_unshard
from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads
from telefuser.ops.attention import attention

MINIMAX_H3_ADALN_MODALITY_NUM = 3
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


@dataclass(frozen=True)
class MiniMaxH3DiTConfig:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str | Path) -> MiniMaxH3DiTConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        values = {key: payload[key] for key in fields if key in payload}
        if "patch_size" in values:
            values["patch_size"] = tuple(int(value) for value in values["patch_size"])
        return cls(**values)

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("MiniMax H3 hidden_size and num_layers must be positive")
        if self.num_attention_heads <= 0 or self.attention_head_dim <= 0:
            raise ValueError("MiniMax H3 attention dimensions must be positive")
        if len(self.patch_size) != 3 or any(value <= 0 for value in self.patch_size):
            raise ValueError("MiniMax H3 patch_size must contain three positive integers")
        if 6 * self.rope_inv_freq_len > self.attention_head_dim:
            raise ValueError("MiniMax H3 rotary dimensions must fit inside attention_head_dim")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        return self.latents_dim * math.prod(self.patch_size)

    @property
    def adaln_out_features(self) -> int:
        return 18 * self.hidden_size

    @property
    def final_adaln_out_features(self) -> int:
        return 2 * self.hidden_size


def _rms_norm(size: int, eps: float) -> nn.RMSNorm:
    return nn.RMSNorm(size, eps=eps, dtype=torch.bfloat16)


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    if weight.shape[0] != num_query_groups * per_group:
        raise ValueError("MiniMax H3 grouped QKV weight has an incompatible output dimension")
    rest = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest)
    q, k, v = torch.split(grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1)
    return torch.cat(
        (
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest),
            k.reshape(num_query_groups * head_dim, *rest),
            v.reshape(num_query_groups * head_dim, *rest),
        ),
        dim=0,
    )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MiniMaxH3Rope(nn.Module):
    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        inv_freq = 10000.0 ** (-torch.arange(inv_freq_len, dtype=torch.float32) / inv_freq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim != 3 or position_ids.shape[0] != 1 or position_ids.shape[-1] != 3:
            raise ValueError("MiniMax H3 position_ids must have shape [1, sequence, 3]")
        per_axis = position_ids[0].float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        half = torch.cat(tuple(per_axis.unbind(dim=1)), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = config.timestep_input_dim
        self.proj_in = nn.Linear(
            config.timestep_input_dim,
            config.time_embed_hidden_size,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            config.time_embed_hidden_size,
            config.time_embed_dim,
            dtype=torch.float32,
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        args = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(embedding)))


class MiniMaxH3Attention(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.inner_dim = config.inner_dim
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * self.inner_dim, bias=False, dtype=torch.bfloat16)
        self.q_norm = _rms_norm(self.head_dim, config.qk_norm_eps)
        self.k_norm = _rms_norm(self.head_dim, config.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, config.hidden_size, bias=False, dtype=torch.bfloat16)
        self.ulysses_group: dist.ProcessGroup | None = None
        self._communication_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def set_ulysses_group(self, group: dist.ProcessGroup | None) -> None:
        self.ulysses_group = group

    def reset_communication_metrics(self) -> None:
        self._communication_events.clear()

    def communication_seconds(self) -> float:
        return sum(start.elapsed_time(end) for start, end in self._communication_events) / 1000.0

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_lengths: list[int],
        rope_frequencies: torch.Tensor | None,
        attention_config: AttentionConfig | None,
    ) -> torch.Tensor:
        sequence, _ = hidden.shape
        qkv = self.qkv_proj(hidden).reshape(sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if rope_frequencies is not None:
            rotary_dim = rope_frequencies.shape[-1]
            cosine = rope_frequencies.cos().unsqueeze(1).to(query.dtype)
            sine = rope_frequencies.sin().unsqueeze(1).to(query.dtype)
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query = torch.cat((query_rotary * cosine + _rotate_half(query_rotary) * sine, query_pass), dim=-1)
            key = torch.cat((key_rotary * cosine + _rotate_half(key_rotary) * sine, key_pass), dim=-1)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        group = self.ulysses_group
        use_ulysses = group is not None and dist.get_world_size(group) > 1
        if use_ulysses:
            scatter_start = torch.cuda.Event(enable_timing=True)
            scatter_end = torch.cuda.Event(enable_timing=True)
            scatter_start.record()
            qkv_wait = ulysses_scatter_heads(torch.cat((query, key, value), dim=-1), group)
            query, key, value = qkv_wait().chunk(3, dim=-1)
            scatter_end.record()
            self._communication_events.append((scatter_start, scatter_end))
        output = attention(
            query,
            key,
            value,
            attention_config=attention_config,
            scale=self.head_dim**-0.5,
            sequence_lengths=sequence_lengths,
        )
        if use_ulysses:
            gather_start = torch.cuda.Event(enable_timing=True)
            gather_end = torch.cuda.Event(enable_timing=True)
            gather_start.record()
            output = ulysses_gather_heads(output, group, num_heads=self.num_heads)()
            gather_end.record()
            self._communication_events.append((gather_start, gather_end))
        return self.out_proj(output[0].reshape(sequence, self.inner_dim))


class MiniMaxH3MLP(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(hidden).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(gate) * up)


class MiniMaxH3AdaLNProjection(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig, *, expand_ratio: int, modality_count: int) -> None:
        super().__init__()
        self.expand_ratio = expand_ratio
        self.modality_count = modality_count
        self.hidden_size = config.hidden_size
        self.linear = nn.Linear(
            config.time_embed_dim,
            expand_ratio * modality_count * config.hidden_size,
            dtype=torch.bfloat16,
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.linear(embedding)
        output = output.reshape(-1, self.expand_ratio * self.hidden_size)
        return tuple(output.chunk(self.expand_ratio, dim=-1))


def _modulate(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return hidden * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.norm1 = _rms_norm(config.hidden_size, config.norm_eps)
        self.norm2 = _rms_norm(config.hidden_size, config.norm_eps)
        self.attn = MiniMaxH3Attention(config)
        self.mlp = MiniMaxH3MLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_lengths: list[int],
        attention_config: AttentionConfig | None,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(
            self.norm1(hidden),
            sequence_lengths=sequence_lengths,
            rope_frequencies=None,
            attention_config=attention_config,
        )
        return hidden + self.mlp(self.norm2(hidden))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(config) for _ in range(config.token_refiner_num_layers)]
        )
        self.final_norm = _rms_norm(config.hidden_size, config.final_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_config: AttentionConfig | None,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(
                hidden,
                sequence_lengths=[hidden.shape[0]],
                attention_config=attention_config,
            )
        return self.final_norm(hidden)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.norm1 = _rms_norm(config.hidden_size, config.norm_eps)
        self.norm2 = _rms_norm(config.hidden_size, config.norm_eps)
        self.attn = MiniMaxH3Attention(config)
        self.mlp = MiniMaxH3MLP(config)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config,
            expand_ratio=6,
            modality_count=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        sequence_lengths: list[int],
        rope_frequencies: torch.Tensor,
        attention_config: AttentionConfig | None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(adaln_input)
        residual = hidden
        value = _modulate(self.norm1(hidden), shift_msa, scale_msa, combined_indices)
        value = self.attn(
            value,
            sequence_lengths=sequence_lengths,
            rope_frequencies=rope_frequencies,
            attention_config=attention_config,
        )
        hidden = residual + gate_msa.index_select(0, combined_indices) * value
        residual = hidden
        value = _modulate(self.norm2(hidden), shift_mlp, scale_mlp, combined_indices)
        value = self.mlp(value)
        return residual + gate_mlp.index_select(0, combined_indices) * value


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.norm = _rms_norm(config.hidden_size, config.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdaLNProjection(config, expand_ratio=2, modality_count=1)
        self.video_out = nn.Linear(config.hidden_size, config.video_patch_dim, dtype=torch.float32)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_latents_dim, dtype=torch.float32)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        hidden = _modulate(self.norm(hidden), shift, scale, inverse_indices).float()
        return self.video_out(hidden), self.audio_out(hidden)


class MiniMaxH3DiT(BaseModel):
    """Faithful packed DiT baseline for H3-Base with optional Ulysses SP."""

    def __init__(self, config: MiniMaxH3DiTConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniMaxH3DiTConfig()
        config = self.config
        self.video_patch_proj = nn.Linear(config.video_patch_dim, config.hidden_size, dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, dtype=torch.float32)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, dtype=torch.bfloat16)
        self.time_embedder = MiniMaxH3TimeEmbedder(config)
        self.rope = MiniMaxH3Rope(config.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(config) for _ in range(config.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(config)
        self.layer_name_list = ["blocks"]
        self.device_mesh: Any | None = None
        self.usp_flag = False

    def _preserve_fp32_boundaries(self) -> None:
        for name in MINIMAX_H3_FP32_PARAM_NAMES:
            parameter = self.get_parameter(name)
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        if self.rope.inv_freq.dtype != torch.float32:
            self.rope.inv_freq.data = self.rope.inv_freq.data.float()

    def to(self, *args: Any, **kwargs: Any) -> MiniMaxH3DiT:
        preserved_parameters = {
            name: parameter.detach().clone()
            for name in MINIMAX_H3_FP32_PARAM_NAMES
            if not (parameter := self.get_parameter(name)).is_meta
        }
        preserved_buffers = {
            name: buffer.detach().clone()
            for name in MINIMAX_H3_FP32_BUFFER_NAMES
            if not (buffer := self.get_buffer(name)).is_meta
        }
        result = super().to(*args, **kwargs)
        for name, value in preserved_parameters.items():
            parameter = result.get_parameter(name)
            parameter.data = value.to(device=parameter.device, dtype=torch.float32)
        for name, value in preserved_buffers.items():
            buffer = result.get_buffer(name)
            buffer.data = value.to(device=buffer.device, dtype=torch.float32)
        result._preserve_fp32_boundaries()
        return result

    @staticmethod
    def _position_ids(value: Any, name: str) -> torch.Tensor:
        position_ids = value.get("position_ids") if isinstance(value, dict) else getattr(value, "position_ids", None)
        if position_ids is None:
            raise ValueError(f"{name}.position_ids is required")
        return position_ids.reshape(-1).long()

    @staticmethod
    def _sequence_lengths(packed: Any) -> list[int]:
        cu = packed.get("cu_seqlens_q") if isinstance(packed, dict) else packed.cu_seqlens_q
        values = [int(value) for value in cu.tolist()]
        return [stop - start for start, stop in zip(values[:-1], values[1:], strict=True) if stop > start]

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        required = (
            "x",
            "audio_x",
            "img_position_ids",
            "unique_timesteps",
            "inverse_indices",
            "update_mask",
            "prompt_embeds",
            "img_pos_info",
            "audio_pos_info",
            "text_pos_info",
            "img_pos_for_infer_output_info",
            "packed_seq_params",
        )
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise ValueError(f"MiniMaxH3DiT.forward missing required inputs: {missing}")
        video_state = kwargs["x"]
        audio_state = kwargs["audio_x"]
        if video_state.ndim != 3 or video_state.shape[0] != 1:
            raise ValueError("x must have shape [1, sequence, video_patch_dim]")
        sequence = video_state.shape[1]
        device = video_state.device
        image_positions = self._position_ids(kwargs["img_pos_info"], "img_pos_info").to(device)
        audio_positions = self._position_ids(kwargs["audio_pos_info"], "audio_pos_info").to(device)
        text_positions = self._position_ids(kwargs["text_pos_info"], "text_pos_info").to(device)
        output_positions = self._position_ids(
            kwargs["img_pos_for_infer_output_info"], "img_pos_for_infer_output_info"
        ).to(device)

        prompt = kwargs["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
        live_text = text_positions.numel()
        prompt = self.condition_proj(prompt[:live_text])
        prompt = self.token_refiner(prompt, attention_config=self.attention_config)
        hidden = torch.zeros(sequence, self.config.hidden_size, device=device, dtype=torch.bfloat16)
        hidden.index_copy_(0, text_positions, prompt)
        video_rows = video_state[0].index_select(0, image_positions).float()
        audio_rows = audio_state[0].index_select(0, audio_positions).float()
        hidden.index_copy_(0, image_positions, self.video_patch_proj(video_rows).to(torch.bfloat16))
        hidden.index_copy_(0, audio_positions, self.audio_patch_proj(audio_rows).to(torch.bfloat16))

        timesteps = kwargs["unique_timesteps"].reshape(-1).to(device)
        adaln_input = nn.functional.silu(self.time_embedder(timesteps)).to(torch.bfloat16)
        inverse_indices = kwargs["inverse_indices"].reshape(-1).long().to(device)
        if inverse_indices.numel() != sequence:
            raise ValueError("inverse_indices must cover the full packed sequence")
        token_tags = kwargs.get("block_token_tags")
        if token_tags is None:
            token_tags = kwargs.get("token_tags")
        if token_tags is None:
            raise ValueError("token_tags or block_token_tags is required")
        token_tags = token_tags.reshape(-1).long().to(device).clamp_min(0)
        combined_indices = token_tags + inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM
        rope_frequencies = self.rope(kwargs["img_position_ids"].to(device))
        sequence_lengths = self._sequence_lengths(kwargs["packed_seq_params"])
        full_sequence = sequence
        if self.usp_flag:
            world_size = get_ulysses_world_size(self.device_mesh)
            if sequence % world_size:
                raise ValueError(
                    f"MiniMax H3 packed sequence length ({sequence}) must be divisible by Ulysses degree ({world_size})"
                )
            inverse_indices = inverse_indices.clone()
            sequence_parallel_shard(
                self.device_mesh,
                [hidden, combined_indices, inverse_indices, rope_frequencies],
                [0, 0, 0, 0],
            )
        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                sequence_lengths=sequence_lengths,
                rope_frequencies=rope_frequencies,
                attention_config=self.attention_config,
            )
        video_logits, audio_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=inverse_indices,
        )
        if self.usp_flag:
            video_logits, audio_logits = sequence_parallel_unshard(
                self.device_mesh,
                [video_logits, audio_logits],
                [0, 0],
                [full_sequence, full_sequence],
            )
        video_logits = video_logits.index_select(0, output_positions)
        audio_logits = audio_logits.index_select(0, audio_positions)
        if not bool(kwargs.get("skip_mask_out_condition", False)):
            video_logits = video_logits * kwargs["update_mask"].reshape(-1, 1).to(video_logits)
            if kwargs.get("update_audio_mask") is not None:
                audio_logits = audio_logits * kwargs["update_audio_mask"].reshape(-1, 1).to(audio_logits)
        return video_logits, audio_logits

    def enable_usp(self, device_mesh: Any | None = None) -> None:
        self.device_mesh = device_mesh if device_mesh is not None else self.device_mesh
        world_size = get_ulysses_world_size(self.device_mesh)
        if self.config.num_attention_heads % world_size:
            raise ValueError(
                f"MiniMax H3 attention heads ({self.config.num_attention_heads}) must be divisible by "
                f"Ulysses degree ({world_size})"
            )
        group = get_ulysses_group(self.device_mesh) if world_size > 1 else None
        self.usp_flag = world_size > 1
        for block in self.blocks:
            block.attn.set_ulysses_group(group)

    def reset_communication_metrics(self) -> None:
        for block in self.blocks:
            block.attn.reset_communication_metrics()

    def communication_seconds(self) -> float:
        return sum(block.attn.communication_seconds() for block in self.blocks)

    def get_fsdp_module_names(self) -> list[str]:
        return ["blocks"]

    @staticmethod
    def state_dict_converter(config_path: str | Path | None = None) -> MiniMaxH3DiTStateDictConverter:
        return MiniMaxH3DiTStateDictConverter(config_path=config_path)


_BLOCK_INDEX = re.compile(r"^blocks\.(\d+)\.")
_REFINER_INDEX = re.compile(r"^token_refiner\.blocks\.(\d+)\.")


class MiniMaxH3DiTStateDictConverter:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = None if config_path is None else Path(config_path)

    def _config(self, state_dict: dict[str, torch.Tensor]) -> MiniMaxH3DiTConfig:
        if self.config_path is not None:
            return MiniMaxH3DiTConfig.from_json(self.config_path)
        q_norm = state_dict["blocks.0.attn.q_norm.weight"]
        qkv = state_dict["blocks.0.attn.qkv_proj.weight"]
        layers = 1 + max(int(match.group(1)) for key in state_dict if (match := _BLOCK_INDEX.match(key)))
        refiners = 1 + max(int(match.group(1)) for key in state_dict if (match := _REFINER_INDEX.match(key)))
        video_patch_dim = state_dict["video_patch_proj.weight"].shape[1]
        return MiniMaxH3DiTConfig(
            hidden_size=state_dict["video_patch_proj.weight"].shape[0],
            num_layers=layers,
            token_refiner_num_layers=refiners,
            num_attention_heads=qkv.shape[0] // (3 * q_norm.numel()),
            attention_head_dim=q_norm.numel(),
            ffn_hidden_size=state_dict["blocks.0.mlp.fc1.weight"].shape[0] // 2,
            latents_dim=video_patch_dim // 4,
            audio_latents_dim=state_dict["audio_patch_proj.weight"].shape[1],
            text_dim=state_dict["condition_proj.weight"].shape[1],
            timestep_input_dim=state_dict["time_embedder.proj_in.weight"].shape[1],
            time_embed_hidden_size=state_dict["time_embedder.proj_in.weight"].shape[0],
            time_embed_dim=state_dict["time_embedder.proj_out.weight"].shape[0],
            rope_inv_freq_len=state_dict["rope.inv_freq"].numel(),
        )

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        config = self._config(state_dict)
        converted = dict(state_dict)
        for key, value in state_dict.items():
            if key.endswith(".attn.qkv_proj.weight"):
                converted[key] = _reorder_grouped_qkv_to_qkv(
                    value,
                    num_query_groups=config.num_attention_heads,
                    heads_per_group=1,
                    head_dim=config.attention_head_dim,
                )
        return converted, {"config": config}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        renamed: dict[str, torch.Tensor] = {}
        qkv_parts: dict[str, dict[str, torch.Tensor]] = {}
        direct = {
            "proj_in.": "video_patch_proj.",
            "audio_proj_in.": "audio_patch_proj.",
            "context_embedder.": "condition_proj.",
            "time_embedder.linear_1.": "time_embedder.proj_in.",
            "time_embedder.linear_2.": "time_embedder.proj_out.",
            "norm_out.norm.": "final_layer.norm.",
            "norm_out.linear.": "final_layer.adaln_proj.linear.",
            "proj_out.": "final_layer.video_out.",
            "audio_proj_out.": "final_layer.audio_out.",
        }
        for key, value in state_dict.items():
            target = key
            for source, destination in direct.items():
                if target.startswith(source):
                    target = destination + target[len(source) :]
                    break
            target = target.replace("transformer_blocks.", "blocks.")
            target = target.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
            target = target.replace(".attn.norm_q.", ".attn.q_norm.")
            target = target.replace(".attn.norm_k.", ".attn.k_norm.")
            target = target.replace(".attn.to_out.0.", ".attn.out_proj.")
            target = target.replace(".ff.net.0.proj.", ".mlp.fc1.")
            target = target.replace(".ff.net.2.", ".mlp.fc2.")
            for part in ("q", "k", "v"):
                marker = f".attn.to_{part}."
                if marker in target:
                    prefix, suffix = target.split(marker, 1)
                    qkv_parts.setdefault(f"{prefix}.attn.qkv_proj.{suffix}", {})[part] = value
                    break
            else:
                renamed[target] = value
        for target, parts in qkv_parts.items():
            if set(parts) != {"q", "k", "v"}:
                raise ValueError(f"incomplete Diffusers QKV weights for {target}")
            renamed[target] = torch.cat((parts["q"], parts["k"], parts["v"]), dim=0)
        config = self._config(renamed)
        return renamed, {"config": config}


__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiT",
    "MiniMaxH3DiTConfig",
    "MiniMaxH3DiTStateDictConverter",
    "_reorder_grouped_qkv_to_qkv",
]
