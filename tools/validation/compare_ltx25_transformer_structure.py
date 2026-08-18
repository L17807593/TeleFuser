"""Compare upstream and isolated LTX-2.5 transformer preprocessing with shared weights.

Run with the pinned upstream ``ltx_core`` package on ``PYTHONPATH``.  This
small CPU probe isolates model-structure differences from checkpoint loading
and CUDA attention numerical variation.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

import torch
from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction
from ltx_core.model.transformer.modality import Modality as UpstreamModality
from ltx_core.model.transformer.model import LTXModel as UpstreamLTXModel
from ltx_core.model.transformer.model import LTXModelType as UpstreamLTXModelType
from ltx_core.model.transformer.rope import LTXRopeType as UpstreamLTXRopeType
from ltx_core.model.transformer.transformer import TransformerOpsConfig

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models.ltx25.transformer import Attention as TeleFuserAttention
from telefuser.models.ltx25.transformer import (
    BatchedPerturbationConfig,
    PerturbationConfig,
)
from telefuser.models.ltx25.transformer import (
    LTXModel as TeleFuserLTXModel,
)
from telefuser.models.ltx25.transformer import (
    LTXModelType as TeleFuserLTXModelType,
)
from telefuser.models.ltx25.transformer import (
    LTXRopeType as TeleFuserLTXRopeType,
)
from telefuser.models.ltx25.transformer import (
    Modality as TeleFuserModality,
)


def _max_abs_error(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None:
        return None if left is right else float("inf")
    return float((left.float() - right.float()).abs().max().detach())


def _model_kwargs(
    rope_type: UpstreamLTXRopeType | TeleFuserLTXRopeType,
    *,
    attention_head_dim: int = 8,
) -> dict[str, object]:
    cross_attention_dim = 3 * attention_head_dim
    return {
        "num_attention_heads": 3,
        "attention_head_dim": attention_head_dim,
        "in_channels": 8,
        "out_channels": 8,
        "num_layers": 1,
        "cross_attention_dim": cross_attention_dim,
        "audio_num_attention_heads": 3,
        "audio_attention_head_dim": attention_head_dim,
        "audio_in_channels": 8,
        "audio_out_channels": 8,
        "audio_cross_attention_dim": cross_attention_dim,
        "positional_embedding_max_pos": [20, 32, 32],
        "audio_positional_embedding_max_pos": [20],
        "cross_attention_adaln": True,
        "apply_gated_attention": True,
        "use_keyframes_abs_pos_embedding": True,
        "double_precision_rope": True,
        "rope_type": rope_type,
    }


def _inputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(4)
    video = {
        "latent": torch.randn(1, 3, 8),
        "sigma": torch.tensor([1.0]),
        "timesteps": torch.ones(1, 3),
        "positions": torch.tensor(
            [
                [
                    [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                ]
            ]
        ),
        "context": torch.randn(1, 2, 24),
        "keyframes_mask": torch.tensor([[[1.0], [0.0], [0.0]]]),
    }
    audio = {
        "latent": torch.randn(1, 2, 8),
        "sigma": torch.tensor([1.0]),
        "timesteps": torch.ones(1, 2),
        "positions": torch.tensor([[[[0.0, 1.0], [1.0, 2.0]]]]),
        "context": torch.randn(1, 2, 24),
    }
    return video, audio


def _cuda_inputs(
    video_tokens: int,
    audio_tokens: int,
    cross_attention_dim: int,
) -> tuple[TeleFuserModality, UpstreamModality, TeleFuserModality, UpstreamModality]:
    video, audio = _inputs()
    video["latent"] = torch.randn(1, video_tokens, 8, dtype=torch.bfloat16, device="cuda")
    video["timesteps"] = torch.full((1, video_tokens, 1), 0.421875, device="cuda")
    video["positions"] = torch.zeros(1, 3, video_tokens, 2, device="cuda")
    video["context"] = torch.randn(1, 2, cross_attention_dim, dtype=torch.bfloat16, device="cuda")
    video["sigma"] = video["sigma"].to(device="cuda")
    video["keyframes_mask"] = torch.zeros(1, video_tokens, 1, device="cuda")
    audio["latent"] = torch.randn(1, audio_tokens, 8, dtype=torch.bfloat16, device="cuda")
    audio["timesteps"] = torch.full((1, audio_tokens, 1), 0.421875, device="cuda")
    audio["positions"] = torch.zeros(1, 1, audio_tokens, 2, device="cuda")
    audio["context"] = torch.randn(1, 2, cross_attention_dim, dtype=torch.bfloat16, device="cuda")
    audio["sigma"] = audio["sigma"].to(device="cuda")
    return TeleFuserModality(**video), UpstreamModality(**video), TeleFuserModality(**audio), UpstreamModality(**audio)


def _cuda_repeat_errors(*, production_shape: bool) -> list[float]:
    """Exercise FA4 at the 8-call stage-1 then 3-call stage-2 shape transition."""
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-repeat requires CUDA")
    torch.manual_seed(3)
    head_dim = 128 if production_shape else 8
    upstream = UpstreamLTXModel(
        model_type=UpstreamLTXModelType.AudioVideo,
        ops=TransformerOpsConfig.from_functions(AttentionFunction.FLASH_ATTENTION_4, MaskedAttentionFunction.PYTORCH),
        **_model_kwargs(UpstreamLTXRopeType.SPLIT, attention_head_dim=head_dim),
    ).eval()
    with torch.no_grad():
        for parameter in upstream.parameters():
            parameter.normal_()
    telefuser = TeleFuserLTXModel(
        model_type=TeleFuserLTXModelType.AudioVideo,
        **_model_kwargs(TeleFuserLTXRopeType.SPLIT, attention_head_dim=head_dim),
    ).eval()
    telefuser.load_state_dict(upstream.state_dict(), strict=True)
    upstream = upstream.to(device="cuda", dtype=torch.bfloat16)
    telefuser = telefuser.to(device="cuda", dtype=torch.bfloat16)
    TeleFuserAttention.attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_4)
    errors: list[float] = []
    with torch.inference_mode():
        stage_shapes = ((48, 9, 8), (192, 9, 3)) if production_shape else ((6, 2, 8), (12, 2, 3))
        for video_tokens, audio_tokens, calls in stage_shapes:
            tv, uv, ta, ua = _cuda_inputs(video_tokens, audio_tokens, 3 * head_dim)
            for _ in range(calls):
                upstream_out = upstream(uv, ua, None)
                telefuser_out = telefuser(tv, ta, BatchedPerturbationConfig([PerturbationConfig.empty()]))
                errors.extend(
                    [
                        _max_abs_error(upstream_out[0], telefuser_out[0]) or 0.0,
                        _max_abs_error(upstream_out[1], telefuser_out[1]) or 0.0,
                    ]
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-repeat", action="store_true")
    parser.add_argument("--cuda-production-shape", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(3)
    upstream = UpstreamLTXModel(
        model_type=UpstreamLTXModelType.AudioVideo,
        ops=TransformerOpsConfig.from_functions(AttentionFunction.PYTORCH, MaskedAttentionFunction.PYTORCH),
        **_model_kwargs(UpstreamLTXRopeType.SPLIT),
    ).eval()
    with torch.no_grad():
        for parameter in upstream.parameters():
            parameter.normal_()

    telefuser = TeleFuserLTXModel(
        model_type=TeleFuserLTXModelType.AudioVideo,
        **_model_kwargs(TeleFuserLTXRopeType.SPLIT),
    ).eval()
    telefuser.load_state_dict(upstream.state_dict(), strict=True)

    video, audio = _inputs()
    upstream_video = UpstreamModality(**video)
    upstream_audio = UpstreamModality(**audio)
    telefuser_video = TeleFuserModality(**video)
    telefuser_audio = TeleFuserModality(**audio)
    upstream_args = upstream.video_args_preprocessor.prepare(upstream_video, upstream_audio)
    telefuser_args = telefuser.video_args_preprocessor.prepare(telefuser_video, telefuser_audio)

    errors = {}
    for field in fields(upstream_args):
        if not hasattr(telefuser_args, field.name):
            continue
        upstream_value = getattr(upstream_args, field.name)
        if isinstance(upstream_value, torch.Tensor) or upstream_value is None:
            errors[field.name] = _max_abs_error(upstream_value, getattr(telefuser_args, field.name))
    print(errors)
    if any(error not in (None, 0.0) for error in errors.values()):
        raise SystemExit(1)

    upstream_out = upstream(upstream_video, upstream_audio, None)
    telefuser_out = telefuser(
        telefuser_video,
        telefuser_audio,
        BatchedPerturbationConfig([PerturbationConfig.empty()]),
    )
    print(
        {
            "video_output": _max_abs_error(upstream_out[0], telefuser_out[0]),
            "audio_output": _max_abs_error(upstream_out[1], telefuser_out[1]),
        }
    )
    if args.cuda_repeat or args.cuda_production_shape:
        errors = _cuda_repeat_errors(production_shape=args.cuda_production_shape)
        print({"cuda_repeat_max_abs_error": max(errors), "cuda_repeat_calls": len(errors) // 2})
        if any(error != 0.0 for error in errors):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
