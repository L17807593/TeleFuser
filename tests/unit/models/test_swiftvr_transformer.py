import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from telefuser.core.config import CompileConfig, ParallelConfig, QuantConfig, QuantKernelBackend, QuantType
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.models.swiftvr_transformer import (
    SwiftVRWanTransformer3DModel,
    _WindowRuntimeMetaCache,
    _make_hw_starts,
    compile_transformer_blocks_with_config,
    get_1d_rotary_pos_embed,
)
from telefuser.pipelines.swiftvr.streaming_dit import _ensure_rope_cache_len, _rope_with_offset


def _rope(length: int = 4) -> SimpleNamespace:
    parts = [get_1d_rotary_pos_embed(4, length) for _ in range(3)]
    return SimpleNamespace(
        t_dim=4,
        h_dim=4,
        w_dim=4,
        freqs_cos=torch.cat([part[0] for part in parts], dim=1),
        freqs_sin=torch.cat([part[1] for part in parts], dim=1),
    )


def _swiftvr_sp_test_model() -> SwiftVRWanTransformer3DModel:
    model = SwiftVRWanTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=4,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=4,
        freq_dim=4,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=False,
        self_attn_window_hw=(2, 2),
    ).cuda()
    model.to(dtype=torch.bfloat16).eval()
    model.prepare_for_inference()
    return model


def _swiftvr_ulysses_parity_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device("cuda", rank))
    try:
        torch.manual_seed(17)
        reference = _swiftvr_sp_test_model()
        torch.manual_seed(17)
        distributed_model = _swiftvr_sp_test_model()
        distributed_model.enable_usp(
            create_device_mesh_from_config(
                ParallelConfig(device_ids=list(range(world_size)), sp_ulysses_degree=world_size)
            )
        )
        torch.manual_seed(23)
        hidden_states = torch.randn((1, 4, 2, 4, 4), device="cuda", dtype=torch.bfloat16)
        timestep = torch.ones((1,), device="cuda", dtype=torch.float32)
        context = torch.randn((1, 4, 4), device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = reference(hidden_states, timestep, context).sample
            actual = distributed_model(hidden_states, timestep, context).sample
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_window_starts_cover_boundary_without_redundant_interior() -> None:
    h_starts, w_starts = _make_hw_starts(11, 10, 4, 4, False)

    assert h_starts.tolist() == [0, 4, 7]
    assert w_starts.tolist() == [0, 4, 6]

    shifted_h, shifted_w = _make_hw_starts(11, 10, 4, 4, True)
    assert shifted_h.tolist() == [0, 2, 6, 7]
    assert shifted_w.tolist() == [0, 2, 6]


def test_shifted_window_owner_scatter_returns_each_global_token_once() -> None:
    for shifted, prefer_front in ((False, True), (True, False)):
        meta = _WindowRuntimeMetaCache.get(
            2,
            5,
            6,
            4,
            4,
            do_shift=shifted,
            prefer_front=prefer_front,
            device=torch.device("cpu"),
        )
        gathered_global_indices = meta.lin_flat
        restored = torch.index_select(gathered_global_indices, 0, meta.owner_pos)
        assert torch.equal(restored, torch.arange(meta.THW))


def test_rope_extension_preserves_existing_values() -> None:
    rope = _rope()
    original_cos = rope.freqs_cos.clone()
    original_sin = rope.freqs_sin.clone()

    _ensure_rope_cache_len(rope, 12)

    assert rope.freqs_cos.shape == (12, 12)
    assert rope.freqs_sin.shape == (12, 12)
    torch.testing.assert_close(rope.freqs_cos[:4], original_cos, rtol=0, atol=0)
    torch.testing.assert_close(rope.freqs_sin[:4], original_sin, rtol=0, atol=0)


def test_rope_offset_uses_global_temporal_position() -> None:
    rope = _rope()
    cos, sin = _rope_with_offset(rope, 3, 2, 2, t_off=5)
    cos_grid = cos.view(3, 2, 2, 12)
    sin_grid = sin.view(3, 2, 2, 12)

    torch.testing.assert_close(cos_grid[:, 0, 0, :4], rope.freqs_cos[5:8, :4])
    torch.testing.assert_close(sin_grid[:, 0, 0, :4], rope.freqs_sin[5:8, :4])


def test_swiftvr_transformer_torchao_fp8_quantization_uses_block_linears(monkeypatch) -> None:
    calls = []

    def replace_linear_layers_with_torchao_fp8(module, *, include_names=None, exclude_names=()):
        calls.append((module, include_names, exclude_names))
        return 2

    monkeypatch.setattr(
        "telefuser.ops.torchao_fp8_linear.replace_linear_layers_with_torchao_fp8",
        replace_linear_layers_with_torchao_fp8,
    )
    model = SwiftVRWanTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=4,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        cross_attn_norm=False,
    )

    model.enable_quant(QuantConfig(enabled=True, quant_type=QuantType.TORCHAO_FP8))

    assert calls == [(model, ("blocks.",), ("head", "time_embedding", "time_projection", "patch_embedding"))]
    assert model.torchao_fp8_replaced_linear == 2
    assert model.quant_type is QuantType.TORCHAO_FP8


def test_swiftvr_transformer_enables_ulysses_sp_on_shifted_window_attention(monkeypatch) -> None:
    model = SwiftVRWanTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=4,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=4,
        freq_dim=4,
        ffn_dim=8,
        num_layers=2,
        cross_attn_norm=False,
    )
    mesh = object()
    monkeypatch.setattr("telefuser.models.swiftvr_transformer.get_attention_strategy", lambda _mesh: "ulysses")
    monkeypatch.setattr("telefuser.models.swiftvr_transformer.get_ulysses_world_size", lambda _mesh: 2)

    model.enable_usp(mesh)

    assert model.device_mesh is mesh
    assert all(block.attn1.device_mesh is mesh for block in model.blocks)


def test_swiftvr_transformer_rejects_non_ulysses_sp(monkeypatch) -> None:
    model = SwiftVRWanTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=4,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=4,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        cross_attn_norm=False,
    )
    monkeypatch.setattr("telefuser.models.swiftvr_transformer.get_attention_strategy", lambda _mesh: "ring")

    with pytest.raises(ValueError, match="Ulysses SP only"):
        model.enable_usp(object())


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_swiftvr_ulysses_sp_matches_single_rank_output() -> None:
    mp.spawn(_swiftvr_ulysses_parity_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_compile_transformer_blocks_uses_runtime_config(monkeypatch) -> None:
    model = SwiftVRWanTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=4,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        cross_attn_norm=False,
    )
    block = model.blocks[0]
    calls = []

    def compile_block(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(torch, "compile", compile_block)
    compile_transformer_blocks_with_config(
        model,
        CompileConfig(enabled=True, mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False),
    )

    assert calls == [
        (
            block,
            {
                "backend": "inductor",
                "fullgraph": False,
                "dynamic": False,
                "mode": "max-autotune-no-cudagraphs",
            },
        )
    ]


def test_swiftvr_transformer_tf_kernel_fp8_quantization_uses_block_linears(monkeypatch) -> None:
    calls = []

    def count_linear_layers(module, *, module_filter=None):
        calls.append(("count", module, module_filter))
        return 3

    def enable_fp8_gemm(module, *, options, module_filter=None):
        calls.append(("enable", module, options, module_filter))
        return module

    monkeypatch.setattr("telefuser.ops.fp8_gemm.count_linear_layers", count_linear_layers)
    monkeypatch.setattr("telefuser.ops.fp8_gemm.enable_fp8_gemm", enable_fp8_gemm)
    model = SwiftVRWanTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=4,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        cross_attn_norm=False,
    )

    model.enable_quant(
        QuantConfig(
            enabled=True,
            quant_type=QuantType.FP8,
            kernel_backend=QuantKernelBackend.TF_KERNEL,
        )
    )

    assert [call[0] for call in calls] == ["count", "enable"]
    assert model.tf_kernel_fp8_replaced_linear == 3
    assert model.quant_type is QuantType.FP8
    options = calls[1][2]
    assert options.fp16_weight_storage == "discard"
    assert options.materialize_fp8_on_wrap is True
    module_filter = calls[0][2]
    assert module_filter("blocks.0.ffn.net.0.proj", torch.nn.Linear(4, 4))
    assert not module_filter("patch_embedding", torch.nn.Linear(4, 4))
