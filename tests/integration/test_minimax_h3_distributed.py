from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.models.minimax_h3_dit import MiniMaxH3DiT, MiniMaxH3DiTConfig
from telefuser.worker import ParallelWorker


def _small_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )


def _inputs() -> dict[str, object]:
    sequence = 64
    return {
        "x": torch.randn(1, sequence, 8),
        "audio_x": torch.randn(1, sequence, 2),
        "img_position_ids": torch.zeros(1, sequence, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(sequence, dtype=torch.long),
        "update_mask": torch.ones(32, dtype=torch.bool),
        "update_audio_mask": torch.ones(16, dtype=torch.bool),
        "token_tags": torch.cat(
            (
                torch.ones(16, dtype=torch.long),
                torch.full((16,), 2, dtype=torch.long),
                torch.zeros(32, dtype=torch.long),
            )
        ),
        "prompt_embeds": torch.randn(16, 16),
        "img_pos_info": {"position_ids": torch.arange(32, 64)},
        "audio_pos_info": {"position_ids": torch.arange(16, 32)},
        "text_pos_info": {"position_ids": torch.arange(0, 16)},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(32, 64)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 32, 64], dtype=torch.int32)},
    }


class _MiniMaxH3UlyssesParityStage(BaseStage):
    def __init__(self, degree: int) -> None:
        super().__init__(
            "minimax-h3-ulysses-parity",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.bfloat16,
                parallel_config=ParallelConfig(device_ids=list(range(degree)), sp_ulysses_degree=degree),
            ),
        )
        torch.manual_seed(17)
        source = MiniMaxH3DiT(_small_config()).eval()
        self.dense = deepcopy(source)
        self.parallel = deepcopy(source)
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        self.parallel = self.parallel.to(self.device)
        if torch.distributed.get_rank() == 0:
            self.dense = self.dense.to(self.device)
        mesh = create_device_mesh_from_config(self.model_runtime_config.parallel_config)
        self.parallel.enable_usp(mesh)

    def compare(self, inputs: dict[str, object]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        parallel = tuple(tensor.cpu() for tensor in self.parallel(**inputs))
        dense: tuple[torch.Tensor, ...] = ()
        if torch.distributed.get_rank() == 0:
            dense = tuple(tensor.cpu() for tensor in self.dense(**inputs))
        return dense, parallel


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.parametrize("degree", [2, 4])
def test_minimax_h3_ulysses_matches_dense_packed_forward(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} CUDA devices")
    torch.manual_seed(29)
    worker = ParallelWorker(_MiniMaxH3UlyssesParityStage(degree))
    try:
        dense, parallel = worker.compare(_inputs(), sync=True)
    finally:
        worker.close()
    assert len(dense) == len(parallel) == 2
    for actual, expected in zip(parallel, dense, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
