from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.minimax_h3.denoising import MiniMaxH3DenoisingStage


def _stage(parallel_config: ParallelConfig) -> tuple[MiniMaxH3DenoisingStage, MagicMock]:
    transformer = MagicMock()
    transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))]
    transformer.get_fsdp_module_names.return_value = ["blocks"]
    manager = MagicMock()
    manager.fetch_module.return_value = transformer
    runtime = ModelRuntimeConfig(
        device_type="cuda",
        device_id=0,
        torch_dtype=torch.bfloat16,
        parallel_config=parallel_config,
    )
    return MiniMaxH3DenoisingStage(manager, runtime), transformer


def test_parallel_models_enables_ulysses_and_preserves_fp32_fsdp_parameters() -> None:
    stage, transformer = _stage(ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2, enable_fsdp=True))
    device_mesh = MagicMock()
    fsdp_model = MagicMock()
    with (
        patch(
            "telefuser.pipelines.minimax_h3.denoising.create_device_mesh_from_config",
            return_value=device_mesh,
        ),
        patch("telefuser.pipelines.minimax_h3.denoising.shard_model", return_value=fsdp_model) as shard,
    ):
        stage.parallel_models()

    transformer.enable_usp.assert_called_once_with(device_mesh)
    shard.assert_called_once()
    call = shard.call_args.kwargs
    assert call["wrap_module_names"] == ["blocks"]
    assert call["buffer_dtype"] == torch.float32
    assert len(call["ignored_states"]) == 1
    assert stage.transformer is fsdp_model


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("cfg_degree", {"cfg_degree": 2}),
        ("sp_ring_degree", {"sp_ring_degree": 2}),
        ("pp_degree", {"pp_degree": 2}),
        ("tp_degree", {"tp_degree": 2}),
    ],
)
def test_parallel_models_rejects_unsupported_degrees(field: str, kwargs: dict[str, int]) -> None:
    stage, _ = _stage(ParallelConfig(device_ids=[0, 1], **kwargs))
    with pytest.raises(NotImplementedError, match=field):
        stage.parallel_models()
