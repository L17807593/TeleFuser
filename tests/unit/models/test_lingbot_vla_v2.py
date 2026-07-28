from types import SimpleNamespace

import torch

from telefuser.models.lingbot_vla_v2 import QwenvlWithExpertV2Model


class _Visual:
    spatial_merge_size = 1

    def __init__(self) -> None:
        self.preprocess_calls = 0

    def preprcess_grid_thw(self, grid_thw: torch.Tensor):
        self.preprocess_calls += 1
        token_count = int(grid_thw.prod(dim=-1).sum())
        position_embeddings = (torch.zeros(token_count, 2), torch.ones(token_count, 2))
        cu_seqlens = torch.tensor([0, token_count], dtype=torch.int32)
        split_sizes = grid_thw.prod(dim=-1).tolist()
        return None, position_embeddings, cu_seqlens, split_sizes, token_count

    def __call__(self, pixel_values: torch.Tensor, **kwargs):
        del kwargs
        embeddings = torch.zeros(pixel_values.shape[0], 3)
        return embeddings, [embeddings.clone()]


def _model(visual: _Visual) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(precompute_grid_thw=True),
        qwenvl=SimpleNamespace(visual=visual),
        pos_embeds=None,
        position_embeddings=None,
        cu_seqlens=None,
        visual_split_sizes=None,
        visual_max_seqlen=None,
        _cached_image_grid_signature=None,
    )


def test_image_grid_cache_is_reused_and_invalidated_by_grid_shape() -> None:
    visual = _Visual()
    model = _model(visual)
    first_grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    second_grid = torch.tensor([[1, 1, 2], [1, 1, 2]])

    first = QwenvlWithExpertV2Model.get_image_features(model, torch.zeros(8, 6), first_grid)
    repeated = QwenvlWithExpertV2Model.get_image_features(model, torch.zeros(8, 6), first_grid.clone())
    changed = QwenvlWithExpertV2Model.get_image_features(model, torch.zeros(4, 6), second_grid)

    assert visual.preprocess_calls == 2
    assert first[0].shape == repeated[0].shape == (2, 4, 3)
    assert changed[0].shape == (2, 2, 3)
