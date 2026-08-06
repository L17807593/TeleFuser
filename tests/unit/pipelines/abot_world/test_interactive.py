from __future__ import annotations

from types import SimpleNamespace

import torch

from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
)


def test_close_interactive_session_clears_all_retained_state() -> None:
    pipeline = ABotWorldInteractivePipeline(device="cpu")
    vae = SimpleNamespace(_feat_cache=[torch.ones(1)], _feat_idx=[3])
    pipeline.vae_stage = SimpleNamespace(vae=vae)
    session = ABotWorldInteractiveSession(
        prompt_emb=torch.ones(1),
        first_frame_latent=torch.ones(1),
        self_cache=[{"k": torch.ones(1)}],
        cross_cache=[{"k": torch.ones(1)}],
        scheduler=object(),
        generator=torch.Generator(device="cpu"),
    )
    pipeline._interactive_session = session

    pipeline.close_interactive_session(session)

    assert session.closed
    assert session.self_cache == []
    assert session.cross_cache == []
    assert vae._feat_cache == []
    assert vae._feat_idx == [0]
    assert pipeline._interactive_session is None
