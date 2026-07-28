from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from telefuser.models.lingbot_vla_v2_loader import (
    build_official_6b_config,
    resolve_lingbot_vla_v2_shards,
    validate_official_6b_checkpoint,
)


def test_resolve_lingbot_vla_v2_shards_uses_index_manifest(tmp_path) -> None:
    shard_names = ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"]
    for name in shard_names:
        (tmp_path / name).write_bytes(b"")
    index = {
        "weight_map": {
            "layer.0": shard_names[0],
            "layer.1": shard_names[1],
            "layer.2": shard_names[0],
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    resolved = resolve_lingbot_vla_v2_shards(tmp_path)

    assert resolved == [str(tmp_path / name) for name in sorted(shard_names)]


def test_resolve_lingbot_vla_v2_shards_rejects_missing_files(tmp_path) -> None:
    index = {"weight_map": {"layer.0": "missing.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint shards"):
        resolve_lingbot_vla_v2_shards(tmp_path)


def test_validate_official_6b_checkpoint_accepts_expected_gate_shapes() -> None:
    prefix = "model.qwenvl_with_expert.qwen_expert.model.layers"
    state_dict = {
        f"{prefix}.0.mlp.experts.gate_proj": SimpleNamespace(shape=(32, 512, 768)),
        f"{prefix}.35.mlp.experts.gate_proj": SimpleNamespace(shape=(32, 512, 768)),
    }

    validate_official_6b_checkpoint(state_dict)


def test_validate_official_6b_checkpoint_rejects_wrong_shape() -> None:
    prefix = "model.qwenvl_with_expert.qwen_expert.model.layers"
    state_dict = {
        f"{prefix}.0.mlp.experts.gate_proj": SimpleNamespace(shape=(1, 2, 3)),
        f"{prefix}.35.mlp.experts.gate_proj": SimpleNamespace(shape=(32, 512, 768)),
    }

    with pytest.raises(ValueError, match="Unexpected shape"):
        validate_official_6b_checkpoint(state_dict)


def test_build_official_6b_config_rejects_non_base_variant(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported LingBot-VLA v2 checkpoint variant"):
        build_official_6b_config(tmp_path, checkpoint_variant="robotwin")
