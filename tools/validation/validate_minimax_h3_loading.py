# SPDX-License-Identifier: Apache-2.0
"""Strictly load one original MiniMax H3 component through ModuleManager."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from telefuser.core.module_manager import ModuleManager
from telefuser.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
from telefuser.models.minimax_h3_dit import MiniMaxH3DiT
from telefuser.models.minimax_h3_encoder import MiniMaxH3Encoder
from telefuser.models.minimax_h3_video_vae import MiniMaxH3VideoVAE


def _shards(component: Path) -> list[str]:
    paths = sorted(str(path) for path in component.glob("model-*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint shards found in {component}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument("--partition", choices=("FL2VA", "Ref2VA"), required=True)
    parser.add_argument("--component", choices=("transformer", "text_encoder", "video_vae", "audio_vae"), required=True)
    args = parser.parse_args()

    root = Path(args.model_root) / args.partition
    specifications = {
        "transformer": (
            _shards(root / "transformer"),
            MiniMaxH3DiT,
            root / "transformer" / "config.json",
            torch.bfloat16,
        ),
        "text_encoder": (
            _shards(root / "text_encoder"),
            MiniMaxH3Encoder,
            root / "text_encoder",
            torch.bfloat16,
        ),
        "video_vae": (
            str(root / "video_vae" / "source" / "model.safetensors"),
            MiniMaxH3VideoVAE,
            root / "video_vae",
            torch.float32,
        ),
        "audio_vae": (
            str(root / "audio_vae" / "model.safetensors"),
            MiniMaxH3AudioVAE,
            root / "audio_vae",
            torch.float32,
        ),
    }
    checkpoint, model_class, config_path, dtype = specifications[args.component]
    name = f"minimax_h3_{args.component}"
    manager = ModuleManager(device="cpu", torch_dtype=dtype)
    manager.load_model(
        checkpoint,
        device="cpu",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        name=name,
        model_class=model_class,
        converter_kwargs={"config_path": config_path},
    )
    model = manager.fetch_module(name)
    if model is None:
        raise RuntimeError(f"ModuleManager did not register {name}")
    meta_parameters = [parameter_name for parameter_name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"component retained meta parameters: {meta_parameters[:5]}")
    payload = {
        "partition": args.partition,
        "component": args.component,
        "state_dict_keys": len(model.state_dict()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(dtype),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
