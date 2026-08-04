import json
from pathlib import Path

import pytest

from examples.minimax_h3.common import (
    MINIMAX_H3_DEFAULT_FL2VA_IMAGE,
    MINIMAX_H3_DEFAULT_REF2VA_AUDIO,
    MINIMAX_H3_DEFAULT_REF2VA_VIDEO,
    load_minimax_h3_request,
    partition_for_minimax_h3_request,
)
from examples.minimax_h3.minimax_h3_fl2va_h100 import build_fl2va_conditions
from examples.minimax_h3.minimax_h3_ref2va_h100 import default_ref2va_conditions


def test_fl2va_example_builds_every_public_keyframe_signature() -> None:
    assert build_fl2va_conditions(mode="t2va", image=None, last_image=None) == []
    assert [item["frame_index"] for item in build_fl2va_conditions(mode="first-frame", image="a", last_image=None)] == [
        0
    ]
    assert [item["frame_index"] for item in build_fl2va_conditions(mode="last-frame", image=None, last_image="b")] == [
        -1
    ]
    assert [item["frame_index"] for item in build_fl2va_conditions(mode="first-last", image="a", last_image="b")] == [
        0,
        -1,
    ]


def test_fl2va_example_keeps_legacy_mode_inference_and_accepts_last_only() -> None:
    assert build_fl2va_conditions(mode=None, image=None, last_image=None) == []
    assert build_fl2va_conditions(mode=None, image=None, last_image="last.png")[0]["frame_index"] == -1
    with pytest.raises(ValueError, match="does not accept"):
        build_fl2va_conditions(mode="t2va", image="first.png", last_image=None)


def test_default_materials_are_source_controlled_example_inputs() -> None:
    assert MINIMAX_H3_DEFAULT_FL2VA_IMAGE.is_file()
    assert MINIMAX_H3_DEFAULT_REF2VA_VIDEO.is_file()
    assert MINIMAX_H3_DEFAULT_REF2VA_AUDIO.is_file()
    assert [item["type"] for item in default_ref2va_conditions()] == ["video", "audio"]


def test_request_loader_resolves_relative_materials_without_reordering(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "ref2va",
                "prompt": "preserve order",
                "conditions": [
                    {"type": "audio", "role": "reference", "uri": "voice.mp3"},
                    {"type": "video", "role": "reference", "uri": "https://example.com/reference.mp4"},
                ],
                "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
            }
        ),
        encoding="utf-8",
    )

    request = load_minimax_h3_request(request_path)

    assert request["conditions"][0]["type"] == "audio"
    assert request["conditions"][0]["uri"] == str(tmp_path / "voice.mp3")
    assert request["conditions"][1]["type"] == "video"
    assert request["conditions"][1]["uri"] == "https://example.com/reference.mp4"
    assert partition_for_minimax_h3_request(request) == "REF2VA"


def test_request_loader_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"task": "t2va", "prompt": "move", "target": {}, "unexpected": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_minimax_h3_request(request_path)
