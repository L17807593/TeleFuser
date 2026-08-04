"""CPU contracts derived from SGLang test_minimax_h3_admission.py."""

import pytest

from telefuser.pipelines.minimax_h3.data import (
    minimax_h3_validate_canonical_request,
    minimax_h3_validate_reference_media_facts,
)
from telefuser.pipelines.minimax_h3.resolved_plan import minimax_h3_resolve_plan
from telefuser.pipelines.minimax_h3.task_profiles import partition_for_task

TARGET = {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}


@pytest.mark.parametrize(
    ("task", "conditions", "partition", "visual", "audio", "chains"),
    [
        ("t2va", [], "fl2va", [], [], []),
        (
            "fl2va",
            [
                {"type": "image", "uri": "file:///first.png", "role": "keyframe", "frame_index": 0},
                {"type": "image", "uri": "file:///last.png", "role": "keyframe", "frame_index": -1},
            ],
            "fl2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.target_canvas"],
        ),
        (
            "ref2va",
            [
                {"type": "image", "uri": "file:///image.png", "role": "reference"},
                {
                    "type": "video",
                    "uri": "file:///video.mp4",
                    "role": "reference",
                    "start_time_seconds": 12.5,
                },
                {"type": "audio", "uri": "file:///audio.wav", "role": "reference"},
                {"type": "video_audio", "uri": "file:///av.mp4", "role": "reference"},
            ],
            "ref2va",
            [0, 1, 3],
            [1, 2, 3],
            [
                "image.reference_preserve",
                "video.reference_preserve",
                "audio",
                "video_audio.reference_preserve",
            ],
        ),
    ],
)
def test_public_tasks_resolve_to_exact_partition_and_encoder_plan(
    task: str,
    conditions: list[dict[str, object]],
    partition: str,
    visual: list[int],
    audio: list[int],
    chains: list[str],
) -> None:
    canonical = minimax_h3_validate_canonical_request(
        task=task, prompt="contract", conditions=conditions, target=TARGET, seed=0
    )
    plan = minimax_h3_resolve_plan(canonical)

    assert partition_for_task(task) == partition
    assert plan.task == task
    assert plan.encoders["visual"] == visual
    assert plan.encoders["audio"] == audio
    assert [material.material_chain for material in plan.materials] == chains
    assert plan.shape["frame_count"] == 124
    assert plan.shape["video_latent_t"] == 37


def test_fl2va_signatures_preserve_first_and_last_semantics() -> None:
    for indices in ([0], [-1], [0, -1]):
        conditions = [
            {"type": "image", "uri": f"file:///{index}.png", "role": "keyframe", "frame_index": index}
            for index in indices
        ]
        plan = minimax_h3_resolve_plan(
            minimax_h3_validate_canonical_request(
                task="fl2va", prompt="contract", conditions=conditions, target=TARGET, seed=0
            )
        )
        assert plan.condition_mask["semantic_frame_indices"] == indices
        assert plan.condition_mask["pixel_frame_indices"] == [123 if index == -1 else index for index in indices]


@pytest.mark.parametrize("duration", [3.9, 15.1])
def test_duration_outside_release_range_fails_before_model_execution(duration: float) -> None:
    with pytest.raises(ValueError, match=r"\[4, 15\]"):
        minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="contract",
            conditions=[],
            target={**TARGET, "duration_seconds": duration},
            seed=0,
        )


def test_ref2va_order_and_audio_duration_source_are_preserved() -> None:
    conditions = [
        {"type": "image", "uri": "file:///subject.png", "role": "reference"},
        {"type": "audio", "uri": "file:///voice.wav", "role": "reference"},
    ]
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="contract",
        conditions=conditions,
        target={"short_edge": 768, "aspect_ratio": "auto"},
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    assert [material.condition_type for material in plan.materials] == ["image", "audio"]
    assert plan.shape["temporal"] == "deferred_from_audio_reference"


def test_invalid_material_combinations_report_field_paths() -> None:
    with pytest.raises(ValueError, match=r"conditions\[0\]"):
        minimax_h3_validate_canonical_request(
            task="fl2va",
            prompt="contract",
            conditions=[{"type": "audio", "uri": "file:///x.wav", "role": "reference"}],
            target=TARGET,
            seed=0,
        )


def test_ref2va_release_count_limits_and_audio_companion_rule() -> None:
    images = [{"type": "image", "uri": f"file:///{index}.png", "role": "reference"} for index in range(10)]
    with pytest.raises(ValueError, match="at most 9 image"):
        minimax_h3_validate_canonical_request(
            task="ref2va", prompt="contract", conditions=images, target=TARGET, seed=0
        )
    with pytest.raises(ValueError, match="require at least one image or video"):
        minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="contract",
            conditions=[{"type": "audio", "uri": "file:///voice.wav", "role": "reference"}],
            target=TARGET,
            seed=0,
        )


def test_ref2va_probed_clip_and_total_duration_limits() -> None:
    conditions = [
        {"type": "image", "uri": "file:///subject.png", "role": "reference"},
        {"type": "video", "uri": "file:///one.mp4", "role": "reference"},
        {"type": "video", "uri": "file:///two.mp4", "role": "reference"},
        {"type": "audio", "uri": "file:///voice.wav", "role": "reference"},
    ]
    minimax_h3_validate_reference_media_facts(conditions, {1: 7.5, 2: 7.5, 3: 2.0})
    with pytest.raises(ValueError, match="video total duration"):
        minimax_h3_validate_reference_media_facts(conditions, {1: 8.0, 2: 7.1, 3: 2.0})
    with pytest.raises(ValueError, match=r"conditions\[3\] duration"):
        minimax_h3_validate_reference_media_facts(conditions, {1: 7.5, 2: 7.5, 3: 1.9})
