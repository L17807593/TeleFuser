from pathlib import Path

import pytest

from telefuser.pipelines.minimax_h3 import material_io


def test_probe_preserves_independent_video_and_audio_durations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        material_io,
        "_probe_av",
        lambda _path: {
            "format": {"duration": "10.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "duration": "9.5",
                },
                {"codec_type": "audio", "sample_rate": "48000", "duration": "6.0"},
            ],
        },
    )

    facts = material_io.minimax_h3_probe_material(Path("reference.mp4"), "video")

    assert facts.duration_seconds == 10.0
    assert facts.video_duration_seconds == 9.5
    assert facts.audio_duration_seconds == 6.0
    assert facts.sample_rate == 48_000
    assert facts.has_audio is True
