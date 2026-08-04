# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from tools.validation.compare_minimax_h3_artifacts import (
    _audio_temporal_metrics,
    _best_lag_samples,
    _compare_video,
)


def _save(path: Path, value: np.ndarray) -> Path:
    np.save(path, value)
    return path


def test_compare_video_records_each_frame_and_worst_frame(tmp_path: Path) -> None:
    reference = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    candidate = reference.copy()
    candidate[1] += 1

    report = _compare_video(_save(tmp_path / "reference.npy", reference), _save(tmp_path / "candidate.npy", candidate))

    assert len(report["per_frame"]) == 2
    assert report["per_frame"][0]["exact_fraction"] == 1.0
    assert report["per_frame_summary"]["max_rmse_frame_index"] == 1
    assert report["per_frame_summary"]["min_psnr_frame_index"] == 1


def test_audio_temporal_metrics_record_exact_windows_and_zero_lag(tmp_path: Path) -> None:
    sample_rate = 2048
    time = np.arange(2 * sample_rate, dtype=np.float32) / sample_rate
    signal = np.stack((np.sin(2 * np.pi * 120 * time), np.sin(2 * np.pi * 240 * time))).astype(np.float32)
    reference_path = _save(tmp_path / "reference.npy", signal)
    candidate_path = _save(tmp_path / "candidate.npy", signal.copy())

    report = _audio_temporal_metrics(reference_path, candidate_path, sample_rate=sample_rate)

    assert len(report["per_second"]) == 2
    assert report["summary"]["min_cosine_similarity"] == pytest.approx(1.0)
    assert report["summary"]["min_log_spectral_cosine"] == pytest.approx(1.0)
    assert report["summary"]["energy_envelope_correlation"] == pytest.approx(1.0)
    assert report["summary"]["best_lag_samples"] == 0


def test_best_lag_samples_reports_candidate_delay() -> None:
    rng = np.random.default_rng(0)
    reference = rng.standard_normal(8192)
    candidate = np.concatenate((np.zeros(64), reference[:-64]))
    reference_before = reference.copy()
    candidate_before = candidate.copy()

    assert _best_lag_samples(reference, candidate, sample_rate=2048) == 64
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(candidate, candidate_before)
