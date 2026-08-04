# SPDX-License-Identifier: Apache-2.0
"""Compare raw MiniMax H3 SGLang and TeleFuser output arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MIN_VIDEO_COSINE = 0.99
MIN_VIDEO_PSNR_DB = 28.0
MIN_AUDIO_COSINE = 0.94
MIN_AUDIO_PSNR_DB = 30.0
DEFAULT_AUDIO_SAMPLE_RATE = 32_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(reference_path: Path, candidate_path: Path, *, data_range: float) -> dict[str, object]:
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    count = reference_flat.size
    chunk_size = 4 * 1024 * 1024
    max_abs = 0.0
    sum_abs = 0.0
    sum_square = 0.0
    sum_relative = 0.0
    dot = 0.0
    reference_square = 0.0
    candidate_square = 0.0
    exact = 0
    relative_floor = 1.0 if np.issubdtype(reference.dtype, np.integer) else 1e-6
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        reference_chunk = reference_flat[start:stop].astype(np.float64)
        candidate_chunk = candidate_flat[start:stop].astype(np.float64)
        difference = candidate_chunk - reference_chunk
        absolute = np.abs(difference)
        max_abs = max(max_abs, float(absolute.max(initial=0.0)))
        sum_abs += float(absolute.sum())
        sum_square += float(np.square(difference).sum())
        sum_relative += float((absolute / np.maximum(np.abs(reference_chunk), relative_floor)).sum())
        dot += float(np.dot(reference_chunk, candidate_chunk))
        reference_square += float(np.square(reference_chunk).sum())
        candidate_square += float(np.square(candidate_chunk).sum())
        exact += int(np.equal(reference_chunk, candidate_chunk).sum())
    rmse = math.sqrt(sum_square / count)
    if reference_square == 0.0 and candidate_square == 0.0:
        cosine = 1.0
    elif reference_square == 0.0 or candidate_square == 0.0:
        cosine = 0.0
    else:
        cosine = dot / math.sqrt(reference_square * candidate_square)
    psnr = math.inf if rmse == 0.0 else 20.0 * math.log10(data_range / rmse)
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "reference_sha256": _sha256(reference_path),
        "candidate_sha256": _sha256(candidate_path),
        "max_abs_error": max_abs,
        "mean_abs_error": sum_abs / count,
        "mean_relative_error": sum_relative / count,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "exact_fraction": exact / count,
        "psnr_db": psnr,
    }


def _array_metrics(reference: np.ndarray, candidate: np.ndarray, *, data_range: float) -> dict[str, float]:
    reference_flat = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference_flat.shape != candidate_flat.shape:
        raise ValueError(f"shape mismatch: {reference_flat.shape} != {candidate_flat.shape}")
    difference = candidate_flat - reference_flat
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_norm = float(np.linalg.norm(reference_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine = 1.0
    elif reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(reference_flat, candidate_flat) / (reference_norm * candidate_norm))
    return {
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "cosine_similarity": cosine,
        "exact_fraction": float(np.mean(reference_flat == candidate_flat)),
        "psnr_db": math.inf if rmse == 0.0 else 20.0 * math.log10(data_range / rmse),
    }


def _compare_video(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    result = _compare(reference_path, candidate_path, data_range=255.0)
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    per_frame = []
    for frame_index in range(reference.shape[0]):
        metrics = _array_metrics(reference[frame_index], candidate[frame_index], data_range=255.0)
        per_frame.append({"frame_index": frame_index, **metrics})
    result["per_frame"] = per_frame
    result["per_frame_summary"] = {
        "min_cosine_similarity": min(item["cosine_similarity"] for item in per_frame),
        "min_cosine_frame_index": min(per_frame, key=lambda item: item["cosine_similarity"])["frame_index"],
        "min_psnr_db": min(item["psnr_db"] for item in per_frame),
        "min_psnr_frame_index": min(per_frame, key=lambda item: item["psnr_db"])["frame_index"],
        "max_rmse": max(item["rmse"] for item in per_frame),
        "max_rmse_frame_index": max(per_frame, key=lambda item: item["rmse"])["frame_index"],
    }
    return result


def _log_spectral_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    n_fft = 1024
    hop_length = 256
    if reference.size < n_fft:
        pad = n_fft - reference.size
        reference = np.pad(reference, (0, pad))
        candidate = np.pad(candidate, (0, pad))
    window = np.hanning(n_fft)
    reference_frames = np.lib.stride_tricks.sliding_window_view(reference, n_fft)[::hop_length]
    candidate_frames = np.lib.stride_tricks.sliding_window_view(candidate, n_fft)[::hop_length]
    reference_spectrum = np.log1p(np.abs(np.fft.rfft(reference_frames * window, axis=-1))).reshape(-1)
    candidate_spectrum = np.log1p(np.abs(np.fft.rfft(candidate_frames * window, axis=-1))).reshape(-1)
    reference_norm = float(np.linalg.norm(reference_spectrum))
    candidate_norm = float(np.linalg.norm(candidate_spectrum))
    if reference_norm == 0.0 and candidate_norm == 0.0:
        return 1.0
    if reference_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    return float(np.dot(reference_spectrum, candidate_spectrum) / (reference_norm * candidate_norm))


def _best_lag_samples(reference: np.ndarray, candidate: np.ndarray, *, sample_rate: int) -> int:
    decimation = 8
    reference_decimated = np.asarray(reference, dtype=np.float64)[::decimation].copy()
    candidate_decimated = np.asarray(candidate, dtype=np.float64)[::decimation].copy()
    reference_decimated -= reference_decimated.mean()
    candidate_decimated -= candidate_decimated.mean()
    correlation_size = reference_decimated.size + candidate_decimated.size - 1
    fft_size = 1 << (correlation_size - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(candidate_decimated, fft_size) * np.fft.rfft(reference_decimated[::-1], fft_size),
        fft_size,
    )[:correlation_size]
    lags = np.arange(-(reference_decimated.size - 1), candidate_decimated.size)
    max_lag = sample_rate // decimation
    allowed = np.abs(lags) <= max_lag
    return int(lags[allowed][np.argmax(correlation[allowed])]) * decimation


def _audio_temporal_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    sample_rate: int,
) -> dict[str, object]:
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    if reference.ndim == 1:
        reference = reference[None, :]
        candidate = candidate[None, :]
    reference_channels = reference.reshape(-1, reference.shape[-1])
    candidate_channels = candidate.reshape(-1, candidate.shape[-1])
    sample_count = reference_channels.shape[-1]
    per_second = []
    for window_index, start in enumerate(range(0, sample_count, sample_rate)):
        stop = min(start + sample_rate, sample_count)
        reference_window = reference_channels[:, start:stop]
        candidate_window = candidate_channels[:, start:stop]
        metrics = _array_metrics(reference_window, candidate_window, data_range=2.0)
        reference_mono = np.asarray(reference_window, dtype=np.float64).mean(axis=0)
        candidate_mono = np.asarray(candidate_window, dtype=np.float64).mean(axis=0)
        per_second.append(
            {
                "window_index": window_index,
                "start_sample": start,
                "end_sample": stop,
                "reference_rms": float(np.sqrt(np.mean(np.square(reference_window)))),
                "candidate_rms": float(np.sqrt(np.mean(np.square(candidate_window)))),
                "log_spectral_cosine": _log_spectral_cosine(reference_mono, candidate_mono),
                **metrics,
            }
        )

    envelope_samples = max(1, sample_rate // 50)
    envelope_count = sample_count // envelope_samples
    reference_envelope = np.sqrt(
        np.mean(
            np.square(np.asarray(reference_channels[:, : envelope_count * envelope_samples], dtype=np.float64)).reshape(
                reference_channels.shape[0], envelope_count, envelope_samples
            ),
            axis=(0, 2),
        )
    )
    candidate_envelope = np.sqrt(
        np.mean(
            np.square(np.asarray(candidate_channels[:, : envelope_count * envelope_samples], dtype=np.float64)).reshape(
                candidate_channels.shape[0], envelope_count, envelope_samples
            ),
            axis=(0, 2),
        )
    )
    if np.std(reference_envelope) == 0.0 or np.std(candidate_envelope) == 0.0:
        envelope_correlation = 1.0 if np.array_equal(reference_envelope, candidate_envelope) else 0.0
    else:
        envelope_correlation = float(np.corrcoef(reference_envelope, candidate_envelope)[0, 1])
    reference_mono = np.asarray(reference_channels, dtype=np.float64).mean(axis=0)
    candidate_mono = np.asarray(candidate_channels, dtype=np.float64).mean(axis=0)
    best_lag_samples = _best_lag_samples(reference_mono, candidate_mono, sample_rate=sample_rate)
    return {
        "sample_rate": sample_rate,
        "window_samples": sample_rate,
        "per_second": per_second,
        "summary": {
            "min_cosine_similarity": min(item["cosine_similarity"] for item in per_second),
            "min_cosine_window_index": min(per_second, key=lambda item: item["cosine_similarity"])["window_index"],
            "min_psnr_db": min(item["psnr_db"] for item in per_second),
            "min_log_spectral_cosine": min(item["log_spectral_cosine"] for item in per_second),
            "energy_envelope_correlation": envelope_correlation,
            "best_lag_samples": best_lag_samples,
            "best_lag_ms": 1000.0 * best_lag_samples / sample_rate,
        },
    }


def _compare_audio(reference_path: Path, candidate_path: Path, *, sample_rate: int) -> dict[str, object]:
    result = _compare(reference_path, candidate_path, data_range=2.0)
    result["temporal"] = _audio_temporal_metrics(reference_path, candidate_path, sample_rate=sample_rate)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-frames", type=Path, required=True)
    parser.add_argument("--candidate-frames", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--candidate-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-name", default="sglang")
    parser.add_argument("--candidate-name", default="telefuser")
    parser.add_argument("--min-video-cosine", type=float, default=MIN_VIDEO_COSINE)
    parser.add_argument("--min-video-psnr-db", type=float, default=MIN_VIDEO_PSNR_DB)
    parser.add_argument("--min-audio-cosine", type=float, default=MIN_AUDIO_COSINE)
    parser.add_argument("--min-audio-psnr-db", type=float, default=MIN_AUDIO_PSNR_DB)
    parser.add_argument("--min-frame-cosine", type=float)
    parser.add_argument("--min-frame-psnr-db", type=float)
    parser.add_argument("--min-audio-window-cosine", type=float)
    parser.add_argument("--min-audio-window-psnr-db", type=float)
    parser.add_argument("--min-audio-log-spectral-cosine", type=float)
    parser.add_argument("--min-audio-envelope-correlation", type=float)
    parser.add_argument("--max-audio-lag-ms", type=float)
    parser.add_argument("--audio-sample-rate", type=int, default=DEFAULT_AUDIO_SAMPLE_RATE)
    args = parser.parse_args()

    frames = _compare_video(args.reference_frames, args.candidate_frames)
    audio = _compare_audio(args.reference_audio, args.candidate_audio, sample_rate=args.audio_sample_rate)
    checks = {
        "video_cosine": float(frames["cosine_similarity"]) >= args.min_video_cosine,
        "video_psnr": float(frames["psnr_db"]) >= args.min_video_psnr_db,
        "audio_cosine": float(audio["cosine_similarity"]) >= args.min_audio_cosine,
        "audio_psnr": float(audio["psnr_db"]) >= args.min_audio_psnr_db,
    }
    frame_summary = frames["per_frame_summary"]
    audio_summary = audio["temporal"]["summary"]
    optional_thresholds = {
        "min_frame_cosine": args.min_frame_cosine,
        "min_frame_psnr_db": args.min_frame_psnr_db,
        "min_audio_window_cosine": args.min_audio_window_cosine,
        "min_audio_window_psnr_db": args.min_audio_window_psnr_db,
        "min_audio_log_spectral_cosine": args.min_audio_log_spectral_cosine,
        "min_audio_envelope_correlation": args.min_audio_envelope_correlation,
        "max_audio_lag_ms": args.max_audio_lag_ms,
    }
    optional_checks = {
        "frame_cosine": (
            "min_frame_cosine",
            frame_summary["min_cosine_similarity"],
            lambda actual, limit: actual >= limit,
        ),
        "frame_psnr": ("min_frame_psnr_db", frame_summary["min_psnr_db"], lambda actual, limit: actual >= limit),
        "audio_window_cosine": (
            "min_audio_window_cosine",
            audio_summary["min_cosine_similarity"],
            lambda actual, limit: actual >= limit,
        ),
        "audio_window_psnr": (
            "min_audio_window_psnr_db",
            audio_summary["min_psnr_db"],
            lambda actual, limit: actual >= limit,
        ),
        "audio_log_spectral_cosine": (
            "min_audio_log_spectral_cosine",
            audio_summary["min_log_spectral_cosine"],
            lambda actual, limit: actual >= limit,
        ),
        "audio_envelope_correlation": (
            "min_audio_envelope_correlation",
            audio_summary["energy_envelope_correlation"],
            lambda actual, limit: actual >= limit,
        ),
        "audio_lag": (
            "max_audio_lag_ms",
            abs(audio_summary["best_lag_ms"]),
            lambda actual, limit: actual <= limit,
        ),
    }
    for check_name, (threshold_name, actual, comparison) in optional_checks.items():
        threshold = optional_thresholds[threshold_name]
        if threshold is not None:
            checks[check_name] = comparison(float(actual), threshold)
    report = {
        "schema_version": 3,
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "thresholds": {
            "min_video_cosine": args.min_video_cosine,
            "min_video_psnr_db": args.min_video_psnr_db,
            "min_audio_cosine": args.min_audio_cosine,
            "min_audio_psnr_db": args.min_audio_psnr_db,
            **{name: value for name, value in optional_thresholds.items() if value is not None},
        },
        "checks": checks,
        "passed": all(checks.values()),
        "frames": frames,
        "audio": audio,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    summary = {
        "output": str(args.output),
        "passed": report["passed"],
        "checks": checks,
        "per_frame_summary": frames["per_frame_summary"],
        "audio_temporal_summary": audio["temporal"]["summary"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
