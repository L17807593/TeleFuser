import math

import torch

from tools.validation.compare_minimax_h3_trajectories import _at, _mappings, _tensor_metrics


def test_tensor_metrics_reports_exact_tensor() -> None:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    metrics = _tensor_metrics(tensor, tensor.clone())
    assert metrics["comparable"] is True
    assert metrics["exact_fraction"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["cosine_similarity"] == 1.0
    assert math.isinf(metrics["psnr_db_reference_range"])


def test_tensor_metrics_reports_numeric_drift() -> None:
    reference = torch.tensor([0.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 2.0], dtype=torch.float32)
    metrics = _tensor_metrics(reference, candidate)
    assert metrics["comparable"] is True
    assert metrics["exact_fraction"] == 0.5
    assert metrics["max_abs_error"] == 1.0
    assert metrics["mean_abs_error"] == 0.5
    assert metrics["rmse"] == math.sqrt(0.5)


def test_tensor_metrics_rejects_shape_mismatch() -> None:
    metrics = _tensor_metrics(torch.zeros(2), torch.zeros(3))
    assert metrics == {
        "reference_shape": [2],
        "candidate_shape": [3],
        "reference_dtype": "torch.float32",
        "candidate_dtype": "torch.float32",
        "comparable": False,
        "reason": "shape_mismatch",
    }


def test_at_concatenates_ref2va_condition_segments() -> None:
    artifact = {"video_audio": torch.tensor([[1.0]]), "audio": torch.tensor([[2.0]])}
    assert torch.equal(_at(artifact, ("video_audio", "audio")), torch.tensor([[1.0], [2.0]]))


def test_ref2va_mappings_use_global_tags_and_full_audio_prediction() -> None:
    mappings = {item["name"]: item for item in _mappings("ref2va")}
    assert mappings["packed.token_tags"]["candidate_path"] == "transformer_layout.block_token_tags"
    assert mappings["processor.pixel_values_videos"]["expected_relation"] == "exact"
    assert mappings["processor.video_grid_thw"]["expected_relation"] == "exact"
    assert mappings["condition_vae.audio_pre_noise"]["candidate_path"] == (
        "conditions_pre_noise.0.audio_rows",
        "conditions_pre_noise.1.audio_rows",
    )
    assert mappings["step.0.dit_prediction.audio"]["candidate_path"] == "steps.0.dit_output.1"
