import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_data
from metrics import denormalize_quality, regression_metrics
from model import FullOpticalIQARegressor, soft_quality_targets
from train import build_optimizer


def _small_model_config():
    return {
        "optics": {
            "input_size": 24,
            "canvas_size": 32,
            "num_layers": 6,
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 16e-6,
            "phase_param": "sigmoid",
            "phase_init": "zeros",
            "input_to_layer_distance_m": 0.0,
            "inter_layer_distance_m": 0.001,
            "detector_distance_m": 0.001,
        },
        "detector": {
            "quality_anchor_count": 10,
            "detector_size": 2,
            "layout": "fixed_2x2",
            "start_pos_x_per_row": [12, 12, 10],
            "start_pos_y": 9,
            "N_det_sets": [3, 3, 4],
            "det_steps_x": [2, 2, 1],
            "det_steps_y": 2,
            "normalize_detector_energy": True,
        },
        "regularization": {"phase_dropout": {"enabled": False}},
    }


def _make_fake_kadid(root):
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for reference_index in range(15):
        reference = f"I{reference_index + 1:02d}.png"
        for distortion_index in range(2):
            name = f"I{reference_index + 1:02d}_{distortion_index + 1:02d}_{distortion_index + 1:02d}.png"
            value = 20 + reference_index * 10 + distortion_index
            Image.fromarray(np.full((16, 20, 3), value, dtype=np.uint8)).save(image_dir / name)
            rows.append({"dist_img": name, "ref_img": reference, "dmos": 1.0 + reference_index / 4.0, "var": 0.1})
    with (root / "dmos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def test_kadid_loader_is_reference_disjoint_and_normalized(tmp_path):
    _make_fake_kadid(tmp_path)
    config = {
        "seed": 42,
        "dataset": {
            "data_root": str(tmp_path), "metadata_csv": "dmos.csv", "image_dir": "images",
            "download": False, "quality_score_higher_is_better": True,
            "test_reference_fraction": 0.2, "validation_reference_fraction": 0.2,
            "resize_size": 24,
        },
    }
    bundle = load_data(config)
    train_refs = set(bundle.train.references); validation_refs = set(bundle.validation.references); test_refs = set(bundle.test.references)
    assert not (train_refs & validation_refs or train_refs & test_refs or validation_refs & test_refs)
    assert bundle.metadata["reference_disjoint_train_validation_test"] is True
    image, target, index = bundle.train[0]
    assert image.shape == (1, 24, 24)
    assert 0.0 <= float(target) <= 1.0
    assert isinstance(index, int)


def test_full_optical_regressor_has_no_electronic_trainable_layer():
    model = FullOpticalIQARegressor(_small_model_config())
    prediction, details = model(torch.rand(2, 1, 24, 24), return_intermediates=True)
    assert prediction.shape == (2,)
    assert torch.all((prediction >= 0) & (prediction <= 1))
    assert details["region_probabilities"].shape == (2, 10)
    assert torch.allclose(details["region_probabilities"].sum(1), torch.ones(2), atol=1e-5)
    assert model.optical_parameter_count() == 6 * 32 * 32
    assert model.electronic_parameter_count() == 0
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 6 * 32 * 32


def test_optimizer_type_supports_adamw():
    model = FullOpticalIQARegressor(_small_model_config())
    config = {"optimizer": {"type": "adamw", "learning_rate": 0.002, "weight_decay": 0.0}}
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)


def test_production_phase_parameter_count():
    config = _small_model_config(); config["optics"].update({"input_size": 300, "canvas_size": 360})
    config["detector"].update({"detector_size": 30, "start_pos_x_per_row": [115, 115, 90], "start_pos_y": 105, "det_steps_x": [20, 20, 20], "det_steps_y": 30})
    model = FullOpticalIQARegressor(config)
    assert model.optical_parameter_count() == 777_600


def test_soft_quality_targets_and_regression_metrics():
    anchors = torch.linspace(0, 1, 10); targets = torch.tensor([0.1, 0.5, 0.9])
    distribution = soft_quality_targets(targets, anchors, 0.08)
    assert distribution.shape == (3, 10)
    assert torch.allclose(distribution.sum(1), torch.ones(3), atol=1e-6)
    metrics = regression_metrics(targets, targets.clone())
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["plcc"] == pytest.approx(1.0)
    assert metrics["srocc"] == pytest.approx(1.0)


def test_score_direction_round_trip():
    normalized = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(denormalize_quality(normalized, 1.0, 5.0, True), torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64))
    assert torch.allclose(denormalize_quality(normalized, 1.0, 5.0, False), torch.tensor([5.0, 3.0, 1.0], dtype=torch.float64))
