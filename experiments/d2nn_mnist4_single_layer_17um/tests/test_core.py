from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.d2nn_mnist4_single_layer_17um.ccd_evaluate import (
    evaluate_directory,
)
from experiments.d2nn_mnist4_single_layer_17um.data import build_datasets
from experiments.d2nn_mnist4_single_layer_17um.hardware_export import (
    _full_amplitude_frame,
)
from experiments.d2nn_mnist4_single_layer_17um.io_utils import write_csv
from experiments.d2nn_mnist4_single_layer_17um.modeling import (
    SingleLayerMNIST4D2NN,
)
from experiments.d2nn_mnist4_single_layer_17um.settings import load_settings


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "mnist4_single_layer_17um_5cm.yaml"
)


def _settings():
    return replace(load_settings(CONFIG), num_workers=0)


def test_raw_phase_starts_at_zero_and_actual_phase_is_pi() -> None:
    model = SingleLayerMNIST4D2NN(_settings())
    assert torch.count_nonzero(model.raw_phase) == 0
    torch.testing.assert_close(model.phase(), torch.full_like(model.raw_phase, torch.pi))


def test_detector_geometry_matches_four_focus_physical_centers() -> None:
    settings = _settings()
    assert settings.detector_bounds() == (
        (95, 95, 144, 144),
        (334, 95, 383, 144),
        (95, 334, 144, 383),
        (334, 334, 383, 383),
    )
    detector_spacing_um = 239 * 17.0
    phase_lens_spacing_um = 508 * 8.0
    assert detector_spacing_um == 4063.0
    assert phase_lens_spacing_um == 4064.0
    assert abs(detector_spacing_um - phase_lens_spacing_um) == 1.0


def test_smoke_limits_remain_class_balanced() -> None:
    settings = replace(
        _settings(), train_limit=16, val_limit=8, test_limit=8
    )
    bundle = build_datasets(settings)
    for split in ("train", "validation", "test"):
        counts = [bundle.metadata["per_class"][str(label)][split] for label in range(4)]
        assert max(counts) - min(counts) <= 1


def test_forward_has_finite_nonzero_raw_phase_gradient() -> None:
    torch.manual_seed(3)
    model = SingleLayerMNIST4D2NN(_settings())
    images = torch.rand(2, 1, 400, 400)
    targets = torch.tensor([0, 3])
    output = model(images)
    assert output["detector_intensity"].shape == (2, 478, 478)
    assert output["detector_fraction"].shape == (2, 4)
    loss, classification, capture = model.optical_routing_loss(output, targets)
    assert torch.isfinite(classification)
    assert torch.isfinite(capture)
    loss.backward()
    assert model.raw_phase.grad is not None
    assert torch.isfinite(model.raw_phase.grad).all()
    assert torch.count_nonzero(model.raw_phase.grad) > 0


def test_normal_polarity_amplitude_export_contract() -> None:
    settings = _settings()
    active = np.zeros((478, 478), dtype=np.float32)
    active[239, 239] = 1.0
    frame, bounds = _full_amplitude_frame(active, settings)
    assert frame.shape == (1024, 1024)
    assert bounds == (273, 273, 751, 751)
    assert frame[0, 0] == 0
    assert frame[273 + 239, 273 + 239] == 255


def test_ccd_evaluator_recovers_four_detector_classes(tmp_path: Path) -> None:
    settings = _settings()
    ccd_dir = tmp_path / "ccd"
    ccd_dir.mkdir()
    manifest_rows = []
    for label, (left, top, right, bottom) in enumerate(settings.detector_bounds()):
        key = f"sample_{label}"
        array = np.zeros((478, 478), dtype=np.uint8)
        array[top:bottom, left:right] = 255
        Image.fromarray(array, mode="L").save(ccd_dir / f"{key}.png")
        manifest_rows.append({"key": key, "label": label})
    manifest = tmp_path / "samples.csv"
    write_csv(manifest, manifest_rows)
    summary = evaluate_directory(
        config=CONFIG,
        manifest=manifest,
        ccd_dir=ccd_dir,
        output_dir=tmp_path / "result",
        roi=None,
        flip_vertical=False,
        flip_horizontal=False,
    )
    assert summary["accuracy"] == 1.0
    assert summary["background_subtraction"] is False
