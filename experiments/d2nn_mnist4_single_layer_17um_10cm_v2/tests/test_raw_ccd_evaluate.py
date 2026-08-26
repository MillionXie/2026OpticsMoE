import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from .. import ccd_evaluate


BOUNDS = (
    (90, 90, 149, 149),
    (329, 90, 388, 149),
    (90, 329, 149, 388),
    (329, 329, 388, 388),
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_demo_stage(root: Path) -> list[dict[str, object]]:
    rows = [
        {
            "key": f"sample_{label}",
            "profile": "demo_topk",
            "selection_policy": "biased_demo",
            "label": label,
        }
        for label in range(4)
    ]
    _write_csv(root / "samples.csv", rows)
    (root / "stage_contract.json").write_text(
        json.dumps(
            {
                "profile": "demo_topk",
                "samples": 4,
                "suitable_for_accuracy_reporting": False,
                "phase_sha256": "test-phase",
                "phase_file": "phase_to_play/test.bmp",
            }
        ),
        encoding="utf-8",
    )
    return rows


def test_raw_ccd_evaluator_uses_unscaled_region_sums(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ccd_evaluate,
        "load_settings",
        lambda _: SimpleNamespace(detector_bounds=lambda: BOUNDS),
    )
    stage = tmp_path / "demo"
    rows = _make_demo_stage(stage)
    captures = stage / "ccd_captured"
    captures.mkdir()
    for row in rows:
        label = int(row["label"])
        frame = np.zeros((478, 478), dtype=np.uint8)
        left, top, right, bottom = BOUNDS[label]
        frame[top:bottom, left:right] = 200
        Image.fromarray(frame, mode="L").save(captures / f"{row['key']}.png")
    output = stage / "evaluation"
    report = ccd_evaluate.evaluate_directory(
        config=tmp_path / "unused.yaml",
        manifest=stage / "samples.csv",
        ccd_dir=captures,
        output_dir=output,
        allow_biased_demo_metric=True,
    )
    assert report["demo_success_rate"] == 1.0
    assert report["normalization"] is False
    assert report["nonlinearity"] is False
    assert report["background_subtraction"] is False
    predictions = list(
        csv.DictReader(
            (output / "hardware_predictions_raw.csv").open(encoding="utf-8")
        )
    )
    assert float(predictions[0]["raw_energy_0"]) == 200.0 * 59.0 * 59.0


def test_raw_ccd_evaluator_refuses_resize(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ccd_evaluate,
        "load_settings",
        lambda _: SimpleNamespace(detector_bounds=lambda: BOUNDS),
    )
    stage = tmp_path / "demo"
    rows = _make_demo_stage(stage)
    captures = stage / "ccd_captured"
    captures.mkdir()
    for row in rows:
        Image.new("L", (477, 478), color=0).save(captures / f"{row['key']}.png")
    with pytest.raises(ValueError, match="Refusing resize"):
        ccd_evaluate.evaluate_directory(
            config=tmp_path / "unused.yaml",
            manifest=stage / "samples.csv",
            ccd_dir=captures,
            output_dir=stage / "evaluation",
            allow_biased_demo_metric=True,
        )


def test_frame_qc_flags_black_saturated_and_flat_frames() -> None:
    black = np.zeros((478, 478), dtype=np.float32)
    black_report = ccd_evaluate._frame_qc(black, [0.0, 0.0, 0.0, 0.0])
    assert black_report["valid"] is False
    assert "near_black_mean_le_1" in black_report["reasons"]
    assert "four_roi_relative_spread_le_2pct" in black_report["reasons"]

    saturated = np.full((478, 478), 255.0, dtype=np.float32)
    saturated_report = ccd_evaluate._frame_qc(
        saturated, [100.0, 50.0, 25.0, 10.0]
    )
    assert saturated_report["valid"] is False
    assert "saturated_pixels_ge_5pct" in saturated_report["reasons"]

    valid = np.full((478, 478), 5.0, dtype=np.float32)
    valid_report = ccd_evaluate._frame_qc(valid, [100.0, 20.0, 10.0, 5.0])
    assert valid_report["valid"] is True


def test_quick40_contract_is_diagnostic_without_being_biased_demo() -> None:
    rows = [
        {"key": f"sample_{label}_{index}", "label": str(label)}
        for label in range(4)
        for index in range(10)
    ]
    assert ccd_evaluate._is_quick40_diagnostic(rows, "quick40", False)
    assert not ccd_evaluate._is_quick40_diagnostic(rows, "demo_topk", False)
