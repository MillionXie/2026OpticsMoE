from __future__ import annotations

import copy
import csv
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.d2nn_mnist4_single_layer_17um_10cm import ccd_evaluate
from experiments.d2nn_mnist4_single_layer_17um_10cm.hardware_profiles import (
    PHASE_FILENAME,
    select_demo_topk,
    select_formal_fixed_random,
)
from experiments.d2nn_mnist4_single_layer_17um_10cm.lab_package import (
    create_lab_zip,
)
from experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline import (
    run_pipeline,
    validate_stage,
)
from experiments.d2nn_mnist4_single_layer_17um_10cm.settings import load_settings


CLASSES = (0, 1, 2, 3)
RELEASE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "mnist4_single_layer_17um_10cm_notebook_mse.yaml"
)


def _candidates(count: int = 150) -> dict[int, list[dict[str, object]]]:
    return {
        label: [
            {
                "dataset_index": label * 1000 + index,
                "label": label,
                "prediction": label if index % 3 else (label + 1) % 4,
                "correct": index % 3 != 0,
                "target_fraction": index / max(1, count),
                "margin": index / max(1, count),
                "detector_fractions": [0.1, 0.2, 0.3, 0.4],
            }
            for index in range(count)
        ]
        for label in CLASSES
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_demo_and_formal_selection_have_distinct_contracts() -> None:
    candidates = _candidates()
    demo = select_demo_topk(candidates, CLASSES, samples_per_class=2)
    assert all(len(demo[label]) == 2 for label in CLASSES)
    assert all(all(item["correct"] for item in demo[label]) for label in CLASSES)

    formal = select_formal_fixed_random(
        candidates,
        CLASSES,
        samples_per_class=100,
        seed=42,
        require_full_count=True,
    )
    repeated = select_formal_fixed_random(
        candidates,
        CLASSES,
        samples_per_class=100,
        seed=42,
        require_full_count=True,
    )
    assert formal == repeated
    assert all(len(formal[label]) == 100 for label in CLASSES)
    assert any(not item["correct"] for values in formal.values() for item in values)


def test_formal_selection_does_not_depend_on_predictions_or_margins() -> None:
    original = _candidates()
    changed = copy.deepcopy(original)
    for values in changed.values():
        for item in values:
            item["prediction"] = (int(item["label"]) + 2) % 4
            item["correct"] = False
            item["margin"] = 999.0 - float(item["margin"])
    selected_original = select_formal_fixed_random(
        original,
        CLASSES,
        samples_per_class=100,
        seed=17,
        require_full_count=True,
    )
    selected_changed = select_formal_fixed_random(
        changed,
        CLASSES,
        samples_per_class=100,
        seed=17,
        require_full_count=True,
    )
    for label in CLASSES:
        assert [item["dataset_index"] for item in selected_original[label]] == [
            item["dataset_index"] for item in selected_changed[label]
        ]


def test_phase_filename_is_explicitly_10cm() -> None:
    assert PHASE_FILENAME == "mnist4_single_layer_17um_10cm.bmp"


def test_release_export_counts_and_polarity_are_enforced() -> None:
    settings = load_settings(RELEASE_CONFIG)
    assert settings.demo_samples_per_class == 10
    assert settings.evaluation_samples_per_class == 100
    assert settings.amplitude_invert_before_export is False
    with pytest.raises(ValueError, match="corrected polarity"):
        replace(settings, amplitude_invert_before_export=True).validate()


def _make_stage(
    stage: Path,
    profile: str = "formal_fixed_random_1_per_class",
    *,
    suitable_for_accuracy_reporting: bool | None = None,
) -> list[dict[str, object]]:
    if suitable_for_accuracy_reporting is None:
        suitable_for_accuracy_reporting = profile.startswith("formal_")
    rows = [
        {
            "key": f"sample_{index}",
            "profile": profile,
            "selection_policy": "fixed_random",
            "label": index,
        }
        for index in range(4)
    ]
    for row in rows:
        path = stage / "amplitude_to_play" / f"{row['key']}.bmp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"amplitude")
        row["amplitude_file"] = path.name
        row["amplitude_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    phase = stage / "phase_to_play" / PHASE_FILENAME
    phase.parent.mkdir(parents=True, exist_ok=True)
    phase.write_bytes(b"phase")
    phase_sha256 = hashlib.sha256(phase.read_bytes()).hexdigest()
    for row in rows:
        row["phase_sha256"] = phase_sha256
    _write_csv(stage / "samples.csv", rows)
    (stage / "README.md").write_text("stage", encoding="utf-8")
    (stage / "stage_contract.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "samples": len(rows),
                "phase_sha256": phase_sha256,
                "phase_file": f"phase_to_play/{PHASE_FILENAME}",
                "suitable_for_accuracy_reporting": (
                    suitable_for_accuracy_reporting
                ),
            }
        ),
        encoding="utf-8",
    )
    return rows


def _make_captures(
    stage: Path,
    rows: list[dict[str, object]],
    arrays: list[np.ndarray],
    *,
    phase_sha256: str | None = None,
) -> Path:
    ccd_dir = stage / "ccd_captured"
    ccd_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads((stage / "stage_contract.json").read_text(encoding="utf-8"))
    phase_sha = phase_sha256 or contract["phase_sha256"]
    capture_rows = []
    ordered_rows = sorted(rows, key=lambda row: str(row["amplitude_file"]))
    arrays_by_key = {
        str(row["key"]): array for row, array in zip(rows, arrays, strict=True)
    }
    for play_index, row in enumerate(ordered_rows):
        key = str(row["key"])
        capture_name = f"{key}.png"
        Image.fromarray(arrays_by_key[key].astype(np.uint8), mode="L").save(
            ccd_dir / capture_name
        )
        capture_rows.append(
            {
                "play_index": play_index,
                "amplitude_bmp": row["amplitude_file"],
                "ccd_capture": capture_name,
                "phase_mask": PHASE_FILENAME,
                "phase_mask_sha256": phase_sha,
            }
        )
    capture_manifest = stage / "acquisition_logs" / "capture_manifest.csv"
    _write_csv(capture_manifest, capture_rows)
    return capture_manifest


def _settings_4x4() -> SimpleNamespace:
    return SimpleNamespace(
        ccd_target_size=4,
        loss_eps=1.0e-8,
        detector_bounds=lambda: (
            (0, 0, 2, 2),
            (2, 0, 4, 2),
            (0, 2, 2, 4),
            (2, 2, 4, 4),
        ),
    )


def _valid_quadrant_frames(settings: SimpleNamespace) -> list[np.ndarray]:
    frames = []
    for left, top, right, bottom in settings.detector_bounds():
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[top:bottom, left:right] = 200
        frames.append(frame)
    return frames


def test_validate_stage_requires_exact_manifest_match(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _make_stage(stage)
    report = validate_stage(stage)
    assert report["samples"] == 4
    assert report["profile"] == "formal_fixed_random_1_per_class"
    (stage / "amplitude_to_play" / "extra.bmp").write_bytes(b"extra")
    try:
        validate_stage(stage)
    except RuntimeError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("An unmanifested amplitude BMP must be rejected")


def test_ccd_evaluation_reports_confusion_and_profile(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_4x4()
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: settings)
    stage = tmp_path / "formal"
    rows = _make_stage(stage)
    _make_captures(stage, rows, _valid_quadrant_frames(settings))
    result = ccd_evaluate.evaluate_directory(
        config=tmp_path / "unused.yaml",
        manifest=stage / "samples.csv",
        ccd_dir=stage / "ccd_captured",
        output_dir=tmp_path / "evaluation",
        roi=None,
        flip_vertical=False,
        flip_horizontal=False,
    )
    assert result["accuracy"] == 1.0
    assert result["invalid_count"] == 0
    assert result["capture_manifest"]["verified"] is True
    assert result["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert result["profiles"] == ["formal_fixed_random_1_per_class"]


def test_demo_evaluation_requires_explicit_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_4x4()
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: settings)
    stage = tmp_path / "demo"
    rows = _make_stage(
        stage, profile="demo_topk", suitable_for_accuracy_reporting=False
    )
    _make_captures(stage, rows, _valid_quadrant_frames(settings))
    arguments = {
        "config": tmp_path / "unused.yaml",
        "manifest": stage / "samples.csv",
        "ccd_dir": stage / "ccd_captured",
        "output_dir": stage / "evaluation",
        "roi": None,
        "flip_vertical": False,
        "flip_horizontal": False,
    }
    with pytest.raises(PermissionError, match="allow-biased-demo-metric"):
        ccd_evaluate.evaluate_directory(**arguments)
    result = ccd_evaluate.evaluate_directory(
        **arguments, allow_biased_demo_metric=True
    )
    assert result["demo_success_rate"] == 1.0
    assert "accuracy" not in result
    assert result["suitable_for_accuracy_reporting"] is False


def test_demo_all_is_rejected_before_opening_devices(tmp_path: Path) -> None:
    stage = tmp_path / "demo"
    _make_stage(stage, profile="demo_topk", suitable_for_accuracy_reporting=False)
    with pytest.raises(PermissionError, match="allow-biased-demo-metric"):
        run_pipeline(
            phase="all",
            stage_dir=stage,
            hardware_config=tmp_path / "must_not_be_opened.yaml",
        )


def test_formal_missing_capture_manifest_has_clear_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: _settings_4x4())
    stage = tmp_path / "formal"
    rows = _make_stage(stage)
    ccd_dir = stage / "ccd_captured"
    ccd_dir.mkdir()
    for row, frame in zip(rows, _valid_quadrant_frames(_settings_4x4()), strict=True):
        Image.fromarray(frame, mode="L").save(ccd_dir / f"{row['key']}.png")
    with pytest.raises(FileNotFoundError, match="Run --phase acquire"):
        ccd_evaluate.evaluate_directory(
            config=tmp_path / "unused.yaml",
            manifest=stage / "samples.csv",
            ccd_dir=ccd_dir,
            output_dir=stage / "evaluation",
            roi=None,
            flip_vertical=False,
            flip_horizontal=False,
        )


def test_formal_capture_manifest_checks_play_count_and_phase_sha(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_4x4()
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: settings)
    stage = tmp_path / "formal"
    rows = _make_stage(stage)
    capture_manifest = _make_captures(
        stage, rows, _valid_quadrant_frames(settings)
    )
    capture_rows = list(csv.DictReader(capture_manifest.open(encoding="utf-8")))
    _write_csv(capture_manifest, capture_rows[:-1])
    arguments = {
        "config": tmp_path / "unused.yaml",
        "manifest": stage / "samples.csv",
        "ccd_dir": stage / "ccd_captured",
        "output_dir": stage / "evaluation",
        "roi": None,
        "flip_vertical": False,
        "flip_horizontal": False,
    }
    with pytest.raises(RuntimeError, match="play count"):
        ccd_evaluate.evaluate_directory(**arguments)
    capture_rows[-1]["phase_mask_sha256"] = "wrong-phase-sha"
    _write_csv(capture_manifest, capture_rows)
    with pytest.raises(RuntimeError, match="phase SHA-256"):
        ccd_evaluate.evaluate_directory(**arguments)


def test_invalid_frames_are_minus_one_and_formal_fails_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_4x4()
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: settings)
    stage = tmp_path / "formal"
    rows = _make_stage(stage)
    frames = _valid_quadrant_frames(settings)
    frames[0] = np.zeros((4, 4), dtype=np.uint8)
    _make_captures(stage, rows, frames)
    output_dir = stage / "evaluation"
    with pytest.raises(RuntimeError, match="invalid CCD"):
        ccd_evaluate.evaluate_directory(
            config=tmp_path / "unused.yaml",
            manifest=stage / "samples.csv",
            ccd_dir=stage / "ccd_captured",
            output_dir=output_dir,
            roi=None,
            flip_vertical=False,
            flip_horizontal=False,
        )
    summary = json.loads(
        (output_dir / "hardware_metrics.json").read_text(encoding="utf-8")
    )
    predictions = list(
        csv.DictReader(
            (output_dir / "hardware_predictions.csv").open(encoding="utf-8")
        )
    )
    assert summary["invalid_count"] == 1
    assert summary["valid_count"] == 3
    assert summary["invalid_by_reason"]["near_black"] == 1
    assert predictions[0]["prediction"] == "-1"


def test_qc_identifies_black_saturated_and_equal_region_frames(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_4x4()
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: settings)
    stage = tmp_path / "formal"
    rows = _make_stage(stage)
    frames = [
        np.zeros((4, 4), dtype=np.uint8),
        np.full((4, 4), 255, dtype=np.uint8),
        np.full((4, 4), 50, dtype=np.uint8),
        _valid_quadrant_frames(settings)[3],
    ]
    _make_captures(stage, rows, frames)
    result = ccd_evaluate.evaluate_directory(
        config=tmp_path / "unused.yaml",
        manifest=stage / "samples.csv",
        ccd_dir=stage / "ccd_captured",
        output_dir=stage / "evaluation",
        roi=None,
        flip_vertical=False,
        flip_horizontal=False,
        allow_invalid_formal=True,
    )
    assert result["invalid_count"] == 3
    assert result["valid_count"] == 1
    assert result["invalid_by_reason"]["near_black"] == 1
    assert result["invalid_by_reason"]["saturated"] == 1
    assert result["invalid_by_reason"]["near_equal_detector_regions"] == 3


def test_lab_zip_contains_payload_and_light_runtime(tmp_path: Path) -> None:
    export = tmp_path / "export"
    for name, value in {
        "README_LAB.md": "readme",
        "detector_regions.csv": "class,left\n0,0\n",
        "detector_roi_478.png": "preview",
        "hardware_contract.json": json.dumps(
            {
                "profiles": {
                    "demo_topk": {},
                    "formal_fixed_random_100_per_class": {},
                }
            }
        ),
        "lab_model_config.yaml": "{}",
    }.items():
        path = export / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    canonical = export / "phase_to_play" / PHASE_FILENAME
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"phase")
    _make_stage(export / "demo_topk", profile="demo_topk")
    _make_stage(
        export / "formal_fixed_random_100_per_class",
        profile="formal_fixed_random_100_per_class",
    )
    output = tmp_path / "bundle.zip"
    report = create_lab_zip(
        export_dir=export,
        output_path=output,
        include_vendor_sdk=False,
    )
    assert report["vendor_sdk_included"] is False
    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "payload/formal_fixed_random_100_per_class/samples.csv" in names
    assert (
        "experiments/d2nn_mnist4_single_layer_17um_10cm/lab_pipeline.py"
        in names
    )
    assert not any("vendor_sdk" in name for name in names)
    sidecar = json.loads(output.with_suffix(".zip.json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == report["sha256"]
