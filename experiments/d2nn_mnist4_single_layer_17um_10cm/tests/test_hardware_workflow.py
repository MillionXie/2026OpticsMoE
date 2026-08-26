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


def _make_stage(stage: Path, profile: str = "formal") -> None:
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
            }
        ),
        encoding="utf-8",
    )


def test_validate_stage_requires_exact_manifest_match(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _make_stage(stage)
    report = validate_stage(stage)
    assert report["samples"] == 4
    assert report["profile"] == "formal"
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
    settings = SimpleNamespace(
        ccd_target_size=4,
        loss_eps=1.0e-8,
        detector_bounds=lambda: (
            (0, 0, 2, 2),
            (2, 0, 4, 2),
            (0, 2, 2, 4),
            (2, 2, 4, 4),
        ),
    )
    monkeypatch.setattr(ccd_evaluate, "load_settings", lambda _: settings)
    manifest = tmp_path / "samples.csv"
    rows = [
        {
            "key": f"sample_{label}",
            "label": label,
            "profile": "formal_fixed_random_100_per_class",
            "selection_policy": "fixed_random",
        }
        for label in range(4)
    ]
    _write_csv(manifest, rows)
    captures = tmp_path / "ccd"
    captures.mkdir()
    bounds = settings.detector_bounds()
    for label, (left, top, right, bottom) in enumerate(bounds):
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[top:bottom, left:right] = 255
        Image.fromarray(frame, mode="L").save(captures / f"sample_{label}.png")
    result = ccd_evaluate.evaluate_directory(
        config=tmp_path / "unused.yaml",
        manifest=manifest,
        ccd_dir=captures,
        output_dir=tmp_path / "evaluation",
        roi=None,
        flip_vertical=False,
        flip_horizontal=False,
    )
    assert result["accuracy"] == 1.0
    assert result["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert result["profiles"] == ["formal_fixed_random_100_per_class"]


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
