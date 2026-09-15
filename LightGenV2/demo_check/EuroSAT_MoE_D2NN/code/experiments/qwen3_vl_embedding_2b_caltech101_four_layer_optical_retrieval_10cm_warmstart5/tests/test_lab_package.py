from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.hardware_sdk.tests._fresnel_roi_vertex_fixture import (
    make_dual_slm_checker_grating_fixture,
    make_fresnel_roi_vertex_fixture,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.offline_tail import (
    LanguageGlobalOfflineTail,
)

from ..lab_package import (
    EXPECTED_ARCHITECTURE,
    STAGES,
    TAIL_PARAMETER_COUNT,
    _runtime_files,
    _sha256,
    _validate_checkpoint,
    create_lab_bundle,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_image(path: Path, size_wh: tuple[int, int], value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((size_wh[1], size_wh[0]), value, dtype=np.uint8), mode="L"
    ).save(path)


def _construction() -> dict[str, object]:
    return {
        "width": 192,
        "max_tokens": 224,
        "expansion": 2.0,
        "dropout": 0.1,
        "initial_residual_weight": 0.1,
        "token_mixer_enabled": True,
        "token_mixer_type": "depthwise_conv1d",
        "token_mixer_kernel_size": 5,
        "detector_size": 478,
        "detector_output_size": 224,
        "detector_layernorm_eps": 1.0e-5,
        "detector_layernorm_affine": False,
        "detector_layernorm_scope": "per_token",
        "detector_nonlinearity": "relu",
        "ccd_relative_clip": 12.0,
        "ccd_log_compression": 1.0,
        "minimum_optical_fusion": 0.05,
        "embedding_dim": 64,
    }


def _make_checkpoint(path: Path, *, test_selected: bool = False) -> Path:
    target_sigmoid = (0.055 - 0.05) / 0.95
    raw = math.log(target_sigmoid / (1.0 - target_sigmoid))
    branch = {
        "core.block1_optical_fusion_logit": torch.tensor(raw),
        "core.block2_optical_fusion_logit": torch.tensor(raw),
    }
    torch.save(
        {
            "checkpoint_version": 2,
            "epoch": 8,
            "train_loss": 2.23,
            "vision_optical": branch,
            "language_optical": branch,
            "retrieval_readout": {},
            "metadata": {
                "optical_architecture": EXPECTED_ARCHITECTURE,
                "selection_criterion": "minimum_training_total_loss",
                "test_metrics_used_for_selection": test_selected,
                "weight_variant": "ema",
            },
        },
        path,
    )
    return path


def _make_fixture(root: Path) -> dict[str, Path]:
    calibration = make_fresnel_roi_vertex_fixture(root)
    dual_slm_calibration = make_dual_slm_checker_grating_fixture(root)
    checkpoint = _make_checkpoint(root / "ema_best_train_loss_checkpoint.pt")
    checkpoint_sha = _sha256(checkpoint)

    phase = root / "phase"
    for stage in STAGES:
        _save_image(phase / "compact_phase" / f"{stage}.png", (478, 478))
        _save_image(phase / "phase_bmp" / f"{stage}.bmp", (1920, 1200))
    _write_json(
        phase / "phase_export_report.json",
        {
            "schema_version": 1,
            "checkpoint_sha256": checkpoint_sha,
            "stages": list(STAGES),
            "logical_phase_shape": [478, 478],
            "logical_pixel_pitch_um": 17.0,
            "propagation_distance_m": 0.1,
            "phase_slm": {
                "size_wh": [1920, 1200],
                "pixel_pitch_um": 8.0,
                "center_xy": [980.0, 590.0],
                "flip_vertical_before_raster": True,
                "flip_horizontal_before_raster": False,
            },
        },
    )

    session = root / "session"
    stage_dir = session / "04_language_global"
    rows: list[dict[str, object]] = []
    amplitude_rows: list[dict[str, object]] = []
    order = 0
    for split, count in (("train", 10), ("gallery", 1), ("test", 10)):
        for label in range(10):
            for item in range(count):
                key = f"{split}__{label:02d}__sample_{item:02d}"
                rows.append(
                    {
                        "order": order,
                        "key": key,
                        "sample_id": key,
                        "split": split,
                        "sku_index": label,
                        "sku_name": f"class_{label}",
                        "image_path": f"not-bundled/{key}.jpg",
                    }
                )
                image_path = stage_dir / "compact_amplitude" / f"{key}.png"
                _save_image(image_path, (478, 478), value=(order % 255))
                amplitude_rows.append(
                    {
                        "key": key,
                        "filename": image_path.name,
                        "sha256": _sha256(image_path),
                    }
                )
                order += 1
    _write_csv(session / "manifest.csv", rows)
    _write_csv(stage_dir / "compact_amplitude_manifest.csv", amplitude_rows)
    _save_image(stage_dir / "compact_phase" / "language_global.png", (478, 478))
    native_phase = stage_dir / "phase_to_play" / "language_global.bmp"
    _save_image(native_phase, (1920, 1200))
    _write_csv(
        stage_dir / "phase_to_play" / "reconstruction_manifest.csv",
        [
            {
                "order": 0,
                "output_bmp": native_phase.name,
                "output_sha256": _sha256(native_phase),
            }
        ],
    )
    _write_json(
        stage_dir / "transport_spec.json",
        {
            "schema_version": 2,
            "stage": "language_global",
            "upstream_source": "simulation",
            "measured_upstream_stages": [],
            "samples": 210,
            "checkpoint_sha256": checkpoint_sha,
        },
    )

    construction = _construction()
    tail = LanguageGlobalOfflineTail(**construction)
    assert sum(parameter.numel() for parameter in tail.parameters()) == TAIL_PARAMETER_COUNT
    offline = stage_dir / "offline_downstream"
    offline.mkdir(parents=True)
    cache = {
        "packed_block2_inputs": torch.zeros(420, 192, dtype=torch.float32),
        "offsets": torch.arange(0, 422, 2, dtype=torch.int64),
        "lengths": torch.full((210,), 2, dtype=torch.int64),
        "labels": torch.tensor([int(row["sku_index"]) for row in rows]),
        "split_codes": torch.tensor(
            [{"train": 0, "gallery": 1, "test": 2}[str(row["split"])] for row in rows],
            dtype=torch.uint8,
        ),
        "orders": torch.arange(210, dtype=torch.int64),
    }
    torch.save(cache, offline / "cache.pt")
    torch.save(tail.state_dict(), offline / "downstream_state.pt")
    split_counts = {"train": 100, "gallery": 10, "test": 100}
    per_class = {
        str(label): {"train": 10, "gallery": 1, "test": 10}
        for label in range(10)
    }
    keys = [str(row["key"]) for row in rows]
    _write_json(
        offline / "contract.json",
        {
            "schema_version": 1,
            "type": "language_global_quick_offline_full_parity",
            "profile": "quick210",
            "stage": "language_global",
            "checkpoint_architecture": EXPECTED_ARCHITECTURE,
            "source_checkpoint_sha256": checkpoint_sha,
            "upstream_source": "simulation",
            "measured_upstream_stages": [],
            "sample_count": 210,
            "manifest_relative_path": "../../manifest.csv",
            "manifest_sha256": _sha256(session / "manifest.csv"),
            "ordered_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            "cache_file": "cache.pt",
            "cache_sha256": _sha256(offline / "cache.pt"),
            "state_file": "downstream_state.pt",
            "state_sha256": _sha256(offline / "downstream_state.pt"),
            "tail_construction": construction,
            "tail_trainable_parameter_count": TAIL_PARAMETER_COUNT,
            "split_codes": {"train": 0, "gallery": 1, "test": 2},
            "split_counts": split_counts,
            "class_names": [f"class_{label}" for label in range(10)],
            "class_split_counts": per_class,
            "ccd_contract": {
                "directory_relative_to_stage": "ccd_captured",
                "mode": "L",
                "dtype": "uint8",
                "shape_hw": [478, 478],
                "background_subtraction": False,
                "resizing": False,
            },
        },
    )
    # Existing CCD files must never be selected into the transfer ZIP.
    _save_image(stage_dir / "ccd_captured" / f"{keys[0]}.png", (478, 478))

    stage_a = root / "stage_a"
    stage_b = root / "stage_b"
    for run_dir in (stage_a, stage_b):
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text("training: {}\n", encoding="utf-8")
        (run_dir / "train_log.csv").write_text(
            "epoch,train_loss,test_top1\n1,2.3,nan\n", encoding="utf-8"
        )
    _write_json(stage_a / "warmstart_initialization_report.json", {"status": "passed"})
    _write_json(
        stage_b / "metrics" / "evaluation_summary.json",
        {
            "student": {
                "query_count": 200,
                "gallery_image_count": 30,
                "sku_count": 10,
                "top1_retrieval_accuracy": 0.81,
                "top3_retrieval_accuracy": 0.93,
                "mrr": 0.876345,
            }
        },
    )
    return {
        "checkpoint": checkpoint,
        "phase": phase,
        "session": session,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "calibration": calibration,
        "dual_slm_calibration": dual_slm_calibration,
    }


def test_bundle_is_hash_bound_minimal_and_offline_ready(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "bundle.zip"
    repository = Path(__file__).resolve().parents[3]
    report = create_lab_bundle(
        checkpoint=fixture["checkpoint"],
        phase_export_dir=fixture["phase"],
        quick_session_dir=fixture["session"],
        stage_a_run_dir=fixture["stage_a"],
        stage_b_run_dir=fixture["stage_b"],
        output_path=output,
        include_vendor_sdk=False,
        repo_root=repository,
        fresnel_calibration_dir=fixture["calibration"],
        dual_slm_calibration_dir=fixture["dual_slm_calibration"],
        expected_checkpoint_sha256=None,
        require_fixed_simulation_report=False,
    )
    assert report["fixed_test_top1"] == pytest.approx(0.81)
    assert report["offline_trainable_parameters"] == 255_811
    assert report["zip_validation"]["crc_and_hash_validation"] == "passed"
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle_manifest.json"))
    assert manifest["architecture_contract"]["minimum_optical_fusion_coefficient"] == 0.05
    assert manifest["architecture_contract"]["initial_optical_fusion_coefficient"] == 0.055
    assert manifest["formal_roi_vertex_calibration_contract"][
        "logical_roi_vertex_spacing_phase_px"
    ] == [1015.75, 1015.75]
    assert not manifest["formal_roi_vertex_calibration_contract"][
        "historical_quadrant_center_arrays_formal"
    ]
    assert report["formal_roi_vertex_calibration_phase_masks"] == 9
    assert report["formal_dual_slm_checker_grating_bmps"] == 2
    assert len(
        [
            name
            for name in names
            if name.startswith(
                "payload/calibration/fresnel_roi_vertex_array_532nm_17um_8um_v2/phase_bmp/"
            )
        ]
    ) == 9
    dual_contract = manifest["formal_dual_slm_checker_grating_contract"]
    assert manifest["formal_calibration_sequence"][0].startswith(
        "dual-SLM checker/grating"
    )
    assert dual_contract["amplitude"]["polarity"] == (
        "255=open/transmissive; 0=closed/opaque"
    )
    assert dual_contract["phase"]["vertical_flip_already_applied"] is True
    assert dual_contract["source_absolute_paths_copied_to_bundle"] is False
    assert (
        "payload/calibration/dual_slm_checker_grating/"
        "amplitude_checker_255open_c64_1024x1024.bmp"
    ) in names
    assert (
        "payload/calibration/dual_slm_checker_grating/"
        "phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp"
    ) in names
    assert not any(name.endswith("pair_manifest.json") for name in names)
    assert (
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_"
        "10cm_warmstart5/result_report.py"
    ) in names
    assert (
        "reference/qwen_project_source/RESULT_PLOTTING.md"
    ) in names
    assert manifest["fixed_simulation_report_contract"]["included"] is False
    assert len(
        [
            name
            for name in names
            if name.startswith("payload/quick210/04_language_global/compact_amplitude/")
            and name.endswith(".png")
        ]
    ) == 210
    assert not any("ccd_captured" in name or "amplitude_to_play" in name for name in names)
    assert "payload/quick210/04_language_global/offline_downstream/cache.pt" in names
    assert "payload/quick210/04_language_global/offline_downstream/downstream_state.pt" in names


def test_vendor_transfer_is_x64_runtime_only() -> None:
    repository = Path(__file__).resolve().parents[3]
    paths = {
        item.archive_path
        for item in _runtime_files(repository, include_vendor_sdk=True)
        if item.category == "vendor_sdk_runtime_x64"
    }
    assert any(path.endswith("/SDK/Blink_C_wrapper.dll") for path in paths)
    assert any(path.endswith("/lib/x64/TUCam.dll") for path in paths)
    assert any(path.endswith("slm7930_at532_30C.lut") for path in paths)
    assert any(path.endswith("slm7930_at532_70C.lut") for path in paths)
    lowered = {path.lower() for path in paths}
    assert not any("/lib/x86/" in path for path in lowered)
    assert not any("/image files/" in path for path in lowered)
    assert not any("/wfc files/" in path for path in lowered)
    assert not any(path.endswith(".pdf") for path in lowered)


def test_bundle_rejects_phase_checkpoint_hash_drift(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture["phase"] / "phase_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = "0" * 64
    _write_json(report_path, report)
    with pytest.raises(RuntimeError, match="phase export checkpoint SHA-256 mismatch"):
        create_lab_bundle(
            checkpoint=fixture["checkpoint"],
            phase_export_dir=fixture["phase"],
            quick_session_dir=fixture["session"],
            stage_a_run_dir=fixture["stage_a"],
            stage_b_run_dir=fixture["stage_b"],
            output_path=tmp_path / "bad.zip",
            include_vendor_sdk=False,
            repo_root=Path(__file__).resolve().parents[3],
            fresnel_calibration_dir=fixture["calibration"],
            dual_slm_calibration_dir=fixture["dual_slm_calibration"],
            expected_checkpoint_sha256=None,
            require_fixed_simulation_report=False,
        )


def test_formal_bundle_rejects_an_unpinned_checkpoint(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="pinned to the sealed 81%"):
        create_lab_bundle(
            checkpoint=fixture["checkpoint"],
            phase_export_dir=fixture["phase"],
            quick_session_dir=fixture["session"],
            stage_a_run_dir=fixture["stage_a"],
            stage_b_run_dir=fixture["stage_b"],
            output_path=tmp_path / "wrong_formal.zip",
            include_vendor_sdk=False,
            repo_root=Path(__file__).resolve().parents[3],
            fresnel_calibration_dir=fixture["calibration"],
            dual_slm_calibration_dir=fixture["dual_slm_calibration"],
            require_fixed_simulation_report=False,
        )


def test_checkpoint_rejects_test_selected_weights(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "bad.pt", test_selected=True)
    with pytest.raises(RuntimeError, match="test_metrics_used_for_selection=false"):
        _validate_checkpoint(checkpoint)


def test_warmstart_offline_import_does_not_load_qwen_or_transformers() -> None:
    repository = Path(__file__).resolve().parents[3]
    module = (
        "experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_"
        "10cm_warmstart5.offline_quick_finetune"
    )
    script = (
        f"import {module}; import sys; "
        "assert 'transformers' not in sys.modules; "
        "assert not any(name.endswith('.modeling') and 'qwen3_vl_embedding' in name "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
