from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from ..agreement_common import sha256_file, stage_directory, write_csv, write_json
from ..agreement_evaluate import (
    apply_orientation,
    evaluate_session,
    network_input_map,
    orientation_diagnostics,
    pcc,
    shape_nrmse,
    signal_mask,
    structural_similarity,
)
from ..agreement_export import (
    _decode_export_phase,
    _existing_session_stages,
    designed_probes,
    simulate_active_field,
)
from ..agreement_report import build_report


def test_core_agreement_metrics_do_not_hide_scale_or_orientation() -> None:
    value = np.zeros((96, 96), dtype=np.float32)
    value[12:35, 21:72] = 1.0
    value[50:81, 62:87] = 0.4
    scaled = value * 9.0
    assert pcc(scaled, value) == pytest.approx(1.0)
    assert shape_nrmse(scaled, value) == pytest.approx(0.0, abs=1e-7)
    assert structural_similarity(value, value) == pytest.approx(1.0)
    assert pcc(np.flip(value, axis=1), value) < 0.95
    mask = signal_mask(value, 0.99)
    assert mask.any() and not mask.all()
    mapped = network_input_map(value, relative_clip=12.0, log_compression=1.0)
    assert mapped.shape == (224, 224)
    assert np.isfinite(mapped).all()


def test_orientation_diagnostic_never_changes_primary_array() -> None:
    probes = designed_probes(96, seed=7)
    theory = probes["quadrant_code"].astype(np.float32)
    measured = apply_orientation(theory, "flip_horizontal").copy()
    before = measured.copy()
    rows = orientation_diagnostics(measured, theory, relative_clip=12.0)
    best = max(rows, key=lambda row: -2.0 if row["pcc_full"] is None else row["pcc_full"])
    assert best["orientation"] == "flip_horizontal"
    assert np.array_equal(measured, before)
    # The designed suite contains directional probes that are not invariant to
    # vertical/horizontal flips.
    assert not np.array_equal(probes["quadrant_code"], np.flip(probes["quadrant_code"], 0))


def test_transport_simulation_places_active_field_without_wraparound() -> None:
    class Aperture:
        y0, y1, x0, x1 = 1, 5, 1, 5

    class Geometry:
        canvas_size = 6
        active_aperture = Aperture()

    class Core:
        geometry = Geometry()

    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.core = Core()

        def _simulate_detector_roi(self, field, modulation, shifts):
            assert shifts == {"input": (0, 0), "phase": (0, 0), "ccd": (0, 0)}
            value = (field * modulation).abs().square()
            return value[:, 1:5, 1:5]

    amplitude = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 15.0
    phase = torch.linspace(0, 2 * torch.pi, 16).reshape(4, 4)
    observed = simulate_active_field(Branch(), amplitude, phase)
    torch.testing.assert_close(observed[0], amplitude.square())


def test_phase_transport_decode_matches_hardware_bridge_256_bin_encoder() -> None:
    encoded = np.asarray([[0, 1, 127, 255]], dtype=np.uint8)
    settings = SimpleNamespace(
        hardware_phase_flip_horizontal=False,
        hardware_phase_flip_vertical=False,
    )
    decoded = _decode_export_phase(encoded, settings)
    np.testing.assert_allclose(
        decoded,
        encoded.astype(np.float32) * (2.0 * np.pi / 256.0),
        rtol=0.0,
        atol=0.0,
    )
    assert float(decoded[0, -1]) < 2.0 * np.pi


def _save_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path)


def _save_npz(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, intensity=np.asarray(value, dtype=np.float32))


def _fake_session(root: Path) -> tuple[Path, Path]:
    session = root / "agreement_session"
    stage = "language_global"
    stage_dir = stage_directory(session, stage)
    size = 478
    y, x = np.mgrid[:size, :size]
    calibration = (
        np.exp(-((x - 155) ** 2 + (y - 116) ** 2) / (2 * 31.0**2))
        + 0.4 * np.exp(-((x - 350) ** 2 + (y - 314) ** 2) / (2 * 47.0**2))
    ).astype(np.float32)
    evaluation = (
        np.exp(-((x - 272) ** 2 + (y - 193) ** 2) / (2 * 52.0**2))
        + 0.2 * ((x > 80) & (x < 145) & (y > 330) & (y < 410))
    ).astype(np.float32)
    entries = [
        ("probe__asymmetric", "probe__asymmetric", "calibration", 0, calibration),
        ("test__00__sample", "test__00__sample", "evaluation", 0, evaluation),
        ("test__00__sample__r01", "test__00__sample", "evaluation", 1, evaluation),
    ]

    phase_compact = stage_dir / "compact_phase" / f"{stage}.png"
    _save_png(phase_compact, np.zeros((size, size), dtype=np.uint8))
    phase_bmp = stage_dir / "phase_to_play" / f"{stage}.bmp"
    _save_png(phase_bmp, np.zeros((1200, 1920), dtype=np.uint8))
    phase_reconstruction = stage_dir / "phase_to_play" / "reconstruction_manifest.csv"
    write_csv(
        phase_reconstruction,
        [{"output_bmp": phase_bmp.name, "output_sha256": sha256_file(phase_bmp)}],
        ("output_bmp", "output_sha256"),
    )

    probe_rows = []
    reconstruction_rows = []
    capture_rows = []
    for order, (capture_key, canonical_key, role, repeat_index, theory) in enumerate(entries):
        amplitude = np.rint(theory / max(float(theory.max()), 1e-9) * 255).astype(np.uint8)
        compact = stage_dir / "compact_amplitude" / f"{capture_key}.png"
        amplitude_bmp = stage_dir / "amplitude_to_play" / f"{capture_key}.bmp"
        _save_png(compact, amplitude)
        _save_png(amplitude_bmp, amplitude)
        ideal = stage_dir / "theoretical_ccd" / "ideal_model_fp32" / f"{capture_key}.npz"
        transport = stage_dir / "theoretical_ccd" / "transport_quantized" / f"{capture_key}.npz"
        _save_npz(ideal, theory)
        _save_npz(transport, theory)
        measured = np.rint(theory / max(float(theory.max()), 1e-9) * 220).astype(np.uint8)
        ccd = stage_dir / "ccd_captured" / f"{capture_key}.png"
        _save_png(ccd, measured)
        probe_rows.append(
            {
                "order": order,
                "stage": stage,
                "upstream_source": "simulation",
                "source_kind": "designed" if role == "calibration" else "model",
                "role": role,
                "split": "calibration" if role == "calibration" else "test",
                "sku_index": "" if role == "calibration" else 0,
                "sku_name": "" if role == "calibration" else "sample_class",
                "sample_id": "" if role == "calibration" else "sample-id",
                "repeat_index": repeat_index,
                "capture_key": capture_key,
                "canonical_key": canonical_key,
                "amplitude_file": amplitude_bmp.name,
                "compact_amplitude_file": compact.relative_to(stage_dir).as_posix(),
                "compact_amplitude_sha256": sha256_file(compact),
                "amplitude_encoding_scale": 1.0,
                "ideal_reference_file": ideal.relative_to(stage_dir).as_posix(),
                "ideal_reference_sha256": sha256_file(ideal),
                "transport_reference_file": transport.relative_to(stage_dir).as_posix(),
                "transport_reference_sha256": sha256_file(transport),
            }
        )
        reconstruction_rows.append(
            {
                "output_bmp": amplitude_bmp.name,
                "source_sha256": sha256_file(compact),
                "output_sha256": sha256_file(amplitude_bmp),
            }
        )
        capture_rows.append(
            {
                "amplitude_bmp": amplitude_bmp.name,
                "amplitude_bmp_sha256": sha256_file(amplitude_bmp),
                "ccd_capture": ccd.name,
                "output_sha256": sha256_file(ccd),
                "detector_geometry_file_sha256": "1" * 64,
                "detector_geometry_payload_sha256": "2" * 64,
                "orientation_canonicalized": True,
                "saved_frame_orientation": "canonical_model_xy",
                "downstream_loader_flip_required": False,
                "background_subtraction": False,
                "per_frame_minmax_normalization": False,
                "phase_mask_sha256": sha256_file(phase_bmp),
            }
        )
    probe_manifest = stage_dir / "probe_manifest.csv"
    write_csv(probe_manifest, probe_rows, list(probe_rows[0]))
    write_csv(
        stage_dir / "amplitude_to_play" / "reconstruction_manifest.csv",
        reconstruction_rows,
        list(reconstruction_rows[0]),
    )
    write_csv(
        stage_dir / "acquisition_logs" / "capture_manifest.csv",
        capture_rows,
        list(capture_rows[0]),
    )
    contract = {
        "schema_version": 1,
        "type": "qwen_warmstart5_sim_to_real_agreement",
        "stage": stage,
        "checkpoint_sha256": "a" * 64,
        "resolved_config_sha256": "b" * 64,
        "probe_manifest": probe_manifest.name,
        "probe_manifest_sha256": sha256_file(probe_manifest),
        "probe_count": len(probe_rows),
        "active_shape_hw": [size, size],
        "phase_reconstruction_manifest": "phase_to_play/reconstruction_manifest.csv",
        "phase_reconstruction_manifest_sha256": sha256_file(phase_reconstruction),
        "agreement": {
            "relative_clip": 12.0,
            "log_compression": 1.0,
            "signal_energy_fraction": 0.99,
            "bootstrap_samples": 50,
            "probe_seed": 42,
        },
    }
    contract_path = stage_dir / "agreement_contract.json"
    write_json(contract_path, contract)
    write_json(
        session / "agreement_manifest.json",
        {
            "schema_version": 1,
            "type": "qwen_warmstart5_sim_to_real_agreement_session",
            "checkpoint_sha256": "a" * 64,
            "resolved_config_sha256": "b" * 64,
            "stages": [
                {
                    "stage": stage,
                    "directory": stage_dir.relative_to(session).as_posix(),
                    "contract": contract_path.relative_to(session).as_posix(),
                    "contract_sha256": sha256_file(contract_path),
                }
            ],
        },
    )
    return session, stage_dir


def test_strict_end_to_end_evaluation_and_report(tmp_path: Path) -> None:
    session, _ = _fake_session(tmp_path)
    result = evaluate_session(session)
    output = session / "agreement_evaluation"
    assert result["pairings_verified"] == 6
    assert result["repeatability_rows"] == 2
    rows = list(csv.DictReader((output / "metrics_per_probe.csv").open(encoding="utf-8-sig")))
    primary = [
        row
        for row in rows
        if row["role"] == "evaluation"
        and row["reference_kind"] == "transport_quantized"
        and row["domain"] == "linear"
    ]
    assert len(primary) == 1
    assert float(primary[0]["pcc_full"]) > 0.999
    orientation = list(csv.DictReader((output / "orientation_summary.csv").open(encoding="utf-8-sig")))
    best = [row for row in orientation if row["diagnostic_best_candidate"] == "True"]
    assert len(best) == 1 and best[0]["orientation"] == "identity"

    pytest.importorskip("matplotlib")
    # A stale unverified registered frame must never override the exact CCD
    # paths recorded by pairing_audit.csv.
    _save_png(
        session / "04_language_global" / "ccd_registered" / "test__00__sample.png",
        np.zeros((8, 8), dtype=np.uint8),
    )
    report = build_report(output, formats=("png",), require_arial=False)
    assert report["font_size_pt"] == 7
    assert all(item["status"] == "available" for item in report["figures"])
    assert (output / "report" / "fig01_agreement_distributions.png").is_file()

    changed_reference = (
        session
        / "04_language_global"
        / "theoretical_ccd"
        / "transport_quantized"
        / "test__00__sample.npz"
    )
    _save_npz(changed_reference, np.zeros((478, 478), dtype=np.float32))
    with pytest.raises(RuntimeError, match="reference changed after evaluation"):
        build_report(output, formats=("png",), require_arial=False)


def test_strict_pairing_rejects_changed_ccd_manifest(tmp_path: Path) -> None:
    session, stage_dir = _fake_session(tmp_path)
    capture_manifest = stage_dir / "acquisition_logs" / "capture_manifest.csv"
    rows = list(csv.DictReader(capture_manifest.open(encoding="utf-8-sig")))
    rows[0]["phase_mask_sha256"] = "0" * 64
    write_csv(capture_manifest, rows, list(rows[0]))
    with pytest.raises(RuntimeError, match="phase SHA mismatch"):
        evaluate_session(session)


def test_strict_pairing_rejects_unbound_played_amplitude(tmp_path: Path) -> None:
    session, stage_dir = _fake_session(tmp_path)
    capture_manifest = stage_dir / "acquisition_logs" / "capture_manifest.csv"
    rows = list(csv.DictReader(capture_manifest.open(encoding="utf-8-sig")))
    rows[0]["amplitude_bmp_sha256"] = "0" * 64
    write_csv(capture_manifest, rows, list(rows[0]))
    with pytest.raises(RuntimeError, match="Acquisition amplitude SHA mismatch"):
        evaluate_session(session)


def test_evaluator_rejects_stage_from_another_checkpoint(tmp_path: Path) -> None:
    session, stage_dir = _fake_session(tmp_path)
    contract_path = stage_dir / "agreement_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["checkpoint_sha256"] = "c" * 64
    write_json(contract_path, contract)
    root_path = session / "agreement_manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["stages"][0]["contract_sha256"] = sha256_file(contract_path)
    write_json(root_path, root)
    with pytest.raises(RuntimeError, match="checkpoint provenance mismatch"):
        evaluate_session(session)


def test_exporter_refuses_to_append_a_different_model(tmp_path: Path) -> None:
    session, _ = _fake_session(tmp_path)
    stages = _existing_session_stages(
        session,
        checkpoint_sha256="a" * 64,
        resolved_config_sha256="b" * 64,
    )
    assert [item["stage"] for item in stages] == ["language_global"]
    with pytest.raises(RuntimeError, match="different checkpoint or resolved config"):
        _existing_session_stages(
            session,
            checkpoint_sha256="c" * 64,
            resolved_config_sha256="b" * 64,
        )
