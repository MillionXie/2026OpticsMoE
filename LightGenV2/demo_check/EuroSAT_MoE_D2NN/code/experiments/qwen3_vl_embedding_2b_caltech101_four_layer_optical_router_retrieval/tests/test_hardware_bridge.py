from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval import (
    build_routed_amplitudes as builder,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval import (
    hardware_bridge as bridge,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval import (
    score_router_ccd as scorer,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.hardware_contract import (
    expected_key_files,
    read_csv,
    read_json,
    require_empty_directory,
    sha256_file,
    stage_directory,
    write_csv,
    write_json,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        router_backend="optical",
        active_size=478,
        expert_size=224,
        expert_pitch=254,
        top_k=2,
        router_temperature=1.0,
        router_weight_normalization="power_l2",
        optical_router_energy_eps=1.0e-8,
        optical_router_score_normalization="standardized_region_energy",
        optical_router_detector_intervals=((164, 223), (255, 314)),
        optical_router_maximum_saturated_pixel_fraction=0.02,
        optical_router_minimum_p99_uint8=8.0,
        optical_router_minimum_dynamic_range_uint8=4.0,
        optical_router_minimum_topk_probability_margin=0.01,
        hardware_amplitude_invert_before_export=False,
        hardware_amplitude_bright_value_uint8=255,
        hardware_amplitude_dark_value_uint8=0,
        hardware_amplitude_slm_width=1024,
        hardware_amplitude_slm_height=1024,
        hardware_amplitude_slm_center_x=512.0,
        hardware_amplitude_slm_center_y=512.0,
        hardware_amplitude_slm_pixel_pitch_um=17.0,
        language_optical_pixel_pitch_um=17.0,
    )


def _write_routing_csv(path: Path, filename: str = "sample.png") -> None:
    probability = (0.4, 0.3, 0.2, 0.1)
    norm = (probability[0] ** 2 + probability[1] ** 2) ** 0.5
    weights = (probability[0] / norm, probability[1] / norm, 0.0, 0.0)
    row: dict[str, object] = {
        "filename": filename,
        "ccd_sha256": "0" * 64,
        "selected_experts": "0,1",
        "raw_capture_fraction": 0.5,
        "topk_probability_margin": 0.1,
        "saturated_pixel_fraction": 0.0,
    }
    for index in range(4):
        row[f"energy_{index}"] = probability[index] * 100.0
        row[f"energy_fraction_{index}"] = probability[index]
        row[f"probability_{index}"] = probability[index]
        row[f"weight_{index}"] = weights[index]
        row[f"selected_{index}"] = index < 2
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_require_empty_directory_rejects_stale_files(tmp_path: Path) -> None:
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "old.png").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="stale"):
        require_empty_directory(destination, label="test output")


def test_expected_key_files_requires_exact_uint8_grayscale_set(tmp_path: Path) -> None:
    root = tmp_path / "ccd"
    root.mkdir()
    Image.fromarray(np.zeros((478, 478), dtype=np.uint8), mode="L").save(
        root / "a.png"
    )
    rows = expected_key_files(root, ["a"], expected_shape_hw=(478, 478))
    assert rows[0]["key"] == "a"
    Image.fromarray(np.zeros((478, 478, 3), dtype=np.uint8), mode="RGB").save(
        root / "b.png"
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        expected_key_files(root, ["a"], expected_shape_hw=(478, 478))


def test_dataset_sample_contract_checks_path_and_content_sha_not_only_key() -> None:
    sealed = [
        {
            "order": "0",
            "key": "train__sample",
            "sample_id": "sample",
            "split": "train",
            "sku_index": "3",
            "sku_name": "Faces",
            "image_path": "C:/dataset/Faces/sample.jpg",
            "image_sha256": "1" * 64,
        }
    ]
    current = [dict(sealed[0], order=0, sku_index=3)]
    bridge._verify_dataset_sample_rows(sealed, current)
    with pytest.raises(RuntimeError, match="image_sha256"):
        bridge._verify_dataset_sample_rows(
            sealed, [dict(current[0], image_sha256="2" * 64)]
        )
    with pytest.raises(RuntimeError, match="image_path"):
        bridge._verify_dataset_sample_rows(
            sealed, [dict(current[0], image_path="C:/other/sample.jpg")]
        )


def test_formal_checkpoint_architecture_requires_exact_metadata(tmp_path: Path) -> None:
    expected = "router_architecture_v1"
    correct = tmp_path / "correct.pt"
    torch.save({"metadata": {"optical_architecture": expected}}, correct)
    bridge._require_checkpoint_architecture(correct, expected)

    missing = tmp_path / "missing.pt"
    torch.save({"metadata": {}}, missing)
    with pytest.raises(RuntimeError, match="no optical_architecture"):
        bridge._require_checkpoint_architecture(missing, expected)

    mismatch = tmp_path / "mismatch.pt"
    torch.save(
        {"metadata": {"optical_architecture": "another_architecture"}}, mismatch
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        bridge._require_checkpoint_architecture(mismatch, expected)


def test_load_session_rechecks_complete_resolved_config_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("config", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    write_csv(tmp_path / bridge.MANIFEST_FILENAME, [{"key": "sample"}])
    resolved = SimpleNamespace(
        router_backend="optical",
        top_k=2,
        learning_rate=1.0e-4,
        output_dir=tmp_path / "run",
        arbitrary_nested={"value": (1, 2, 3)},
    )
    state = {
        "checkpoint_architecture": "architecture",
        "resolved_config_identity": bridge._resolved_config_identity(resolved),
        "resolved_hardware_contract": {"contract": "fixed"},
    }
    write_json(tmp_path / bridge.STATE_FILENAME, state)
    monkeypatch.setattr(bridge, "validate_state_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, "load_settings", lambda _: resolved)
    monkeypatch.setattr(bridge, "architecture_label", lambda _: "architecture")
    monkeypatch.setattr(
        bridge, "_resolved_hardware_contract", lambda _: {"contract": "fixed"}
    )
    bridge._load_session(
        config=config, checkpoint=checkpoint, session_dir=tmp_path
    )
    resolved.learning_rate = 2.0e-4
    with pytest.raises(RuntimeError, match="learning_rate"):
        bridge._load_session(
            config=config, checkpoint=checkpoint, session_dir=tmp_path
        )


def test_score_router_ccd_uses_canonical_regions_and_real_uint8_saturation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    monkeypatch.setattr(scorer, "load_settings", lambda _: settings)
    config = tmp_path / "config.yaml"
    config.write_text("router: optical\n", encoding="utf-8")
    source = tmp_path / "ccd"
    source.mkdir()
    value = np.zeros((478, 478), dtype=np.uint8)
    value[164:223, 164:223] = 240
    value[164:223, 255:314] = 128
    value[255:314, 164:223] = 32
    value[255:314, 255:314] = 16
    Image.fromarray(value, mode="L").save(source / "sample.png")
    output = tmp_path / "routing"
    report = scorer.score_directory(config, source, output)
    rows = list(csv.DictReader((output / "routing.csv").open(encoding="utf-8-sig")))
    assert rows[0]["selected_experts"] == "0,1"
    assert float(rows[0]["saturated_pixel_fraction"]) == pytest.approx(
        float(np.mean(value == 255))
    )
    assert "raw_capture_fraction" in rows[0]
    assert "capture_fraction" not in rows[0]
    assert "mean_raw_capture_fraction" in report
    assert "mean_capture_fraction" not in report
    assert report["routing_csv_sha256"]
    with pytest.raises(RuntimeError, match="stale"):
        scorer.score_directory(config, source, output)


@pytest.mark.parametrize(
    ("gray", "expected_reason"),
    ((128, "uniform_detector_region_energy"), (255, "all_pixels_saturated")),
)
def test_score_router_ccd_rejects_uniform_and_all_saturated_frames_with_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gray: int,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(scorer, "load_settings", lambda _: _settings())
    config = tmp_path / "config.yaml"
    config.write_text("router: optical\n", encoding="utf-8")
    source = tmp_path / "ccd"
    source.mkdir()
    Image.fromarray(np.full((478, 478), gray, dtype=np.uint8), mode="L").save(
        source / "bad.png"
    )
    output = tmp_path / "routing"
    with pytest.raises(RuntimeError, match="quality gate rejected"):
        scorer.score_directory(config, source, output)
    assert not (output / "routing.csv").exists()
    failures = read_csv(output / "routing_quality_failures.csv")
    assert expected_reason in failures[0]["failure_reasons"]
    report = read_json(output / "routing_quality_report.json")
    assert report["quality_gate_passed"] is False
    assert report["failed_filenames"] == ["bad.png"]


@pytest.mark.parametrize(
    ("setting_name", "threshold", "expected_reason"),
    (
        (
            "optical_router_maximum_saturated_pixel_fraction",
            0.01,
            "saturated_pixel_fraction_above_maximum",
        ),
        ("optical_router_minimum_p99_uint8", 250.0, "p99_below_minimum"),
        (
            "optical_router_minimum_dynamic_range_uint8",
            250.0,
            "dynamic_range_below_minimum",
        ),
        (
            "optical_router_minimum_topk_probability_margin",
            0.99,
            "topk_probability_margin_below_minimum",
        ),
    ),
)
def test_score_router_ccd_applies_each_configured_quality_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting_name: str,
    threshold: float,
    expected_reason: str,
) -> None:
    settings = _settings()
    setattr(settings, setting_name, threshold)
    monkeypatch.setattr(scorer, "load_settings", lambda _: settings)
    config = tmp_path / "config.yaml"
    config.write_text("router: optical\n", encoding="utf-8")
    source = tmp_path / "ccd"
    source.mkdir()
    value = np.zeros((478, 478), dtype=np.uint8)
    value[164:223, 164:223] = (
        255
        if setting_name == "optical_router_maximum_saturated_pixel_fraction"
        else 240
    )
    value[164:223, 255:314] = 120
    value[255:314, 164:223] = 40
    value[255:314, 255:314] = 5
    Image.fromarray(value, mode="L").save(source / "bad.png")
    with pytest.raises(RuntimeError, match="quality gate rejected"):
        scorer.score_directory(config, source, tmp_path / "routing")
    failures = read_csv(tmp_path / "routing" / "routing_quality_failures.csv")
    assert expected_reason in failures[0]["failure_reasons"]


def test_score_router_ccd_honors_session_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scorer, "load_settings", lambda _: _settings())
    config = tmp_path / "config.yaml"
    config.write_text("router: optical\n", encoding="utf-8")
    source = tmp_path / "ccd"
    source.mkdir()
    value = np.ones((478, 478), dtype=np.uint8)
    value[164:223, 164:223] = 20
    value[164:223, 255:314] = 12
    for stem in ("gallery__a", "train__z"):
        Image.fromarray(value, mode="L").save(source / f"{stem}.png")
    output = tmp_path / "routing"
    scorer.score_directory(
        config,
        source,
        output,
        expected_stems=["train__z", "gallery__a"],
    )
    rows = list(csv.DictReader((output / "routing.csv").open(encoding="utf-8-sig")))
    assert [Path(row["filename"]).stem for row in rows] == [
        "train__z",
        "gallery__a",
    ]


def test_build_routed_amplitudes_enforces_power_l2_and_moe4_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    monkeypatch.setattr(builder, "load_settings", lambda _: settings)
    monkeypatch.setattr(
        builder,
        "reconstruct_directory",
        lambda *args, **kwargs: {"mapping_mode": "physical_pitch_nearest", "files": 1},
    )
    config = tmp_path / "config.yaml"
    config.write_text("router: optical\n", encoding="utf-8")
    inputs = tmp_path / "central"
    inputs.mkdir()
    Image.fromarray(np.full((224, 224), 100, dtype=np.uint8), mode="L").save(
        inputs / "sample.png"
    )
    routing = tmp_path / "routing.csv"
    _write_routing_csv(routing)
    output = tmp_path / "routed"
    report = builder.build(config, inputs, routing, output)
    active = np.asarray(Image.open(output / "compact_amplitude" / "sample.png"))
    assert active.shape == (478, 478)
    assert np.all(active[:224, :224] == 80)
    assert np.all(active[:224, 254:478] == 60)
    assert np.all(active[224:254] == 0)
    assert report["routing_csv_sha256"]


def test_routing_payload_is_batch_aligned_and_contains_real_optical_metrics(
    tmp_path: Path,
) -> None:
    routing = stage_directory(tmp_path, "vision_router") / "routing" / "routing.csv"
    _write_routing_csv(routing)
    payload = bridge._routing_payload(
        tmp_path,
        "vision_router",
        ["sample"],
        device=torch.device("cpu"),
        top_k=2,
    )
    assert set(payload) == {
        "probabilities",
        "weights",
        "selected_mask",
        "selected_indices",
        "detector_energy",
        "detector_energy_fraction",
        "raw_capture_fraction",
    }
    assert payload["selected_indices"].tolist() == [[0, 1]]
    assert float(payload["raw_capture_fraction"][0]) == pytest.approx(0.5)
    assert float(payload["weights"].square().sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("stage", bridge.FEATURE_STAGES)
def test_all_feature_stages_preserve_canonical_asymmetric_orientation(
    tmp_path: Path, stage: str
) -> None:
    root = stage_directory(tmp_path, stage) / "ccd_captured"
    root.mkdir(parents=True)
    value = np.zeros((478, 478), dtype=np.uint8)
    value[7, 19] = 231
    value[470, 451] = 17
    Image.fromarray(value, mode="L").save(root / "sample.png")
    loaded = bridge._load_canonical_feature_ccd(
        tmp_path, stage, "sample", active_size=478
    )
    assert float(loaded[7, 19]) == 231.0
    assert float(loaded[470, 451]) == 17.0
    assert float(loaded[470, 458]) == 0.0  # would be nonzero after a 180-degree flip


@pytest.mark.parametrize("stage", bridge.ROUTER_STAGES)
def test_both_router_stages_score_canonical_asymmetric_orientation_without_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    settings = _settings()
    monkeypatch.setattr(scorer, "load_settings", lambda _: settings)
    config = tmp_path / "config.yaml"
    config.write_text("router: optical\n", encoding="utf-8")
    source = stage_directory(tmp_path, stage) / "ccd_captured"
    source.mkdir(parents=True)
    value = np.zeros((478, 478), dtype=np.uint8)
    # An intentionally asymmetric field: canonical top-left must remain expert 0;
    # a hidden 180-degree flip would instead make expert 3 dominant.
    value[164:223, 164:223] = 240
    value[164:223, 255:314] = 120
    value[255:314, 164:223] = 40
    value[255:314, 255:314] = 5
    Image.fromarray(value, mode="L").save(source / "sample.png")
    output = stage_directory(tmp_path, stage) / "routing"
    scorer.score_directory(config, source, output, expected_stems=["sample"])
    rows = read_csv(output / "routing.csv")
    assert rows[0]["selected_experts"] == "0,1"
    assert float(rows[0]["energy_0"]) > float(rows[0]["energy_3"])


def test_downstream_export_requires_preceding_feature_finetune() -> None:
    state = {"stages": {stage: {} for stage in bridge.SIX_STAGES}}
    state["stages"]["vision_expert"]["capture"] = {"sealed": True}
    with pytest.raises(RuntimeError, match="finetune"):
        bridge._require_export_dependency(state, "vision_global")
    state["stages"]["vision_expert"]["finetune"] = {"checkpoint": "after.pt"}
    bridge._require_export_dependency(state, "vision_global")


def _write_minimal_export_chain(
    session_dir: Path, *, bad_reconstruction_source_sha: bool = False
) -> dict[str, Path]:
    stage = "vision_router"
    destination = stage_directory(session_dir, stage)
    compact_dir = destination / "compact_amplitude"
    amplitude_dir = destination / "amplitude_to_play"
    phase_dir = destination / "phase_to_play"
    for directory in (compact_dir, amplitude_dir, phase_dir):
        directory.mkdir(parents=True, exist_ok=True)
    compact = compact_dir / "sample.png"
    Image.fromarray(np.full((224, 224), 71, dtype=np.uint8), mode="L").save(
        compact
    )
    compact_manifest = destination / "compact_amplitude_manifest.csv"
    write_csv(
        compact_manifest,
        [
            {
                "order": 0,
                "key": "sample",
                "filename": compact.name,
                "sha256": sha256_file(compact),
            }
        ],
    )
    amplitude = amplitude_dir / "sample.bmp"
    Image.fromarray(np.full((1024, 1024), 71, dtype=np.uint8), mode="L").save(
        amplitude
    )
    reconstruction_manifest = amplitude_dir / "reconstruction_manifest.csv"
    write_csv(
        reconstruction_manifest,
        [
            {
                "order": 0,
                "basename": "sample",
                "source_png": compact.name,
                "source_sha256": (
                    "f" * 64
                    if bad_reconstruction_source_sha
                    else sha256_file(compact)
                ),
                "output_bmp": amplitude.name,
                "output_sha256": sha256_file(amplitude),
            }
        ],
    )
    phase = phase_dir / f"{stage}.bmp"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(phase)
    transport = destination / "transport_spec.json"
    write_json(
        transport,
        {
            "stage": stage,
            "amplitude_manifest_sha256": sha256_file(compact_manifest),
            "amplitude_reconstruction_manifest_sha256": sha256_file(
                reconstruction_manifest
            ),
            "phase_bmp_sha256": sha256_file(phase),
        },
    )
    state = {"stages": {name: {} for name in bridge.SIX_STAGES}}
    state["stages"][stage]["export"] = {
        "transport_spec_sha256": sha256_file(transport),
        "amplitude_manifest_sha256": sha256_file(compact_manifest),
        "amplitude_reconstruction_manifest_sha256": sha256_file(
            reconstruction_manifest
        ),
        "phase_bmp_sha256": sha256_file(phase),
    }
    write_json(session_dir / bridge.STATE_FILENAME, state)
    return {
        "compact": compact,
        "compact_manifest": compact_manifest,
        "reconstruction_manifest": reconstruction_manifest,
    }


def test_export_chain_rejects_compact_png_tampering(tmp_path: Path) -> None:
    paths = _write_minimal_export_chain(tmp_path)
    bridge._verify_export_artifact(tmp_path, "vision_router")
    Image.fromarray(np.full((224, 224), 72, dtype=np.uint8), mode="L").save(
        paths["compact"]
    )
    with pytest.raises(RuntimeError, match="Compact amplitude PNG"):
        bridge._verify_export_artifact(tmp_path, "vision_router")


def test_export_chain_rejects_manifest_tampering(tmp_path: Path) -> None:
    paths = _write_minimal_export_chain(tmp_path)
    rows = read_csv(paths["compact_manifest"])
    rows[0]["sha256"] = "a" * 64
    write_csv(paths["compact_manifest"], rows)
    with pytest.raises(RuntimeError, match="compact amplitude manifest"):
        bridge._verify_export_artifact(tmp_path, "vision_router")


def test_export_chain_rejects_reconstruction_source_sha_mismatch(
    tmp_path: Path,
) -> None:
    _write_minimal_export_chain(tmp_path, bad_reconstruction_source_sha=True)
    with pytest.raises(RuntimeError, match="source SHA"):
        bridge._verify_export_artifact(tmp_path, "vision_router")


def test_validate_capture_binds_canonical_ccd_to_exact_amplitude_and_phase_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = "vision_router"
    destination = stage_directory(tmp_path, stage)
    compact = destination / "compact_amplitude"
    amplitude_dir = destination / "amplitude_to_play"
    phase_dir = destination / "phase_to_play"
    ccd_dir = destination / "ccd_captured"
    log_dir = destination / "acquisition_logs"
    for directory in (compact, amplitude_dir, phase_dir, ccd_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    compact_png = compact / "sample.png"
    Image.fromarray(np.full((224, 224), 63, dtype=np.uint8), mode="L").save(
        compact_png
    )
    write_csv(
        destination / "compact_amplitude_manifest.csv",
        [
            {
                "order": 0,
                "key": "sample",
                "filename": compact_png.name,
                "sha256": sha256_file(compact_png),
            }
        ],
    )
    amplitude_bmp = amplitude_dir / "sample.bmp"
    Image.fromarray(np.full((1024, 1024), 63, dtype=np.uint8), mode="L").save(
        amplitude_bmp
    )
    write_csv(
        amplitude_dir / "reconstruction_manifest.csv",
        [
            {
                "order": 0,
                "basename": "sample",
                "source_png": compact_png.name,
                "source_sha256": sha256_file(compact_png),
                "output_bmp": amplitude_bmp.name,
                "output_sha256": sha256_file(amplitude_bmp),
            }
        ],
    )
    phase_bmp = phase_dir / f"{stage}.bmp"
    Image.fromarray(np.zeros((1200, 1920), dtype=np.uint8), mode="L").save(
        phase_bmp
    )
    ccd = ccd_dir / "sample.png"
    ccd_value = np.zeros((478, 478), dtype=np.uint8)
    ccd_value[23, 71] = 213
    Image.fromarray(ccd_value, mode="L").save(ccd)

    transport = destination / "transport_spec.json"
    compact_manifest = destination / "compact_amplitude_manifest.csv"
    reconstruction_manifest = amplitude_dir / "reconstruction_manifest.csv"
    write_json(
        transport,
        {
            "stage": stage,
            "phase_bmp_sha256": sha256_file(phase_bmp),
            "amplitude_manifest_sha256": sha256_file(compact_manifest),
            "amplitude_reconstruction_manifest_sha256": sha256_file(
                reconstruction_manifest
            ),
        },
    )
    acquisition_manifest = log_dir / "capture_manifest.csv"
    acquisition_row = {
        "amplitude_bmp": amplitude_bmp.name,
        "amplitude_bmp_sha256": sha256_file(amplitude_bmp),
        "ccd_capture": ccd.name,
        "output_sha256": sha256_file(ccd),
        "orientation_canonicalized": True,
        "downstream_loader_flip_required": False,
        "phase_mask_sha256": sha256_file(phase_bmp),
        "phase_manifest_verified": True,
    }
    write_csv(acquisition_manifest, [acquisition_row])
    state = {
        "events": [],
        "stages": {name: {} for name in bridge.SIX_STAGES},
    }
    state["stages"][stage]["export"] = {
        "transport_spec_sha256": sha256_file(transport),
        "amplitude_manifest_sha256": sha256_file(compact_manifest),
        "amplitude_reconstruction_manifest_sha256": sha256_file(
            reconstruction_manifest
        ),
        "phase_bmp_sha256": sha256_file(phase_bmp),
    }
    write_json(tmp_path / bridge.STATE_FILENAME, state)
    monkeypatch.setattr(
        bridge,
        "_load_session",
        lambda **_: (read_json(tmp_path / bridge.STATE_FILENAME), []),
    )
    write_csv(
        acquisition_manifest,
        [dict(acquisition_row, amplitude_bmp_sha256="e" * 64)],
    )
    with pytest.raises(RuntimeError, match="Amplitude BMP SHA"):
        bridge.validate_capture(
            config=tmp_path / "config.yaml",
            checkpoint=tmp_path / "checkpoint.pt",
            session_dir=tmp_path,
            stage=stage,
        )
    write_csv(acquisition_manifest, [acquisition_row])
    report = bridge.validate_capture(
        config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "checkpoint.pt",
        session_dir=tmp_path,
        stage=stage,
    )
    assert report["images"] == 1
    assert report["orientation"] == "canonical_model_xy_no_further_flip"

    # The stage is now sealed.  A modified physical CCD cannot be accepted as a
    # replacement under the same session contract.
    Image.fromarray(np.zeros((478, 478), dtype=np.uint8), mode="L").save(ccd)
    with pytest.raises(RuntimeError, match="already sealed"):
        bridge.validate_capture(
            config=tmp_path / "config.yaml",
            checkpoint=tmp_path / "checkpoint.pt",
            session_dir=tmp_path,
            stage=stage,
        )


def test_routing_artifact_rechecks_score_report_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = "vision_router"
    routing_dir = stage_directory(tmp_path, stage) / "routing"
    routing_csv = routing_dir / "routing.csv"
    _write_routing_csv(routing_csv)
    score_report = routing_dir / "routing_report.json"
    write_json(score_report, {"quality_gate_passed": True})
    state = {"stages": {name: {} for name in bridge.SIX_STAGES}}
    state["stages"][stage]["routing"] = {
        "routing_csv_sha256": sha256_file(routing_csv),
        "score_report": str(score_report.resolve()),
        "score_report_sha256": sha256_file(score_report),
    }
    write_json(tmp_path / bridge.STATE_FILENAME, state)
    monkeypatch.setattr(bridge, "_verify_capture_artifact", lambda *_: None)
    bridge._verify_routing_artifact(tmp_path, stage)
    write_json(score_report, {"quality_gate_passed": False})
    with pytest.raises(RuntimeError, match="score report"):
        bridge._verify_routing_artifact(tmp_path, stage)


def test_legacy_finetune_patch_restores_shared_module_globals() -> None:
    original_stages = bridge.legacy_bridge.STAGES
    original_stage_dir = bridge.legacy_bridge._stage_dir
    with bridge._patched_legacy_finetune(target_stage="vision_expert"):
        assert bridge.legacy_bridge.STAGES == bridge.FEATURE_STAGES
        assert bridge.legacy_bridge._stage_dir is stage_directory
    assert bridge.legacy_bridge.STAGES is original_stages
    assert bridge.legacy_bridge._stage_dir is original_stage_dir
