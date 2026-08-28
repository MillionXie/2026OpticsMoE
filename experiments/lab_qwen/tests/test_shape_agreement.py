from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.lab_qwen import shape_agreement as benchmark


@pytest.fixture(scope="module")
def generated_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shape_agreement") / "session"
    report = benchmark.generate_session(root)
    assert report["phase_masks"] == 6
    assert report["amplitude_shapes_per_mask"] == 6
    assert report["expected_captures"] == 36
    return root


def test_shape_suite_is_asymmetric_and_nonempty() -> None:
    amplitudes = benchmark.amplitude_shapes()
    phases = benchmark.phase_shapes_rad()
    assert len(amplitudes) == 6
    assert len(phases) == 6
    for value in amplitudes.values():
        assert value.shape == (478, 478)
        assert value.dtype == np.uint8
        assert int(value.max()) > 0
    for value in phases.values():
        assert value.shape == (478, 478)
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    asymmetric = amplitudes["input_04_letter_L"]
    assert not np.array_equal(asymmetric, np.fliplr(asymmetric))
    assert not np.array_equal(asymmetric, np.flipud(asymmetric))


def test_generator_writes_native_bmps_and_bound_manifests(
    generated_session: Path,
) -> None:
    stage = generated_session / "phase_01_circle_0p75turn"
    phase_paths = list((stage / "phase_to_play").glob("*.bmp"))
    amplitude_paths = list((stage / "amplitude_to_play").glob("*.bmp"))
    assert len(phase_paths) == 1
    assert len(amplitude_paths) == 6
    with Image.open(phase_paths[0]) as image:
        assert image.mode == "L"
        assert image.size == (1920, 1200)
    with Image.open(amplitude_paths[0]) as image:
        assert image.mode == "L"
        assert image.size == (1024, 1024)
    phase_rows = benchmark._read_csv(
        stage / "phase_to_play" / "reconstruction_manifest.csv"
    )
    amplitude_rows = benchmark._read_csv(
        stage / "amplitude_to_play" / "reconstruction_manifest.csv"
    )
    assert phase_rows[0]["active_size_wh"] == "1016,1016"
    assert all(row["active_size_wh"] == "478,478" for row in amplitude_rows)
    assert (generated_session / "RUN_COMMANDS.md").is_file()
    assert (generated_session / "BENCHMARK_PROTOCOL.md").is_file()


def test_numpy_propagator_matches_training_torch_implementation() -> None:
    torch = pytest.importorskip("torch")
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
        AngularSpectrumPropagator,
    )

    amplitude = benchmark.amplitude_shapes()["input_05_notched_triangle"]
    phase = benchmark.phase_shapes_rad()["phase_03_ring_0p5turn"]
    expected = benchmark.angular_spectrum_intensity(amplitude, phase)
    field = np.zeros((benchmark.CANVAS_SIZE, benchmark.CANVAS_SIZE), dtype=np.complex64)
    margin = benchmark.ACTIVE_MARGIN
    field[
        margin : margin + benchmark.ACTIVE_SIZE,
        margin : margin + benchmark.ACTIVE_SIZE,
    ] = (amplitude.astype(np.float32) / 255.0) * np.exp(1j * phase)
    propagator = AngularSpectrumPropagator(
        wavelength_m=benchmark.WAVELENGTH_M,
        pixel_size_m=benchmark.PIXEL_PITCH_M,
        grid_size=benchmark.CANVAS_SIZE,
        distance_m=benchmark.DISTANCE_M,
        k_space_constraint_enabled=True,
        theta_max_deg=benchmark.THETA_MAX_DEG,
    )
    with torch.no_grad():
        observed_field = propagator(torch.from_numpy(field)[None])[0]
        observed = observed_field.abs().square().cpu().numpy()
    observed = observed[
        margin : margin + benchmark.ACTIVE_SIZE,
        margin : margin + benchmark.ACTIVE_SIZE,
    ]
    assert np.allclose(observed, expected, rtol=3.0e-4, atol=2.0e-6)


def test_ideal_canonical_capture_scores_one(generated_session: Path) -> None:
    root = benchmark._read_json(
        generated_session / "shape_agreement_manifest.json"
    )
    detector_hash = "a" * 64
    for item in root["stages"]:
        stage = generated_session / item["directory"]
        contract = benchmark._read_json(stage / "shape_contract.json")
        probes = benchmark._read_csv(stage / "probe_manifest.csv")
        captures = []
        for index, probe in enumerate(probes):
            theory = benchmark._load_reference(
                stage / probe["transport_reference_file"],
                probe["transport_reference_sha256"],
            )
            capture_name = f"{Path(probe['amplitude_file']).stem}.npy"
            capture_path = stage / "ccd_captured" / capture_name
            np.save(capture_path, theory)
            amplitude_path = stage / "amplitude_to_play" / probe["amplitude_file"]
            captures.append(
                {
                    "play_index": index,
                    "amplitude_bmp": probe["amplitude_file"],
                    "amplitude_bmp_sha256": benchmark._sha256_file(amplitude_path),
                    "ccd_capture": capture_name,
                    "output_sha256": benchmark._sha256_file(capture_path),
                    "camera_exposure_us": "5000.0",
                    "detector_geometry_payload_sha256": detector_hash,
                    "orientation_canonicalized": True,
                    "saved_frame_orientation": "canonical_model_xy",
                    "downstream_loader_flip_required": False,
                    "background_subtraction": False,
                    "per_frame_minmax_normalization": False,
                    "phase_mask_sha256": contract["phase_export"][
                        "phase_bmp_sha256"
                    ],
                    "phase_manifest_verified": True,
                }
            )
        manifest = stage / "acquisition_logs" / "capture_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(captures[0]))
            writer.writeheader()
            writer.writerows(captures)

    report = benchmark.evaluate_session(generated_session, make_plots=False)
    assert report["pairs"] == 36
    assert report["metric_rows"] == 144
    assert report["detector_geometry_payload_sha256"] == detector_hash
    assert report["primary_pcc_mean"] == pytest.approx(1.0, abs=1.0e-7)
    assert report["primary_ssim_mean"] == pytest.approx(1.0, abs=1.0e-7)
    assert report["primary_cosine_mean"] == pytest.approx(1.0, abs=1.0e-7)

