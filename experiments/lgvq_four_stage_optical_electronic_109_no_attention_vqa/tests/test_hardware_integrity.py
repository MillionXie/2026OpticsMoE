from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ..hardware_bridge import (
    _ensure_session_manifest,
    _load_sealed_session_manifest,
    _read_csv,
    _sha256,
    _write_csv,
    validate_capture_pass,
)
from ..hardware_contract import PASS_DIRECTORIES


def _payload() -> dict[str, object]:
    return {
        "sample_ids": ["train-a", "train-b", "test-a"],
        "splits": ["train", "train", "test"],
        "targets": [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        "video_paths": ["train-a.mp4", "train-b.mp4", "test-a.mp4"],
    }


def _image(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def _sealed_session(root: Path) -> tuple[Path, list[dict[str, str]]]:
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "frames.pt"
    cache.write_bytes(b"fixed frame cache bytes")
    rows = _ensure_session_manifest(
        root,
        _payload(),
        frame_cache=cache,
        max_train=1,
        max_test=1,
    )
    return cache, rows


def _capture_fixture(
    root: Path,
) -> tuple[list[dict[str, str]], Path, dict[str, Path]]:
    _, rows = _sealed_session(root)
    optical_pass = "stage1_router"
    stage = root / PASS_DIRECTORIES[optical_pass]
    compact_dir = stage / "compact_amplitude"
    amplitude_dir = stage / "amplitude_to_play"
    ccd_dir = stage / "ccd_captured"
    compact_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        key = row["key"]
        compact = compact_dir / f"{key}.png"
        amplitude = amplitude_dir / f"{key}.bmp"
        ccd = ccd_dir / f"{key}.png"
        _image(compact, (478, 478), 40 + index)
        _image(amplitude, (1024, 1024), 40 + index)
        _image(ccd, (478, 478), 80 + index)
        compact_rows.append(
            {
                "order": row["order"],
                "key": key,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "amplitude_file": compact.name,
                "amplitude_sha256": _sha256(compact),
            }
        )
        reconstruction_rows.append(
            {
                "order": index,
                "source_png": compact.name,
                "output_bmp": amplitude.name,
                "source_sha256": _sha256(compact),
                "output_sha256": _sha256(amplitude),
            }
        )
    _write_csv(stage / "amplitude_manifest.csv", compact_rows)
    _write_csv(amplitude_dir / "reconstruction_manifest.csv", reconstruction_rows)

    compact_phase = stage / "compact_phase" / f"{optical_pass}.png"
    phase = stage / "phase_to_play" / f"{optical_pass}.bmp"
    _image(compact_phase, (478, 478), 17)
    _image(phase, (1920, 1200), 17)
    phase_reconstruction = stage / "phase_to_play" / "reconstruction_manifest.csv"
    _write_csv(
        phase_reconstruction,
        [
            {
                "order": 0,
                "source_png": compact_phase.name,
                "output_bmp": phase.name,
                "source_sha256": _sha256(compact_phase),
                "output_sha256": _sha256(phase),
            }
        ],
    )
    _write_csv(
        root / "phase_export_manifest.csv",
        [
            {
                "optical_pass": optical_pass,
                "logical_phase_sha256": _sha256(compact_phase),
                "physical_phase_sha256": _sha256(phase),
            }
        ],
    )

    checkpoint = root / "source.pt"
    checkpoint.write_bytes(b"source checkpoint")
    (stage / "export_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "optical_pass": optical_pass,
                "sample_count": len(rows),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    phase_manifest_sha = _sha256(phase_reconstruction)
    geometry_file_sha = "a" * 64
    geometry_payload_sha = "b" * 64
    acquisition_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        key = row["key"]
        amplitude = amplitude_dir / f"{key}.bmp"
        ccd = ccd_dir / f"{key}.png"
        acquisition_rows.append(
            {
                "play_index": index,
                "amplitude_bmp": amplitude.name,
                "amplitude_bmp_sha256": _sha256(amplitude),
                "ccd_capture": ccd.name,
                "saved_frame_size_wh": json.dumps([478, 478]),
                "saved_dtype": "uint8",
                "output_sha256": _sha256(ccd),
                "detector_geometry_file_sha256": geometry_file_sha,
                "detector_geometry_payload_sha256": geometry_payload_sha,
                "orientation_canonicalized": True,
                "saved_frame_orientation": "canonical_model_xy",
                "downstream_loader_flip_required": False,
                "background_subtraction": False,
                "per_frame_minmax_normalization": False,
                "phase_mask": phase.name,
                "phase_mask_sha256": _sha256(phase),
                "phase_manifest_sha256": phase_manifest_sha,
                "phase_manifest_verified": True,
            }
        )
    acquisition = stage / "acquisition_logs" / "capture_manifest.csv"
    _write_csv(acquisition, acquisition_rows)
    return rows, checkpoint, {
        "stage": stage,
        "ccd": ccd_dir / f"{rows[0]['key']}.png",
        "amplitude": amplitude_dir / f"{rows[0]['key']}.bmp",
        "phase": phase,
        "acquisition": acquisition,
    }


class SessionIdentityTests(unittest.TestCase):
    def test_same_selection_and_cache_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, expected = _sealed_session(root)
            observed = _ensure_session_manifest(
                root,
                _payload(),
                frame_cache=cache,
                max_train=1,
                max_test=1,
            )
            self.assertEqual(observed, expected)
            self.assertEqual(
                _load_sealed_session_manifest(
                    root, _payload(), frame_cache=cache
                ),
                expected,
            )

    def test_changed_sample_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _ = _sealed_session(root)
            with self.assertRaisesRegex(RuntimeError, "sample limits"):
                _ensure_session_manifest(
                    root,
                    _payload(),
                    frame_cache=cache,
                    max_train=2,
                    max_test=1,
                )

    def test_manifest_or_cache_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _ = _sealed_session(root)
            manifest = root / "session_manifest.csv"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "modified"):
                _load_sealed_session_manifest(
                    root, _payload(), frame_cache=cache
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _ = _sealed_session(root)
            cache.write_bytes(b"different frame cache")
            with self.assertRaisesRegex(RuntimeError, "Frame cache SHA"):
                _load_sealed_session_manifest(
                    root, _payload(), frame_cache=cache
                )


class CaptureIntegrityTests(unittest.TestCase):
    def test_valid_capture_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, checkpoint, paths = _capture_fixture(root)
            report = validate_capture_pass(
                root,
                "stage1_router",
                rows,
                expected_checkpoint=checkpoint,
            )
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["orientation"], "canonical_model_xy")
            self.assertTrue((paths["stage"] / "ccd_capture_manifest.csv").is_file())
            repeated = validate_capture_pass(
                root,
                "stage1_router",
                rows,
                expected_checkpoint=checkpoint,
            )
            self.assertEqual(
                repeated["ccd_capture_manifest_sha256"],
                report["ccd_capture_manifest_sha256"],
            )

    def test_modified_ccd_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, checkpoint, paths = _capture_fixture(root)
            _image(paths["ccd"], (478, 478), 250)
            with self.assertRaisesRegex(RuntimeError, "CCD SHA mismatch"):
                validate_capture_pass(
                    root,
                    "stage1_router",
                    rows,
                    expected_checkpoint=checkpoint,
                )

    def test_modified_amplitude_or_phase_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, checkpoint, paths = _capture_fixture(root)
            _image(paths["amplitude"], (1024, 1024), 222)
            with self.assertRaisesRegex(RuntimeError, "Reconstructed amplitude SHA"):
                validate_capture_pass(
                    root,
                    "stage1_router",
                    rows,
                    expected_checkpoint=checkpoint,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, checkpoint, paths = _capture_fixture(root)
            _image(paths["phase"], (1920, 1200), 222)
            with self.assertRaisesRegex(RuntimeError, "Physical phase SHA"):
                validate_capture_pass(
                    root,
                    "stage1_router",
                    rows,
                    expected_checkpoint=checkpoint,
                )

    def test_wrong_phase_in_acquisition_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, checkpoint, paths = _capture_fixture(root)
            acquisition = _read_csv(paths["acquisition"])
            acquisition[0]["phase_mask_sha256"] = "c" * 64
            _write_csv(paths["acquisition"], acquisition)
            with self.assertRaisesRegex(RuntimeError, "Wrong phase SHA"):
                validate_capture_pass(
                    root,
                    "stage1_router",
                    rows,
                    expected_checkpoint=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
