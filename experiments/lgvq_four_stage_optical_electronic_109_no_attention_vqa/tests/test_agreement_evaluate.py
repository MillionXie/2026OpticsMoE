from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ..agreement_evaluate import evaluate_agreement


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AgreementEvaluationTests(unittest.TestCase):
    def _stage(self, root: Path, *, per_frame_minmax: bool = False) -> Path:
        stage = root / "01_stage1_router"
        theory_dir = stage / "theoretical_ccd"
        measured_dir = stage / "ccd_captured"
        theory_dir.mkdir(parents=True)
        measured_dir.mkdir(parents=True)
        key = "test__sample"
        measured = (
            np.arange(478 * 478, dtype=np.uint32).reshape(478, 478) % 251 + 1
        ).astype(np.uint8)
        measured_path = measured_dir / f"{key}.png"
        Image.fromarray(measured, mode="L").save(measured_path)
        np.savez_compressed(
            theory_dir / f"{key}.npz",
            intensity=measured.astype(np.float32) * 7.0,
            optical_pass="stage1_router",
            key=key,
        )
        _write_csv(
            stage / "amplitude_manifest.csv",
            [{"key": key, "amplitude_file": f"{key}.png"}],
        )
        _write_csv(
            stage / "acquisition_logs" / "capture_manifest.csv",
            [
                {
                    "amplitude_bmp": f"{key}.bmp",
                    "ccd_capture": measured_path.name,
                    "output_sha256": hashlib.sha256(measured_path.read_bytes()).hexdigest(),
                    "orientation_canonicalized": True,
                    "saved_frame_orientation": "canonical_model_xy",
                    "downstream_loader_flip_required": False,
                    "background_subtraction": False,
                    "per_frame_minmax_normalization": per_frame_minmax,
                }
            ],
        )
        return stage

    def test_identical_up_to_gain_scores_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = self._stage(Path(directory))
            report = evaluate_agreement(
                stage_dir=stage, optical_pass="stage1_router"
            )
            self.assertEqual(report["sample_count"], 1)
            self.assertAlmostEqual(report["metrics"]["pcc"]["mean"], 1.0, places=6)
            self.assertAlmostEqual(report["metrics"]["ssim"]["mean"], 1.0, places=6)
            self.assertAlmostEqual(
                report["metrics"]["gain_aligned_nmae"]["mean"], 0.0, places=6
            )
            self.assertTrue((stage / "agreement" / "agreement_per_sample.csv").is_file())
            self.assertTrue((stage / "agreement" / "agreement_summary.json").is_file())
            self.assertTrue((stage / "agreement" / "agreement_examples.png").is_file())

    def test_rejects_per_frame_minmax_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = self._stage(Path(directory), per_frame_minmax=True)
            with self.assertRaisesRegex(RuntimeError, "per-frame min/max"):
                evaluate_agreement(stage_dir=stage, optical_pass="stage1_router")


if __name__ == "__main__":
    unittest.main()
