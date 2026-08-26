from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.lab_package import (
    PROJECT_PACKAGE,
    QUICK_SAMPLE_COUNT,
    STAGES,
    _quick210_files,
    create_lab_bundle,
)


def _write(path: Path, value: str | bytes = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LabPackageTests(unittest.TestCase):
    def _make_repo(self, root: Path) -> None:
        project = root / "experiments" / PROJECT_PACKAGE
        runtime = (
            "experiments/__init__.py",
            "experiments/hardware_sdk/__init__.py",
            "experiments/hardware_sdk/devices.py",
            "experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml",
            "experiments/hardware_sdk/workflows/__init__.py",
            "experiments/hardware_sdk/workflows/acquire_folder.py",
            "experiments/hardware_sdk/workflows/calibration_common.py",
            "experiments/hardware_sdk/workflows/reconstruct_slm.py",
            "experiments/hardware_sdk/drivers/__init__.py",
            "experiments/hardware_sdk/drivers/meadowlark_pcie_slm.py",
            "experiments/hardware_sdk/drivers/tucam_camera.py",
        )
        for relative in runtime:
            _write(root / relative)
        reference = (
            "__init__.py",
            "__main__.py",
            "settings.py",
            "optical_blocks.py",
            "modeling.py",
            "run.py",
            "hardware_bridge.py",
            "export_phase_bmps.py",
            "README.md",
            "ARCHITECTURE_AUDIT.md",
            "DATA_PIPELINE.md",
            "RUN_COMMANDS.md",
            "environment_server_and_lab.yml",
            "configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml",
            "configs/release/caltech101_four_layer_optical_quick_last_stage_10x10.yaml",
        )
        for relative in reference:
            _write(project / relative)
        _write(project / "requirements-lab.txt", "numpy\nPillow\nPyYAML\n")
        _write(project / "LAB_BUNDLE.md", "lab readme")

    def _make_phases(self, root: Path) -> Path:
        phases = root / "phase_export"
        _write(
            phases / "phase_export_report.json",
            json.dumps(
                {
                    "stages": list(STAGES),
                    "propagation_distance_m": 0.1,
                }
            ),
        )
        for stage in STAGES:
            _write(phases / "compact_phase" / f"{stage}.png", b"png")
            _write(phases / "phase_bmp" / f"{stage}.bmp", b"bmp")
        return phases

    def _make_quick(self, root: Path, count: int = QUICK_SAMPLE_COUNT) -> Path:
        session = root / "quick"
        rows = [{"key": f"sample_{index:03d}"} for index in range(count)]
        _write_csv(session / "manifest.csv", rows)
        stage = session / "04_language_global"
        _write(
            stage / "transport_spec.json",
            json.dumps(
                {
                    "stage": "language_global",
                    "upstream_source": "simulation",
                    "samples": count,
                }
            ),
        )
        amplitude_rows = []
        for index in range(count):
            name = f"sample_{index:03d}.png"
            amplitude_rows.append({"filename": name})
            _write(stage / "compact_amplitude" / name, b"png")
        _write_csv(stage / "compact_amplitude_manifest.csv", amplitude_rows)
        _write(stage / "compact_phase" / "language_global.png", b"png")
        _write(stage / "phase_to_play" / "language_global.bmp", b"bmp")
        return session

    def test_bundle_contains_only_compact_quick_payload_and_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_repo(root)
            phases = self._make_phases(root)
            quick = self._make_quick(root)
            checkpoint = root / "run" / "ema_best_train_loss_checkpoint.pt"
            _write(checkpoint, b"checkpoint")
            _write(quick / "04_language_global" / "ccd_captured" / "sample_000.png", b"ccd")
            _write(quick / "04_language_global" / "amplitude_to_play" / "sample_000.bmp", b"large")
            _write(root / "run" / "other_checkpoint.pt", b"other")
            _write(root / "run" / "cache" / "feature.bin", b"cache")
            output = root / "bundle.zip"

            report = create_lab_bundle(
                checkpoint=checkpoint,
                phase_export_dir=phases,
                quick_session_dir=quick,
                output_path=output,
                include_vendor_sdk=False,
                repo_root=root,
            )

            self.assertEqual(report["quick_samples"], QUICK_SAMPLE_COUNT)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("bundle_manifest.json"))
            self.assertIn(
                "payload/checkpoint/ema_best_train_loss_checkpoint.pt", names
            )
            self.assertEqual(
                len(
                    [
                        name
                        for name in names
                        if name.startswith(
                            "payload/quick210/04_language_global/compact_amplitude/"
                        )
                    ]
                ),
                QUICK_SAMPLE_COUNT,
            )
            self.assertEqual(
                len(
                    [
                        name
                        for name in names
                        if name.startswith("payload/four_phase_export/phase_bmp/")
                        and name.endswith(".bmp")
                    ]
                ),
                4,
            )
            self.assertFalse(any("ccd_captured" in name for name in names))
            self.assertFalse(any("amplitude_to_play" in name for name in names))
            self.assertFalse(any("other_checkpoint" in name for name in names))
            self.assertFalse(any("feature.bin" in name for name in names))
            self.assertEqual(manifest["hardware_contract"]["quick_samples"], 210)
            self.assertEqual(
                manifest["hardware_contract"]["optical_fusion_minimum"], 0.1
            )
            self.assertFalse(manifest["include_training_evidence"])

    def test_optional_evidence_is_strictly_whitelisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_repo(root)
            phases = self._make_phases(root)
            quick = self._make_quick(root)
            run = root / "run"
            checkpoint = run / "ema_best_train_loss_checkpoint.pt"
            _write(checkpoint, b"checkpoint")
            _write(run / "train_log.csv", "epoch,loss\n1,1.0\n")
            _write(run / "metrics" / "training_latest.json", "{}")
            _write(run / "dataset" / "image.jpg", b"dataset")
            _write(run / "cache" / "model.bin", b"cache")
            output = root / "bundle.zip"

            create_lab_bundle(
                checkpoint=checkpoint,
                phase_export_dir=phases,
                quick_session_dir=quick,
                output_path=output,
                include_evidence=True,
                run_dir=run,
                include_vendor_sdk=False,
                repo_root=root,
            )
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertIn("reference/training_evidence/train_log.csv", names)
            self.assertIn(
                "reference/training_evidence/metrics/training_latest.json", names
            )
            self.assertFalse(any("image.jpg" in name for name in names))
            self.assertFalse(any("model.bin" in name for name in names))

    def test_quick_payload_rejects_non_210_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            quick = self._make_quick(Path(temporary), count=209)
            with self.assertRaisesRegex(ValueError, "must contain 210 samples"):
                list(_quick210_files(quick))


if __name__ == "__main__":
    unittest.main()
