"""Build one complete, short-path Qwen + MNIST-4 optical laboratory ZIP.

This combines a checkpoint-matched quick210 export with current calibration,
agreement, four-stage, capture, evaluation, and plotting assets.  The current
formal build replaces the legacy 5% checkpoint with the selected 1% strong CCD-
noise continuation.  Historical calibration payloads and command documents are
intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ARCHIVE_ROOT = "experiments/lab_qwen"
DEFAULT_NAME = "qwen_mnist4_strong_noise_full_lab.zip"
REQUIRED_FRESNEL = {
    "A_WHITE.bmp",
    "P1_POINT.bmp",
    "P4_POINT.bmp",
    "P9_POINT.bmp",
    "manifest.json",
    "targets.csv",
}


@dataclass(frozen=True)
class Entry:
    archive_path: str
    category: str
    source_path: Path | None = None
    source_zip: Path | None = None
    source_member: str | None = None
    literal: bytes | None = None
    transform: str | None = None

    def read(self) -> bytes:
        if self.literal is not None:
            return self.literal
        if self.source_path is not None:
            data = self.source_path.read_bytes()
        else:
            if self.source_zip is None or self.source_member is None:
                raise RuntimeError(f"Entry has no source: {self.archive_path}")
            with zipfile.ZipFile(self.source_zip) as archive:
                data = archive.read(self.source_member)
        if self.transform == "compact_478_to_native_1024_bmp":
            return _compact_478_to_native_1024_bmp(data, label=self.source_member)
        if self.transform is not None:
            raise RuntimeError(f"Unknown entry transform: {self.transform}")
        return data


def _safe_name(value: str) -> str:
    name = PurePosixPath(value.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError(f"Unsafe archive path: {value!r}")
    return name.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_478_to_native_1024_bmp(data: bytes, *, label: object) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as opened:
        opened.load()
        if opened.mode != "L" or opened.size != (478, 478):
            raise RuntimeError(
                f"Expected 478x478 L compact amplitude, got "
                f"{opened.mode}/{opened.size}: {label}"
            )
        compact = opened.copy()
    native = Image.new("L", (1024, 1024), color=0)
    native.paste(compact, (273, 273))
    buffer = io.BytesIO()
    native.save(buffer, format="BMP")
    return buffer.getvalue()


def _amplitude_reconstruction_manifest(
    records: list[tuple[str, bytes, str, bytes]],
) -> bytes:
    """Describe the exact compact-PNG to ready-BMP transform in the ZIP."""

    rows: list[dict[str, object]] = []
    for order, (source_name, source_data, output_name, output_data) in enumerate(
        records
    ):
        rows.append(
            {
                "order": order,
                "basename": Path(source_name).stem,
                "source_png": source_name,
                "output_bmp": output_name,
                "source_sha256": _sha256_bytes(source_data),
                "output_sha256": _sha256_bytes(output_data),
                "logical_size_wh": "478,478",
                "active_size_wh": "478,478",
                "slm_size_wh": "1024,1024",
                "active_bounds_xyxy": "273,273,751,751",
                "active_center_xy": "512,512",
                "canvas_center_offset_xy": "0,0",
                "mapping_mode": "physical_pitch_nearest",
                "scale_factor": "",
                "logical_pixel_pitch_um": 17.0,
                "slm_pixel_pitch_um": 17.0,
                "physical_ratio": 1.0,
            }
        )
    if not rows:
        raise RuntimeError("Cannot create an empty amplitude reconstruction manifest")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _put(entries: dict[str, Entry], entry: Entry, *, replace: bool = False) -> None:
    name = _safe_name(entry.archive_path)
    if name in entries and not replace:
        raise RuntimeError(f"Duplicate package path: {name}")
    entries[name] = Entry(
        archive_path=name,
        category=entry.category,
        source_path=entry.source_path,
        source_zip=entry.source_zip,
        source_member=entry.source_member,
        literal=entry.literal,
        transform=entry.transform,
    )


def _map_directory(
    entries: dict[str, Entry],
    source: Path,
    destination: str,
    category: str,
    *,
    replace: bool = False,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required {category} directory is missing: {source}")
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.name.casefold() != "readme.md"
    )
    if not files:
        raise RuntimeError(f"Required {category} directory is empty: {source}")
    for path in files:
        _put(
            entries,
            Entry(
                f"{destination}/{path.relative_to(source).as_posix()}",
                category,
                source_path=path,
            ),
            replace=replace,
        )


def _add_ready_bmps_from_compact(
    entries: dict[str, Entry],
    *,
    source_session: Path,
    destination: str,
    category: str,
) -> int:
    compact_files = sorted(source_session.glob("*/compact_amplitude/*.png"))
    stage_records: dict[str, list[tuple[str, bytes, str, bytes]]] = {}
    for compact in compact_files:
        stage_name = compact.parent.parent.name
        source_data = compact.read_bytes()
        output_name = f"{compact.stem}.bmp"
        output_data = _compact_478_to_native_1024_bmp(
            source_data, label=compact
        )
        _put(
            entries,
            Entry(
                f"{destination}/{stage_name}/amplitude_to_play/{output_name}",
                category,
                literal=output_data,
            ),
        )
        stage_records.setdefault(stage_name, []).append(
            (compact.name, source_data, output_name, output_data)
        )
    for stage_name, records in sorted(stage_records.items()):
        _put(
            entries,
            Entry(
                f"{destination}/{stage_name}/amplitude_to_play/"
                "reconstruction_manifest.csv",
                category,
                literal=_amplitude_reconstruction_manifest(records),
            ),
            replace=True,
        )
    return len(compact_files)


def _formal_entries(
    entries: dict[str, Entry],
    formal_zip: Path,
    *,
    use_formal_model: bool = True,
    use_formal_quick: bool = True,
    use_formal_model_reference: bool = True,
) -> None:
    if not formal_zip.is_file():
        raise FileNotFoundError(f"Sealed formal ZIP is missing: {formal_zip}")
    with zipfile.ZipFile(formal_zip) as archive:
        names = set(archive.namelist())
        required = {
            "payload/checkpoint/ema_best_train_loss_checkpoint.pt",
            "payload/quick210/manifest.csv",
            "payload/quick210/04_language_global/offline_downstream/cache.pt",
            "payload/quick210/04_language_global/offline_downstream/downstream_state.pt",
        }
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"Formal ZIP is incomplete: {sorted(missing)}")
        for member in sorted(names):
            if member.endswith("/"):
                continue
            destination: str | None = None
            category = "formal_runtime"
            if (
                use_formal_model
                and member == "payload/checkpoint/ema_best_train_loss_checkpoint.pt"
            ):
                destination = f"{ARCHIVE_ROOT}/model/ema.pt"
                category = "formal_checkpoint"
            elif use_formal_quick and member.startswith("payload/quick210/"):
                destination = f"{ARCHIVE_ROOT}/last/{member[len('payload/quick210/'):]}"
                category = "last_stage_quick210"
            elif (
                use_formal_model_reference
                and member.startswith("payload/four_phase_export/")
            ):
                destination = (
                    f"{ARCHIVE_ROOT}/reference/four_phase_export/"
                    f"{member[len('payload/four_phase_export/'):]}"
                )
                category = "formal_phase_reference"
            elif (
                use_formal_model_reference
                and member.startswith("reference/training_evidence/")
            ):
                destination = f"{ARCHIVE_ROOT}/{member}"
                category = "training_evidence"
            elif (
                use_formal_model_reference
                and member.startswith("reference/fixed_simulation_report/")
            ):
                destination = f"{ARCHIVE_ROOT}/{member}"
                category = "fixed_simulation_report"
            elif member.startswith("experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/"):
                destination = member
                category = "meadowlark_runtime"
            elif member.startswith("experiments/hardware_sdk/vendor_sdk/camera_tucam_mosaic/"):
                destination = member
                category = "tucam_runtime"
            elif member == "bundle_manifest.json":
                destination = f"{ARCHIVE_ROOT}/reference/sealed_bundle_manifest.json"
                category = "formal_provenance"
            # README, historical calibration and copied historical project
            # source/docs are deliberately excluded from the current-only ZIP.
            if destination is not None:
                _put(
                    entries,
                    Entry(
                        destination,
                        category,
                        source_zip=formal_zip,
                        source_member=member,
                    ),
                )
        if not use_formal_quick:
            return
        compact_prefix = "payload/quick210/04_language_global/compact_amplitude/"
        compact_members = sorted(
            member
            for member in names
            if member.startswith(compact_prefix) and member.lower().endswith(".png")
        )
        if len(compact_members) != 210:
            raise RuntimeError(
                f"Formal ZIP must contain 210 compact quick inputs, got {len(compact_members)}"
            )
        reconstruction_records: list[tuple[str, bytes, str, bytes]] = []
        for member in compact_members:
            stem = Path(member).stem
            source_data = archive.read(member)
            output_name = f"{stem}.bmp"
            output_data = _compact_478_to_native_1024_bmp(
                source_data, label=member
            )
            _put(
                entries,
                Entry(
                    f"{ARCHIVE_ROOT}/last/04_language_global/amplitude_to_play/"
                    f"{output_name}",
                    "last_stage_ready_bmp",
                    literal=output_data,
                ),
            )
            reconstruction_records.append(
                (Path(member).name, source_data, output_name, output_data)
            )
        _put(
            entries,
            Entry(
                f"{ARCHIVE_ROOT}/last/04_language_global/amplitude_to_play/"
                "reconstruction_manifest.csv",
                "last_stage_ready_bmp",
                literal=_amplitude_reconstruction_manifest(
                    reconstruction_records
                ),
            ),
            replace=True,
        )


def _mnist_entries(entries: dict[str, Entry], mnist_zip: Path) -> None:
    """Import only the sealed MNIST payload, not its retired device/docs tree."""

    if not mnist_zip.is_file():
        raise FileNotFoundError(f"Sealed MNIST-4 ZIP is missing: {mnist_zip}")
    with zipfile.ZipFile(mnist_zip) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle_manifest.json"))
        records = manifest.get("archive_files")
        if not isinstance(records, list):
            raise RuntimeError("MNIST-4 bundle_manifest.json has no archive_files")
        for record in records:
            member = str(record.get("archive_path", ""))
            if member not in names:
                raise RuntimeError(f"MNIST-4 manifest member is missing: {member}")
            data = archive.read(member)
            if (
                len(data) != int(record.get("size_bytes", -1))
                or _sha256_bytes(data) != record.get("sha256")
            ):
                raise RuntimeError(f"MNIST-4 sealed payload hash failed: {member}")
        payload_members = sorted(
            member
            for member in names
            if member.startswith("payload/") and not member.endswith("/")
        )
        compact = [
            member
            for member in payload_members
            if member.startswith("payload/samples/compact_amplitude/")
            and member.lower().endswith(".png")
        ]
        masks = [
            member
            for member in payload_members
            if member.startswith("payload/phase_masks/")
            and member.lower().endswith(".bmp")
        ]
        if len(compact) != 400 or len(masks) != 4:
            raise RuntimeError(
                "MNIST-4 sealed payload must contain 400 compact amplitudes and "
                f"4 phase masks; got amplitudes={len(compact)}, masks={len(masks)}"
            )
        for member in payload_members:
            _put(
                entries,
                Entry(
                    f"{ARCHIVE_ROOT}/mnist4/{member}",
                    "mnist4_sealed_payload",
                    source_zip=mnist_zip,
                    source_member=member,
                ),
            )
        _put(
            entries,
            Entry(
                f"{ARCHIVE_ROOT}/mnist4/bundle_manifest.json",
                "mnist4_provenance",
                source_zip=mnist_zip,
                source_member="bundle_manifest.json",
            ),
        )


def _selected_noise_evidence(
    entries: dict[str, Entry], noise_run_dir: Path
) -> None:
    """Include auditable metrics without duplicating optimizer/checkpoint tensors."""

    names = (
        "config.yaml",
        "continuation_initialization_report.json",
        "dataset.json",
        "environment.json",
        "model.json",
        "student_metrics.json",
        "train_log.csv",
        "per_sku_metrics.csv",
        "retrieval_results.csv",
        "confusion_matrix.png",
    )
    for name in names:
        source = noise_run_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Required strong-noise evidence is missing: {source}")
        _put(
            entries,
            Entry(
                f"{ARCHIVE_ROOT}/reference/strong_noise_training/{name}",
                "strong_noise_training_evidence",
                source_path=source,
            ),
        )


def _require_session_checkpoint(
    session_dir: Path, expected_sha256: str, *, sequential_four_stage: bool = False
) -> None:
    """Reject mixed exports whose amplitudes/phases came from another model."""

    observed: list[tuple[Path, str]] = []
    for path in sorted(session_dir.rglob("transport_spec.json")):
        value = json.loads(path.read_text(encoding="utf-8")).get("checkpoint_sha256")
        if value:
            observed.append((path, str(value).lower()))
    agreement = session_dir / "agreement_manifest.json"
    if agreement.is_file():
        value = json.loads(agreement.read_text(encoding="utf-8")).get(
            "checkpoint_sha256"
        )
        if value:
            observed.append((agreement, str(value).lower()))
    if not observed:
        raise RuntimeError(f"Session has no checkpoint-bound transport contract: {session_dir}")
    expected_by_stage = {"01_vision_expert": expected_sha256}
    if sequential_four_stage:
        for stage, previous in (
            ("02_vision_global", "vision_expert"),
            ("03_language_expert", "vision_global"),
            ("04_language_global", "language_expert"),
        ):
            checkpoint = session_dir / "checkpoints" / f"after_{previous}.pt"
            if (session_dir / stage / "transport_spec.json").is_file():
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Sequential stage {stage} requires checkpoint {checkpoint}"
                    )
                expected_by_stage[stage] = _sha256_file(checkpoint)
    mismatched: list[str] = []
    for path, digest in observed:
        if path.name == "agreement_manifest.json":
            expected = expected_sha256
        elif sequential_four_stage:
            expected = expected_by_stage.get(path.parent.name)
            if expected is None:
                raise RuntimeError(f"Unknown four-stage transport directory: {path.parent}")
        else:
            expected = expected_sha256
        if digest != expected:
            mismatched.append(str(path))
    if mismatched:
        raise RuntimeError(
            "Session was exported from a different checkpoint: "
            f"expected={expected_sha256}, files={mismatched}"
        )


def _current_source_files(repo_root: Path) -> Iterable[Path]:
    required = (
        "experiments/__init__.py",
        "experiments/hardware_sdk/__init__.py",
        "experiments/hardware_sdk/devices.py",
        "experiments/hardware_sdk/drivers/__init__.py",
        "experiments/hardware_sdk/drivers/meadowlark_pcie_slm.py",
        "experiments/hardware_sdk/drivers/tucam_camera.py",
        "experiments/hardware_sdk/workflows/__init__.py",
        "experiments/hardware_sdk/workflows/acquire_folder.py",
        "experiments/hardware_sdk/workflows/amplitude_lut_calibration.py",
        "experiments/hardware_sdk/workflows/calibration_common.py",
        "experiments/hardware_sdk/workflows/detector_homography.py",
        "experiments/hardware_sdk/workflows/reconstruct_slm.py",
        "experiments/hardware_sdk/workflows/roi_calibration.py",
        "experiments/hardware_sdk/demos/__init__.py",
        "experiments/hardware_sdk/demos/phase_slm_demo.py",
        "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/slm7930_at532_30C.lut",
        "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/slm7930_at532_70C.lut",
        "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/slm7930_at532-70c-pixel-2.lut",
        "experiments/lab_qwen/__init__.py",
        "experiments/lab_qwen/prepare_lab.py",
        "experiments/lab_qwen/shape_agreement.py",
        "experiments/lab_qwen/mnist_timing_diagnostic.py",
        "experiments/lab_qwen/local_four_stage.py",
        "experiments/lab_qwen/COMMANDS.md",
        "experiments/lab_qwen/README.md",
        "experiments/lab_qwen/LAB_CONFIG.yaml",
        "experiments/lab_qwen/internal/hardware_template.yaml",
        "experiments/d2nn_mnist4_single_layer_17um_10cm/__init__.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm/settings.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm/ccd_evaluate.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/__init__.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/io_utils.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/settings.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/modeling.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/lab_session.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/lab_pipeline.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/ccd_evaluate.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/simulation_agreement.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/paper_evaluation.py",
        "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/requirements-lab.txt",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/__init__.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/agreement_common.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/agreement_evaluate.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/agreement_report.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/offline_quick_finetune.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/result_report.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/requirements-lab.txt",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/requirements-offline-finetune.txt",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/requirements-local-four-stage.txt",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/__init__.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/offline_quick_finetune.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/offline_tail.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/__init__.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/modeling.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/settings.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/hardware_bridge.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/agreement_export.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/export_phase_bmps.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/offline_quick_finetune.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/agreement_evaluate.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/agreement_report.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/result_report.py",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/common.yaml",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/noise_strong.yaml",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/quick_last_stage_10x10.yaml",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/agreement_quick_language_global.yaml",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/stage_hardware_canonical_ccd.yaml",
    )
    seen: set[Path] = set()
    for relative in required:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required laboratory source is missing: {path}")
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path

    # Full local stages load frozen Qwen and the compact optical replacement.
    # Package only executable Python and inherited YAML contracts, not tests,
    # old run outputs, caches, or documentation from these source projects.
    runtime_packages = (
        "experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval",
        "experiments/qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval",
        "experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval",
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1",
    )
    for relative in runtime_packages:
        package = repo_root / relative
        if not package.is_dir():
            raise FileNotFoundError(f"Required local-finetune package is missing: {package}")
        for path in sorted(package.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in {".py", ".yaml", ".txt"}
                or "tests" in path.parts
                or "__pycache__" in path.parts
            ):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def create_bundle(
    *,
    repo_root: Path,
    formal_zip: Path,
    mnist_zip: Path,
    fresnel_dir: Path,
    dual_dir: Path,
    exposure_dir: Path,
    agreement_session: Path,
    four_session: Path,
    output: Path,
    overwrite: bool,
    model_checkpoint: Path | None = None,
    last_session: Path | None = None,
    noise_run_dir: Path | None = None,
    phase_export_dir: Path | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    entries: dict[str, Entry] = {}
    selected_checkpoint = model_checkpoint.resolve() if model_checkpoint else None
    selected_checkpoint_sha256 = (
        _sha256_file(selected_checkpoint) if selected_checkpoint else None
    )
    _formal_entries(
        entries,
        formal_zip.resolve(),
        use_formal_model=selected_checkpoint is None,
        use_formal_quick=last_session is None,
        use_formal_model_reference=(
            selected_checkpoint is None and noise_run_dir is None
        ),
    )
    _mnist_entries(entries, mnist_zip.resolve())

    if selected_checkpoint is not None:
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(
                f"Selected strong-noise checkpoint is missing: {selected_checkpoint}"
            )
        _put(
            entries,
            Entry(
                f"{ARCHIVE_ROOT}/model/ema.pt",
                "strong_noise_checkpoint",
                source_path=selected_checkpoint,
            ),
            replace=True,
        )
    if last_session is not None:
        last_session = last_session.resolve()
        if selected_checkpoint_sha256 is None:
            raise ValueError("last_session requires model_checkpoint")
        _require_session_checkpoint(last_session, selected_checkpoint_sha256)
        _map_directory(
            entries,
            last_session,
            f"{ARCHIVE_ROOT}/last",
            "strong_noise_last_stage",
            replace=True,
        )
        last_ready = _add_ready_bmps_from_compact(
            entries,
            source_session=last_session,
            destination=f"{ARCHIVE_ROOT}/last",
            category="strong_noise_last_stage_ready_bmp",
        )
        if last_ready != 210:
            raise RuntimeError(
                f"Strong-noise last-stage session must contain 210 inputs, got {last_ready}"
            )
    if noise_run_dir is not None:
        _selected_noise_evidence(entries, noise_run_dir.resolve())
    if phase_export_dir is not None:
        _map_directory(
            entries,
            phase_export_dir.resolve(),
            f"{ARCHIVE_ROOT}/reference/four_phase_export",
            "strong_noise_phase_reference",
            replace=True,
        )

    observed_fresnel = {path.name for path in fresnel_dir.iterdir() if path.is_file()}
    missing_fresnel = REQUIRED_FRESNEL.difference(observed_fresnel)
    if missing_fresnel:
        raise RuntimeError(f"Current Fresnel payload is incomplete: {sorted(missing_fresnel)}")
    _map_directory(entries, fresnel_dir, f"{ARCHIVE_ROOT}/calib/fresnel", "fresnel_v4")
    _put(
        entries,
        Entry(
            f"{ARCHIVE_ROOT}/calib/fresnel/amplitude_manifest.csv",
            "fresnel_v4",
            literal=b"amplitude_bmp\r\nA_WHITE.bmp\r\n",
        ),
    )
    _map_directory(entries, dual_dir, f"{ARCHIVE_ROOT}/calib/dual", "dual_slm_k1")
    # Each pair folder also contains a phase BMP.  Bind an amplitude-only CSV
    # allowlist so acquire_folder never tries to send the 1920x1200 phase frame
    # to the 1024x1024 Meadowlark amplitude SLM.
    for pair_directory in (
        "01_checker_c64",
        "02_large_blocks_c48_x",
        "03_large_blocks_c48_y",
    ):
        _put(
            entries,
            Entry(
                f"{ARCHIVE_ROOT}/calib/dual/{pair_directory}/amplitude_manifest.csv",
                "dual_slm_k1",
                literal=b"amplitude_bmp\r\namplitude_1024x1024.bmp\r\n",
            ),
        )
    _map_directory(entries, exposure_dir, f"{ARCHIVE_ROOT}/calib/exposure", "exposure_32x3")
    _map_directory(entries, agreement_session, f"{ARCHIVE_ROOT}/agree", "agreement_session")
    _map_directory(entries, four_session, f"{ARCHIVE_ROOT}/four", "four_stage_initial")
    if selected_checkpoint_sha256 is not None:
        _require_session_checkpoint(
            agreement_session.resolve(), selected_checkpoint_sha256
        )
        _require_session_checkpoint(
            four_session.resolve(),
            selected_checkpoint_sha256,
            sequential_four_stage=True,
        )
    agreement_ready = _add_ready_bmps_from_compact(
        entries,
        source_session=agreement_session,
        destination=f"{ARCHIVE_ROOT}/agree",
        category="agreement_ready_bmp",
    )
    four_ready = _add_ready_bmps_from_compact(
        entries,
        source_session=four_session,
        destination=f"{ARCHIVE_ROOT}/four",
        category="four_stage_ready_bmp",
    )
    if agreement_ready == 0:
        raise RuntimeError("Agreement session contains no compact amplitudes to reconstruct")
    if four_ready == 0:
        raise RuntimeError("Four-stage session contains no compact amplitudes to reconstruct")

    for path in _current_source_files(root):
        relative = path.relative_to(root).as_posix()
        _put(
            entries,
            Entry(relative, "current_source", source_path=path),
            replace=True,
        )

    # The all-white frame must be physically all 255, not merely named so.
    try:
        from PIL import Image
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Pillow and NumPy are required to validate calibration BMPs") from error
    white = np.asarray(Image.open(fresnel_dir / "A_WHITE.bmp"))
    if white.shape != (1024, 1024) or not bool(np.all(white == 255)):
        raise RuntimeError("A_WHITE.bmp must be 1024x1024 with every pixel equal to 255")

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name, entry in sorted(entries.items()):
            data = entry.read()
            archive.writestr(name, data)
            records.append(
                {
                    "path": name,
                    "category": entry.category,
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                }
            )
        manifest = {
            "schema_version": 1,
            "type": "qwen_mnist4_complete_short_path_optical_lab_bundle",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "archive_root": ARCHIVE_ROOT,
            "formal_source_zip_sha256": _sha256_file(formal_zip),
            "mnist4_source_zip_sha256": _sha256_file(mnist_zip),
            "selected_model": {
                "kind": (
                    "strong_truncated_biased_gaussian_ccd_noise"
                    if selected_checkpoint is not None
                    else "formal_source_checkpoint"
                ),
                "sha256": selected_checkpoint_sha256,
                "noise_mean_fraction_of_clean_frame_mean": (
                    0.06 if selected_checkpoint is not None else None
                ),
                "noise_std_fraction_of_clean_frame_mean": (
                    0.05 if selected_checkpoint is not None else None
                ),
                "minimum_optical_fusion_coefficient": (
                    0.01 if selected_checkpoint is not None else None
                ),
            },
            "workflow": {
                "dual_slm_calibration": True,
                "fresnel_ccd_calibration": (
                    "full-white amplitude; teacher-MATLAB square Fresnel arrays"
                ),
                "brightness_calibration": "32 gray levels x 3 frames",
                "amplitude_lut_recalibration": (
                    "64 gray levels x 3 frames for base and generated LUT; "
                    "isotonic monotonic-branch fit plus piecewise-linear inverse"
                ),
                "sim_to_real_agreement": True,
                "last_stage_quick210": True,
                "four_stage": (
                    "initial stage included; all four measured-CCD fine-tunes and "
                    "next-stage exports can run locally with development-selected checkpoints"
                ),
                "mnist4_simple_task": "quick40 diagnostic + formal400 reportable evaluation + played-BMP sim-to-real agreement",
                "mnist4_timing_diagnostic": (
                    "5 configurable SLM settle delays x 4 fixed digits; "
                    "per-stage timing log and canonical four-ROI orientation overlay"
                ),
            },
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        archive.writestr(
            f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    digest = _sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    result = verify_bundle(output)
    result.update({"sha256": digest, "bytes": output.stat().st_size})
    return result


def verify_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP contains duplicate paths")
        manifest_name = f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json"
        manifest = json.loads(archive.read(manifest_name))
        for record in manifest["files"]:
            data = archive.read(record["path"])
            if len(data) != int(record["bytes"]) or _sha256_bytes(data) != record["sha256"]:
                raise RuntimeError(f"ZIP verification failed: {record['path']}")
        required = {
            f"{ARCHIVE_ROOT}/COMMANDS.md",
            f"{ARCHIVE_ROOT}/LAB_CONFIG.yaml",
            f"{ARCHIVE_ROOT}/internal/hardware_template.yaml",
            f"{ARCHIVE_ROOT}/prepare_lab.py",
            f"{ARCHIVE_ROOT}/shape_agreement.py",
            f"{ARCHIVE_ROOT}/mnist_timing_diagnostic.py",
            f"{ARCHIVE_ROOT}/local_four_stage.py",
            f"{ARCHIVE_ROOT}/calib/fresnel/A_WHITE.bmp",
            f"{ARCHIVE_ROOT}/calib/fresnel/P1_POINT.bmp",
            f"{ARCHIVE_ROOT}/calib/fresnel/P4_POINT.bmp",
            f"{ARCHIVE_ROOT}/calib/fresnel/P9_POINT.bmp",
            f"{ARCHIVE_ROOT}/calib/dual/k1_pair_manifest.json",
            f"{ARCHIVE_ROOT}/calib/dual/01_checker_c64/amplitude_manifest.csv",
            f"{ARCHIVE_ROOT}/calib/dual/02_large_blocks_c48_x/amplitude_manifest.csv",
            f"{ARCHIVE_ROOT}/calib/dual/03_large_blocks_c48_y/amplitude_manifest.csv",
            f"{ARCHIVE_ROOT}/agree/agreement_manifest.json",
            f"{ARCHIVE_ROOT}/last/04_language_global/offline_downstream/cache.pt",
            f"{ARCHIVE_ROOT}/model/ema.pt",
            f"{ARCHIVE_ROOT}/mnist4/payload/samples/quick40.csv",
            f"{ARCHIVE_ROOT}/mnist4/payload/samples/formal400.csv",
            f"{ARCHIVE_ROOT}/mnist4/payload/phase_masks/phase_masks.json",
            "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/simulation_agreement.py",
            "experiments/hardware_sdk/workflows/reconstruct_slm.py",
            "experiments/hardware_sdk/workflows/amplitude_lut_calibration.py",
            "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/requirements-local-four-stage.txt",
            "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_bridge.py",
            "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/slm7930_at532_30C.lut",
            "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/slm7930_at532_70C.lut",
            "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/slm7930_at532-70c-pixel-2.lut",
        }
        selected_model = manifest.get("selected_model", {})
        if selected_model.get("kind") == (
            "strong_truncated_biased_gaussian_ccd_noise"
        ):
            required.update(
                {
                    f"{ARCHIVE_ROOT}/reference/strong_noise_training/student_metrics.json",
                    f"{ARCHIVE_ROOT}/reference/strong_noise_training/train_log.csv",
                    f"{ARCHIVE_ROOT}/reference/four_phase_export/phase_export_report.json",
                    "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/hardware_bridge.py",
                    "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/stage_hardware_canonical_ccd.yaml",
                }
            )
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"ZIP required-data check failed: {sorted(missing)}")
        if selected_model.get("kind") == (
            "strong_truncated_biased_gaussian_ccd_noise"
        ):
            model_digest = _sha256_bytes(
                archive.read(f"{ARCHIVE_ROOT}/model/ema.pt")
            )
            if model_digest != selected_model.get("sha256"):
                raise RuntimeError("Strong-noise model SHA-256 differs from manifest")
        fresnel = json.loads(
            archive.read(f"{ARCHIVE_ROOT}/calib/fresnel/manifest.json")
        )
        if fresnel.get("source_principle") != (
            "teacher MATLAB quadratic square-window array"
        ):
            raise RuntimeError("ZIP does not contain the current teacher-MATLAB Fresnel set")
        forbidden_fragments = (
            "amplitude_holoeye",
            "fresnel_roi_vertex",
            "fresnel_square_aperture",
            "LAB_VALIDATION_BUNDLE",
        )
        forbidden = [
            name for name in names if any(fragment in name for fragment in forbidden_fragments)
        ]
        if forbidden:
            raise RuntimeError(
                "ZIP contains retired hardware/calibration material: "
                f"{forbidden[:5]}"
            )
        operational_markdown = [
            name
            for name in names
            if name.startswith("experiments/")
            and not name.startswith(f"{ARCHIVE_ROOT}/reference/")
            and name.lower().endswith(".md")
        ]
        expected_markdown = {
            f"{ARCHIVE_ROOT}/COMMANDS.md",
            f"{ARCHIVE_ROOT}/README.md",
        }
        if set(operational_markdown) != expected_markdown:
            raise RuntimeError(
                "ZIP must expose exactly one command document and one short README; "
                f"got {operational_markdown}"
            )
        four_amplitudes = [
            name for name in names if name.startswith(f"{ARCHIVE_ROOT}/four/") and "/amplitude_to_play/" in name and name.lower().endswith(".bmp")
        ]
        if not four_amplitudes:
            raise RuntimeError("ZIP contains no ready-to-play four-stage amplitude BMP")
        last_amplitudes = [
            name for name in names if name.startswith(f"{ARCHIVE_ROOT}/last/04_language_global/amplitude_to_play/") and name.lower().endswith(".bmp")
        ]
        if len(last_amplitudes) != 210:
            raise RuntimeError(f"ZIP must contain 210 last-stage inputs, got {len(last_amplitudes)}")
        ready_amplitudes = [
            name
            for name in names
            if name.lower().endswith(".bmp")
            and "/amplitude_to_play/" in name
            and name.startswith(
                (
                    f"{ARCHIVE_ROOT}/agree/",
                    f"{ARCHIVE_ROOT}/four/",
                    f"{ARCHIVE_ROOT}/last/",
                )
            )
        ]
        for parent in sorted({name.rsplit("/", 1)[0] for name in ready_amplitudes}):
            manifest_name = f"{parent}/reconstruction_manifest.csv"
            if manifest_name not in names:
                raise RuntimeError(
                    f"ZIP ready amplitudes have no reconstruction manifest: {parent}"
                )
            rows = list(
                csv.DictReader(
                    io.StringIO(archive.read(manifest_name).decode("utf-8-sig"))
                )
            )
            expected_outputs = {
                name.rsplit("/", 1)[1]
                for name in ready_amplitudes
                if name.rsplit("/", 1)[0] == parent
            }
            observed_outputs = {str(row.get("output_bmp", "")) for row in rows}
            if observed_outputs != expected_outputs or len(rows) != len(expected_outputs):
                raise RuntimeError(
                    f"ZIP reconstruction manifest does not exactly bind {parent}"
                )
            compact_parent = parent.rsplit("/amplitude_to_play", 1)[0]
            for row in rows:
                output_path = f"{parent}/{row['output_bmp']}"
                source_path = f"{compact_parent}/compact_amplitude/{row['source_png']}"
                if source_path not in names:
                    raise RuntimeError(
                        f"ZIP reconstruction source is missing: {source_path}"
                    )
                if _sha256_bytes(archive.read(output_path)) != row["output_sha256"]:
                    raise RuntimeError(
                        f"ZIP reconstruction output hash failed: {output_path}"
                    )
                if _sha256_bytes(archive.read(source_path)) != row["source_sha256"]:
                    raise RuntimeError(
                        f"ZIP reconstruction source hash failed: {source_path}"
                    )
        mnist_amplitudes = [
            name
            for name in names
            if name.startswith(f"{ARCHIVE_ROOT}/mnist4/payload/samples/compact_amplitude/")
            and name.lower().endswith(".png")
        ]
        mnist_masks = [
            name
            for name in names
            if name.startswith(f"{ARCHIVE_ROOT}/mnist4/payload/phase_masks/")
            and name.lower().endswith(".bmp")
        ]
        if len(mnist_amplitudes) != 400 or len(mnist_masks) != 4:
            raise RuntimeError(
                f"ZIP MNIST payload is incomplete: amplitudes={len(mnist_amplitudes)}, "
                f"masks={len(mnist_masks)}"
            )
    return {
        "path": str(path),
        "files": len(names),
        "last_stage_inputs": len(last_amplitudes),
        "four_stage_amplitude_bmps": len(four_amplitudes),
        "mnist4_compact_amplitudes": len(mnist_amplitudes),
        "mnist4_phase_masks": len(mnist_masks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-zip")
    mode.add_argument("--create", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--formal-zip")
    parser.add_argument("--mnist-zip")
    parser.add_argument("--fresnel-dir")
    parser.add_argument("--dual-dir")
    parser.add_argument("--exposure-dir", default="experiments/lab_qwen/calib/exposure")
    parser.add_argument("--agreement-session")
    parser.add_argument("--four-session")
    parser.add_argument(
        "--model-checkpoint",
        help="Optional strong-noise EMA checkpoint replacing the formal source model",
    )
    parser.add_argument(
        "--last-session",
        help="Checkpoint-matched 210-sample language-global export replacing formal quick210",
    )
    parser.add_argument(
        "--noise-run-dir",
        help="Strong-noise run directory whose non-checkpoint evidence is archived",
    )
    parser.add_argument(
        "--phase-export-dir",
        help="Checkpoint-matched four-mask reference export",
    )
    parser.add_argument("--output", default=DEFAULT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.verify_zip:
        report = verify_bundle(Path(args.verify_zip).expanduser().resolve())
    else:
        required = {
            "formal_zip": args.formal_zip,
            "mnist_zip": args.mnist_zip,
            "fresnel_dir": args.fresnel_dir,
            "dual_dir": args.dual_dir,
            "agreement_session": args.agreement_session,
            "four_session": args.four_session,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"creation requires {', '.join('--' + name.replace('_', '-') for name in missing)}")
        root = Path(args.repo_root).expanduser().resolve()
        report = create_bundle(
            repo_root=root,
            formal_zip=Path(args.formal_zip).expanduser().resolve(),
            mnist_zip=Path(args.mnist_zip).expanduser().resolve(),
            fresnel_dir=Path(args.fresnel_dir).expanduser().resolve(),
            dual_dir=Path(args.dual_dir).expanduser().resolve(),
            exposure_dir=(root / args.exposure_dir).resolve(),
            agreement_session=Path(args.agreement_session).expanduser().resolve(),
            four_session=Path(args.four_session).expanduser().resolve(),
            output=Path(args.output).expanduser().resolve(),
            overwrite=args.overwrite,
            model_checkpoint=(
                Path(args.model_checkpoint).expanduser().resolve()
                if args.model_checkpoint
                else None
            ),
            last_session=(
                Path(args.last_session).expanduser().resolve()
                if args.last_session
                else None
            ),
            noise_run_dir=(
                Path(args.noise_run_dir).expanduser().resolve()
                if args.noise_run_dir
                else None
            ),
            phase_export_dir=(
                Path(args.phase_export_dir).expanduser().resolve()
                if args.phase_export_dir
                else None
            ),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
