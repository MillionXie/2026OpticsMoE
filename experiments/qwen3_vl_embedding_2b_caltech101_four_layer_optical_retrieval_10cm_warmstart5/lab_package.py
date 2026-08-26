"""Build and verify the independent warmstart5 laboratory transfer bundle.

Only explicit, compact artifacts are selected.  The Caltech101 images, model
caches, CCD captures, reconstructed full-size amplitude frames, and any
checkpoint other than the declared Stage-B EMA checkpoint are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_PACKAGE = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_"
    "warmstart5"
)
ROBUST_PACKAGE = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust"
)
EXPECTED_ARCHITECTURE = "vision2_language2_moe4_10cm_warmstart5_stage_b_v1"
STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
EXPECTED_SPLITS = {"train": 100, "gallery": 10, "test": 100}
EXPECTED_PER_CLASS = {"train": 10, "gallery": 1, "test": 10}
QUICK_SAMPLE_COUNT = 210
TAIL_PARAMETER_COUNT = 255_811
FUSION_MINIMUM = 0.05
FUSION_INITIAL = 0.055
DEFAULT_ZIP_NAME = "qwen_caltech101_10cm_warmstart5_quick210_lab_bundle.zip"


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str
    category: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_require_file(path, "JSON file").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with _require_file(path, "CSV file").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _require_digest(value: Any, expected: str, label: str) -> None:
    observed = str(value or "").strip().lower()
    if observed != expected:
        raise RuntimeError(
            f"{label} checkpoint SHA-256 mismatch: expected {expected}, got {observed!r}"
        )


def _torch_load(path: Path, label: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - server packaging guard
        raise RuntimeError(
            f"PyTorch is required on the server to validate {label}; "
            "it is not required by the capture-only laboratory environment"
        ) from error
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a dictionary: {path}")
    return value


def _tensor_is_finite(value: Any) -> bool:
    import torch

    return not value.is_floating_point() or bool(torch.isfinite(value).all())


def _bounded_gate(raw: float) -> float:
    if raw >= 0.0:
        sigmoid = 1.0 / (1.0 + math.exp(-raw))
    else:
        exp_raw = math.exp(raw)
        sigmoid = exp_raw / (1.0 + exp_raw)
    return FUSION_MINIMUM + (1.0 - FUSION_MINIMUM) * sigmoid


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    payload = _torch_load(path, "selected checkpoint")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Selected checkpoint has no metadata dictionary")
    if metadata.get("optical_architecture") != EXPECTED_ARCHITECTURE:
        raise RuntimeError(
            "Selected checkpoint architecture mismatch: expected "
            f"{EXPECTED_ARCHITECTURE!r}, got {metadata.get('optical_architecture')!r}"
        )
    if metadata.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError(
            "Selected checkpoint must declare test_metrics_used_for_selection=false"
        )
    if metadata.get("selection_criterion") != "minimum_training_total_loss":
        raise RuntimeError("Selected checkpoint was not selected by minimum train loss")
    if metadata.get("weight_variant") != "ema":
        raise RuntimeError("Selected checkpoint must contain EMA weights")

    gate_rows: list[dict[str, Any]] = []
    for branch in ("vision_optical", "language_optical"):
        state = payload.get(branch)
        if not isinstance(state, dict):
            raise RuntimeError(f"Selected checkpoint has no {branch} state dictionary")
        for name, tensor in state.items():
            if not str(name).endswith("_optical_fusion_logit"):
                continue
            if not hasattr(tensor, "numel") or tensor.numel() != 1:
                raise RuntimeError(f"Fusion logit {branch}.{name} is not scalar")
            raw = float(tensor.detach().cpu())
            if not math.isfinite(raw):
                raise RuntimeError(f"Fusion logit {branch}.{name} is non-finite")
            coefficient = _bounded_gate(raw)
            if coefficient + 1.0e-12 < FUSION_MINIMUM:
                raise RuntimeError(f"Fusion coefficient fell below {FUSION_MINIMUM}")
            gate_rows.append(
                {"state_key": f"{branch}.{name}", "coefficient": coefficient}
            )
    if len(gate_rows) != 4:
        raise RuntimeError(
            f"Stage-B checkpoint must expose four optical fusion gates, got {len(gate_rows)}"
        )
    return {
        "epoch": int(payload.get("epoch", -1)),
        "train_loss": float(payload.get("train_loss", float("nan"))),
        "metadata": {
            "optical_architecture": metadata["optical_architecture"],
            "selection_criterion": metadata["selection_criterion"],
            "test_metrics_used_for_selection": False,
            "weight_variant": metadata["weight_variant"],
        },
        "fusion_gates": gate_rows,
    }


def _validate_image(path: Path, size_wh: tuple[int, int], label: str) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if image.size != size_wh:
            raise RuntimeError(
                f"{label} must be {size_wh[0]}x{size_wh[1]}, got {image.size}: {path}"
            )
        if image.mode not in {"L", "P"}:
            raise RuntimeError(f"{label} must be an 8-bit image, got {image.mode}: {path}")


def _phase_export_files(
    phase_export_dir: Path, checkpoint_sha256: str
) -> tuple[list[BundleFile], dict[str, Any]]:
    report_path = _require_file(
        phase_export_dir / "phase_export_report.json", "four-phase export report"
    )
    report = _read_json(report_path)
    _require_digest(report.get("checkpoint_sha256"), checkpoint_sha256, "phase export")
    if tuple(report.get("stages", ())) != STAGES:
        raise RuntimeError("Four-phase report does not contain the ordered four stages")
    if report.get("logical_phase_shape") != [478, 478]:
        raise RuntimeError("Four-phase report must use a 478x478 logical phase")
    if abs(float(report.get("logical_pixel_pitch_um", -1.0)) - 17.0) > 1.0e-9:
        raise RuntimeError("Four-phase report must use the 17 um logical grid")
    if abs(float(report.get("propagation_distance_m", -1.0)) - 0.1) > 1.0e-9:
        raise RuntimeError("Four-phase report must use 10 cm propagation")
    phase_slm = report.get("phase_slm", {})
    if phase_slm.get("size_wh") != [1920, 1200] or abs(
        float(phase_slm.get("pixel_pitch_um", -1.0)) - 8.0
    ) > 1.0e-9:
        raise RuntimeError("Four-phase report does not describe the 1920x1200, 8 um SLM")

    selected = [report_path]
    for stage in STAGES:
        compact = _require_file(
            phase_export_dir / "compact_phase" / f"{stage}.png",
            f"compact phase {stage}",
        )
        native = _require_file(
            phase_export_dir / "phase_bmp" / f"{stage}.bmp",
            f"native phase {stage}",
        )
        _validate_image(compact, (478, 478), "compact phase")
        _validate_image(native, (1920, 1200), "native phase")
        selected.extend((compact, native))
    for optional in (
        phase_export_dir / "phase_preview.png",
        phase_export_dir / "phase_bmp" / "reconstruction_manifest.csv",
        phase_export_dir / "phase_bmp" / "reconstruction_report.json",
    ):
        if optional.is_file():
            selected.append(optional)
    files = [
        BundleFile(
            path,
            (
                Path("payload")
                / "four_phase_export"
                / path.relative_to(phase_export_dir)
            ).as_posix(),
            "four_phase_export",
        )
        for path in selected
    ]
    return files, report


def _validate_quick_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    required = {"order", "key", "split", "sku_index", "sku_name"}
    if len(rows) != QUICK_SAMPLE_COUNT or not rows or not required.issubset(rows[0]):
        raise RuntimeError("quick210 manifest must contain 210 rows and required columns")
    keys: list[str] = []
    split_counts: Counter[str] = Counter()
    class_counts: dict[int, Counter[str]] = defaultdict(Counter)
    class_names: dict[int, str] = {}
    for index, row in enumerate(rows):
        if int(row["order"]) != index:
            raise RuntimeError(f"quick210 order is not contiguous at row {index}")
        key = row["key"]
        split = row["split"]
        label = int(row["sku_index"])
        if not key or Path(key).name != key or key in keys:
            raise RuntimeError(f"quick210 contains invalid/duplicate key {key!r}")
        if split not in EXPECTED_SPLITS:
            raise RuntimeError(f"quick210 contains unsupported split {split!r}")
        if label in class_names and class_names[label] != row["sku_name"]:
            raise RuntimeError(f"quick210 class {label} has inconsistent names")
        class_names[label] = row["sku_name"]
        keys.append(key)
        split_counts[split] += 1
        class_counts[label][split] += 1
    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(f"quick210 split counts are wrong: {dict(split_counts)}")
    if sorted(class_names) != list(range(10)) or any(
        dict(class_counts[label]) != EXPECTED_PER_CLASS for label in range(10)
    ):
        raise RuntimeError("quick210 requires 10/1/10 train/gallery/test per class")
    return {
        "keys": keys,
        "labels": [int(row["sku_index"]) for row in rows],
        "splits": [row["split"] for row in rows],
        "class_names": [class_names[index] for index in range(10)],
    }


def _validate_offline_payload(
    *,
    offline_dir: Path,
    manifest_path: Path,
    rows_info: dict[str, Any],
    checkpoint_sha256: str,
) -> tuple[list[Path], dict[str, Any]]:
    contract_path = _require_file(offline_dir / "contract.json", "offline contract")
    contract = _read_json(contract_path)
    required_values = {
        "schema_version": 1,
        "type": "language_global_quick_offline_full_parity",
        "profile": "quick210",
        "stage": "language_global",
        "checkpoint_architecture": EXPECTED_ARCHITECTURE,
        "upstream_source": "simulation",
        "sample_count": QUICK_SAMPLE_COUNT,
        "manifest_relative_path": "../../manifest.csv",
        "cache_file": "cache.pt",
        "state_file": "downstream_state.pt",
        "tail_trainable_parameter_count": TAIL_PARAMETER_COUNT,
    }
    for key, expected in required_values.items():
        if contract.get(key) != expected:
            raise RuntimeError(
                f"Offline contract {key!r} must be {expected!r}, got {contract.get(key)!r}"
            )
    _require_digest(
        contract.get("source_checkpoint_sha256"), checkpoint_sha256, "offline contract"
    )
    if contract.get("measured_upstream_stages") not in ([], None):
        raise RuntimeError("quick210 must simulate all three upstream stages")
    if contract.get("split_counts") != EXPECTED_SPLITS:
        raise RuntimeError("Offline split counts do not match 100/10/100")
    expected_class_counts = {str(index): EXPECTED_PER_CLASS for index in range(10)}
    if contract.get("class_split_counts") != expected_class_counts:
        raise RuntimeError("Offline per-class split counts do not match 10/1/10")
    if contract.get("class_names") != rows_info["class_names"]:
        raise RuntimeError("Offline class names do not match manifest")
    if _sha256(manifest_path) != contract.get("manifest_sha256"):
        raise RuntimeError("Offline contract manifest SHA-256 mismatch")
    ordered_digest = hashlib.sha256(
        "\n".join(rows_info["keys"]).encode("utf-8")
    ).hexdigest()
    if contract.get("ordered_keys_sha256") != ordered_digest:
        raise RuntimeError("Offline contract ordered key digest mismatch")

    construction = contract.get("tail_construction")
    if not isinstance(construction, dict):
        raise RuntimeError("Offline contract has no tail construction")
    if (
        int(construction.get("width", -1)) != 192
        or int(construction.get("detector_size", -1)) != 478
        or int(construction.get("detector_output_size", -1)) != 224
        or int(construction.get("embedding_dim", -1)) != 64
        or abs(float(construction.get("minimum_optical_fusion", -1.0)) - 0.05)
        > 1.0e-12
    ):
        raise RuntimeError("Offline tail construction is not the formal warmstart5 tail")
    ccd = contract.get("ccd_contract")
    if not isinstance(ccd, dict) or any(
        (
            ccd.get("directory_relative_to_stage") != "ccd_captured",
            ccd.get("mode") != "L",
            ccd.get("dtype") != "uint8",
            ccd.get("shape_hw") != [478, 478],
            ccd.get("background_subtraction") is not False,
            ccd.get("resizing") is not False,
        )
    ):
        raise RuntimeError("Offline CCD contract is not strict 478x478 uint8/no-resize")

    cache_path = _require_file(offline_dir / "cache.pt", "offline latent cache")
    state_path = _require_file(offline_dir / "downstream_state.pt", "offline tail state")
    if _sha256(cache_path) != contract.get("cache_sha256"):
        raise RuntimeError("Offline cache SHA-256 mismatch")
    if _sha256(state_path) != contract.get("state_sha256"):
        raise RuntimeError("Offline downstream state SHA-256 mismatch")
    cache = _torch_load(cache_path, "offline latent cache")
    expected_cache_keys = {
        "packed_block2_inputs",
        "offsets",
        "lengths",
        "labels",
        "split_codes",
        "orders",
    }
    if set(cache) != expected_cache_keys:
        raise RuntimeError("Offline cache tensor key set is wrong")
    import torch

    packed = cache["packed_block2_inputs"]
    if (
        packed.dtype != torch.float32
        or packed.ndim != 2
        or packed.shape[1] != 192
        or not torch.isfinite(packed).all()
    ):
        raise RuntimeError("Offline packed Block-2 inputs are invalid")
    if not torch.equal(cache["orders"], torch.arange(QUICK_SAMPLE_COUNT)):
        raise RuntimeError("Offline cache order tensor is invalid")
    if cache["labels"].tolist() != rows_info["labels"]:
        raise RuntimeError("Offline cache labels disagree with manifest")
    split_codes = {str(k): int(v) for k, v in contract["split_codes"].items()}
    expected_codes = [split_codes[split] for split in rows_info["splits"]]
    if cache["split_codes"].tolist() != expected_codes:
        raise RuntimeError("Offline cache split codes disagree with manifest")
    offsets = cache["offsets"]
    lengths = cache["lengths"]
    if (
        offsets.shape != (QUICK_SAMPLE_COUNT + 1,)
        or lengths.shape != (QUICK_SAMPLE_COUNT,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(packed)
        or not torch.equal(offsets[1:] - offsets[:-1], lengths)
        or bool(torch.any(lengths <= 0))
        or bool(torch.any(lengths > int(construction["max_tokens"])))
    ):
        raise RuntimeError("Offline cache token offsets/lengths are invalid")

    state = _torch_load(state_path, "offline tail state")
    if not state or any(
        not isinstance(value, torch.Tensor) or not _tensor_is_finite(value)
        for value in state.values()
    ):
        raise RuntimeError("Offline tail state must contain only finite tensors")
    observed_parameters = sum(value.numel() for value in state.values())
    if observed_parameters != TAIL_PARAMETER_COUNT:
        raise RuntimeError(
            f"Offline state has {observed_parameters:,} values, expected {TAIL_PARAMETER_COUNT:,}"
        )
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.offline_tail import (
        LanguageGlobalOfflineTail,
    )

    validation_tail = LanguageGlobalOfflineTail(**construction)
    try:
        validation_tail.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Offline downstream state does not exactly match the declared tail"
        ) from error
    if sum(parameter.numel() for parameter in validation_tail.parameters()) != TAIL_PARAMETER_COUNT:
        raise RuntimeError("Declared offline tail does not have exactly 255,811 parameters")
    return [contract_path, cache_path, state_path], contract


def _quick210_files(
    quick_session_dir: Path, checkpoint_sha256: str
) -> tuple[list[BundleFile], dict[str, Any]]:
    manifest_path = _require_file(quick_session_dir / "manifest.csv", "quick manifest")
    rows_info = _validate_quick_manifest(_read_csv(manifest_path))
    stage_dir = quick_session_dir / "04_language_global"
    transport_path = _require_file(stage_dir / "transport_spec.json", "quick transport")
    transport = _read_json(transport_path)
    _require_digest(transport.get("checkpoint_sha256"), checkpoint_sha256, "transport")
    if (
        transport.get("stage") != "language_global"
        or transport.get("upstream_source") != "simulation"
        or int(transport.get("samples", -1)) != QUICK_SAMPLE_COUNT
        or transport.get("measured_upstream_stages") not in ([], None)
    ):
        raise RuntimeError("Quick transport must be 210 Language-global simulation samples")

    amplitude_manifest_path = _require_file(
        stage_dir / "compact_amplitude_manifest.csv", "compact amplitude manifest"
    )
    amplitude_rows = _read_csv(amplitude_manifest_path)
    if len(amplitude_rows) != QUICK_SAMPLE_COUNT:
        raise RuntimeError("Compact amplitude manifest must contain 210 rows")
    amplitude_by_key = {row.get("key", ""): row for row in amplitude_rows}
    if set(amplitude_by_key) != set(rows_info["keys"]):
        raise RuntimeError("Compact amplitude manifest key set differs from quick manifest")
    amplitude_dir = stage_dir / "compact_amplitude"
    amplitude_paths: list[Path] = []
    for key in rows_info["keys"]:
        row = amplitude_by_key[key]
        filename = row.get("filename", "")
        if not filename or Path(filename).name != filename:
            raise RuntimeError(f"Unsafe compact amplitude filename {filename!r}")
        path = _require_file(amplitude_dir / filename, "compact amplitude PNG")
        _validate_image(path, (478, 478), "compact amplitude")
        if row.get("sha256") and row["sha256"] != _sha256(path):
            raise RuntimeError(f"Compact amplitude hash mismatch for {filename}")
        amplitude_paths.append(path)
    disk_names = {path.name for path in amplitude_dir.glob("*.png")}
    if disk_names != {path.name for path in amplitude_paths}:
        raise RuntimeError("compact_amplitude contains undeclared or missing PNG files")

    compact_phase = _require_file(
        stage_dir / "compact_phase" / "language_global.png", "quick compact phase"
    )
    native_phase = _require_file(
        stage_dir / "phase_to_play" / "language_global.bmp", "quick native phase"
    )
    _validate_image(compact_phase, (478, 478), "quick compact phase")
    _validate_image(native_phase, (1920, 1200), "quick native phase")
    offline_paths, offline_contract = _validate_offline_payload(
        offline_dir=stage_dir / "offline_downstream",
        manifest_path=manifest_path,
        rows_info=rows_info,
        checkpoint_sha256=checkpoint_sha256,
    )

    selected = [
        manifest_path,
        transport_path,
        amplitude_manifest_path,
        *amplitude_paths,
        compact_phase,
        native_phase,
        *offline_paths,
    ]
    for optional in (
        stage_dir / "phase_to_play" / "reconstruction_manifest.csv",
        stage_dir / "phase_to_play" / "reconstruction_report.json",
    ):
        if optional.is_file():
            selected.append(optional)
    files = [
        BundleFile(
            path,
            (Path("payload") / "quick210" / path.relative_to(quick_session_dir)).as_posix(),
            "quick210_payload",
        )
        for path in selected
    ]
    return files, {
        "transport": transport,
        "offline_contract": offline_contract,
        "split_counts": EXPECTED_SPLITS,
        "class_names": rows_info["class_names"],
    }


def _runtime_files(repo_root: Path, include_vendor_sdk: bool) -> Iterable[BundleFile]:
    project = f"experiments/{PROJECT_PACKAGE}"
    robust = f"experiments/{ROBUST_PACKAGE}"
    relative_files = (
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
        f"{project}/__init__.py",
        f"{project}/offline_quick_finetune.py",
        f"{project}/requirements-lab.txt",
        f"{project}/requirements-offline-finetune.txt",
        f"{robust}/__init__.py",
        f"{robust}/offline_quick_finetune.py",
        f"{robust}/offline_tail.py",
    )
    for relative in relative_files:
        yield BundleFile(
            _require_file(repo_root / relative, "laboratory runtime file"),
            relative,
            "lab_runtime",
        )
    if not include_vendor_sdk:
        return
    for relative in (
        "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark",
        "experiments/hardware_sdk/vendor_sdk/camera_tucam_mosaic",
    ):
        directory = repo_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required vendor SDK directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            yield BundleFile(
                path, path.relative_to(repo_root).as_posix(), "vendor_sdk"
            )


def _reference_files(repo_root: Path) -> Iterable[BundleFile]:
    root = repo_root / "experiments" / PROJECT_PACKAGE
    names = (
        "__main__.py",
        "settings.py",
        "modeling.py",
        "run.py",
        "hardware_bridge.py",
        "export_phase_bmps.py",
        "lab_package.py",
        "ARCHITECTURE.md",
        "DATA_PIPELINE.md",
        "RUN_COMMANDS.md",
        "FORMAL_RESULT.md",
        "LAB_BUNDLE.md",
        "configs/release/stage1_optical_calibration.yaml",
        "configs/release/stage2_joint_sealed_test.yaml",
        "configs/release/quick_last_stage_10x10.yaml",
    )
    for name in names:
        path = _require_file(root / name, "warmstart5 reference file")
        yield BundleFile(
            path,
            (Path("reference") / "qwen_project_source" / name).as_posix(),
            "qwen_reference_source",
        )


def _evidence_files(
    stage_a_run_dir: Path, stage_b_run_dir: Path
) -> tuple[list[BundleFile], dict[str, Any]]:
    mandatory = {
        "stage_a": (
            stage_a_run_dir,
            ("config.yaml", "train_log.csv", "warmstart_initialization_report.json"),
        ),
        "stage_b": (
            stage_b_run_dir,
            ("config.yaml", "train_log.csv", "metrics/evaluation_summary.json"),
        ),
    }
    optional = (
        "dataset.json",
        "environment.json",
        "model.json",
        "formal_train.log",
        "fast_2h_train.log",
    )
    files: list[BundleFile] = []
    for label, (run_dir, required) in mandatory.items():
        selected = [_require_file(run_dir / name, f"{label} evidence") for name in required]
        selected.extend(run_dir / name for name in optional if (run_dir / name).is_file())
        for path in selected:
            files.append(
                BundleFile(
                    path,
                    (
                        Path("reference")
                        / "training_evidence"
                        / label
                        / path.relative_to(run_dir)
                    ).as_posix(),
                    "training_evidence",
                )
            )
    evaluation = _read_json(stage_b_run_dir / "metrics" / "evaluation_summary.json")
    student = evaluation.get("student")
    if not isinstance(student, dict):
        raise RuntimeError("Stage-B evaluation summary has no student metrics")
    if (
        int(student.get("query_count", -1)) != 200
        or int(student.get("sku_count", -1)) != 10
        or float(student.get("top1_retrieval_accuracy", -1.0)) <= 0.80
    ):
        raise RuntimeError(
            "Formal Stage-B evidence must contain the fixed 200-query result with Top-1 > 0.80"
        )
    return files, {
        "top1_retrieval_accuracy": float(student["top1_retrieval_accuracy"]),
        "top3_retrieval_accuracy": float(student["top3_retrieval_accuracy"]),
        "mrr": float(student["mrr"]),
        "query_count": int(student["query_count"]),
        "gallery_image_count": int(student["gallery_image_count"]),
        "sku_count": int(student["sku_count"]),
    }


def _source_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _deduplicate(files: Iterable[BundleFile]) -> list[BundleFile]:
    result: list[BundleFile] = []
    names: set[str] = set()
    for item in files:
        name = Path(item.archive_path).as_posix().lstrip("/")
        if not name or name.startswith("../") or "/../" in f"/{name}/":
            raise RuntimeError(f"Unsafe archive path {item.archive_path!r}")
        if name in names:
            raise RuntimeError(f"Duplicate archive path {name}")
        names.add(name)
        result.append(BundleFile(item.source.resolve(), name, item.category))
    return result


def _verify_zip(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {row["archive_path"]: row for row in manifest["archive_files"]}
    expected["bundle_manifest.json"] = None
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("ZIP entry set does not exactly match the manifest")
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        embedded = json.loads(archive.read("bundle_manifest.json"))
        if embedded != manifest:
            raise RuntimeError("Embedded bundle manifest changed after serialization")
        for name, row in expected.items():
            if row is None:
                continue
            data = archive.read(name)
            if len(data) != row["size_bytes"] or _sha256_bytes(data) != row["sha256"]:
                raise RuntimeError(f"ZIP content hash mismatch: {name}")

    names = set(expected)
    forbidden_fragments = (
        "/ccd_captured/",
        "/amplitude_to_play/",
        "huggingface_cache",
        "teacher_cache",
        "feature_cache",
    )
    normalized = {f"/{name.lower()}" for name in names}
    if any(fragment in name for name in normalized for fragment in forbidden_fragments):
        raise RuntimeError("ZIP contains a forbidden data/cache directory")
    compact = [
        name
        for name in names
        if name.startswith("payload/quick210/04_language_global/compact_amplitude/")
        and name.endswith(".png")
    ]
    if len(compact) != QUICK_SAMPLE_COUNT:
        raise RuntimeError("ZIP does not contain exactly 210 compact amplitudes")
    allowed_tensor_payloads = {
        "payload/quick210/04_language_global/offline_downstream/cache.pt",
        "payload/quick210/04_language_global/offline_downstream/downstream_state.pt",
        manifest["selected_checkpoint"]["archive_path"],
    }
    observed_tensor_payloads = {
        name
        for name in names
        if Path(name).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    }
    if observed_tensor_payloads != allowed_tensor_payloads:
        raise RuntimeError("ZIP contains an extra or missing tensor/checkpoint payload")
    return {"entry_count": len(names), "crc_and_hash_validation": "passed"}


def create_lab_bundle(
    *,
    checkpoint: str | Path,
    phase_export_dir: str | Path,
    quick_session_dir: str | Path,
    stage_a_run_dir: str | Path,
    stage_b_run_dir: str | Path,
    output_path: str | Path,
    include_vendor_sdk: bool = True,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    checkpoint_path = _require_file(
        Path(checkpoint).expanduser().resolve(), "selected Stage-B checkpoint"
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint_report = _validate_checkpoint(checkpoint_path)
    phase_files, phase_report = _phase_export_files(
        Path(phase_export_dir).expanduser().resolve(), checkpoint_sha256
    )
    quick_files, quick_report = _quick210_files(
        Path(quick_session_dir).expanduser().resolve(), checkpoint_sha256
    )
    evidence_files, fixed_metrics = _evidence_files(
        Path(stage_a_run_dir).expanduser().resolve(),
        Path(stage_b_run_dir).expanduser().resolve(),
    )
    project_root = root / "experiments" / PROJECT_PACKAGE
    readme = _require_file(project_root / "LAB_BUNDLE.md", "laboratory README")
    files: list[BundleFile] = [
        BundleFile(readme, "README_LAB_AND_SERVER.md", "documentation"),
        BundleFile(
            checkpoint_path,
            f"payload/checkpoint/{checkpoint_path.name}",
            "selected_checkpoint",
        ),
        *phase_files,
        *quick_files,
        *evidence_files,
        *_runtime_files(root, include_vendor_sdk),
        *_reference_files(root),
    ]
    files = _deduplicate(files)
    archive_rows = [
        {
            "archive_path": item.archive_path,
            "category": item.category,
            "source": _source_label(item.source, root),
            "size_bytes": item.source.stat().st_size,
            "sha256": _sha256(item.source),
        }
        for item in files
    ]
    selected_archive_path = f"payload/checkpoint/{checkpoint_path.name}"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT_PACKAGE,
        "architecture_contract": {
            "checkpoint_architecture": EXPECTED_ARCHITECTURE,
            "optical_fusion_formula": "0.05 + 0.95 * sigmoid(raw_logit)",
            "minimum_optical_fusion_coefficient": FUSION_MINIMUM,
            "initial_optical_fusion_coefficient": FUSION_INITIAL,
            "coefficient_note": "coefficient floor, not measured optical-energy fraction",
            "current_checkpoint_gates": checkpoint_report["fusion_gates"],
        },
        "hardware_contract": {
            "wavelength_nm": 532.0,
            "propagation_distance_m": 0.1,
            "active_shape_hw": [478, 478],
            "amplitude_slm": {"size_wh": [1024, 1024], "pixel_pitch_um": 17.0},
            "phase_slm": phase_report["phase_slm"],
            "amplitude_polarity": "255=bright/transmissive, 0=dark/blocking",
        },
        "selected_checkpoint": {
            "archive_path": selected_archive_path,
            "sha256": checkpoint_sha256,
            **checkpoint_report,
        },
        "quick210_contract": {
            "stage": "language_global",
            "upstream_source": "simulation",
            "split_counts": EXPECTED_SPLITS,
            "class_count": 10,
            "compact_amplitude_count": QUICK_SAMPLE_COUNT,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "offline_finetune_contract": {
            "qwen_loaded": False,
            "transformers_loaded": False,
            "optical_simulator_loaded": False,
            "trainable_parameters": TAIL_PARAMETER_COUNT,
            "source_checkpoint_sha256": quick_report["offline_contract"][
                "source_checkpoint_sha256"
            ],
            "checkpoint_selection": "minimum_train_loss_only",
        },
        "fixed_simulation_metrics": fixed_metrics,
        "responsibility_split": {
            "capture_only_lab": (
                "reconstruct 1024x1024 amplitude BMP, play amplitude SLM, manually load "
                "the pinned phase BMP, and capture strict 478x478 uint8 CCD PNGs"
            ),
            "optional_lab_offline_tail": (
                "micro-tune only the 255,811-parameter fourth-stage electronic tail; "
                "never loads Qwen or Transformers"
            ),
            "server": (
                "full Qwen training/export, four-stage sequential fine-tuning, and "
                "formal checkpoint generation"
            ),
        },
        "include_vendor_sdk": bool(include_vendor_sdk),
        "exclusion_contract": [
            "no Caltech101 image files",
            "no Hugging Face/Qwen download cache",
            "no teacher or general feature cache",
            "no CCD captures",
            "no reconstructed full-size amplitude_to_play BMPs",
            "no extra checkpoint beyond the selected Stage-B EMA checkpoint",
            "offline_downstream/cache.pt is the sole intentional latent tail cache",
        ],
        "category_file_counts": dict(
            sorted(Counter(row["category"] for row in archive_rows).items())
        ),
        "archive_files": archive_rows,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Bundle exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for item in files:
            archive.write(item.source, item.archive_path)
        archive.writestr("bundle_manifest.json", manifest_bytes)
    zip_validation = _verify_zip(output, manifest)
    report = {
        "zip": str(output),
        "zip_sha256": _sha256(output),
        "zip_size_bytes": output.stat().st_size,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_architecture": EXPECTED_ARCHITECTURE,
        "fixed_test_top1": fixed_metrics["top1_retrieval_accuracy"],
        "quick_samples": QUICK_SAMPLE_COUNT,
        "offline_trainable_parameters": TAIL_PARAMETER_COUNT,
        "vendor_sdk_included": bool(include_vendor_sdk),
        "manifest_entry_count": len(archive_rows),
        "zip_validation": zip_validation,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the strict warmstart5 quick210 laboratory bundle"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase-export-dir", required=True)
    parser.add_argument("--quick-session-dir", required=True)
    parser.add_argument("--stage-a-run-dir", required=True)
    parser.add_argument("--stage-b-run-dir", required=True)
    parser.add_argument("--output", default=DEFAULT_ZIP_NAME)
    parser.add_argument(
        "--omit-vendor-sdk",
        action="store_true",
        help="Test-only compact archive; formal laboratory delivery must include SDKs",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = create_lab_bundle(
        checkpoint=args.checkpoint,
        phase_export_dir=args.phase_export_dir,
        quick_session_dir=args.quick_session_dir,
        stage_a_run_dir=args.stage_a_run_dir,
        stage_b_run_dir=args.stage_b_run_dir,
        output_path=args.output,
        include_vendor_sdk=not args.omit_vendor_sdk,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
