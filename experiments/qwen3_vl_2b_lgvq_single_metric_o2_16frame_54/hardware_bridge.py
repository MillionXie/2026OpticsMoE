"""Export, acquire, validate and fine-tune the exact LGVQ single-metric model.

This is the laboratory entry point for the 9/16/36-frame Temporal models and
the 4-frame Spatial model. It consumes the same cached Qwen-front tensors as
the simulator and replaces a contiguous prefix of the six optical passes with
canonical 478x478 CCD captures. It never loads a Transformer or Attention
module at inference/fine-tuning time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
)

from .data import file_sha256, load_single_metric_cache
from .export_hardware_masks import build_stage_planes
from .hardware_contract import FUSION_STAGES, OPTICAL_PASSES, PASS_DIRECTORIES, forward_hardware
from .metrics import regression_metrics
from .modeling import LGVQSingleMetricOEO16, build_model
from .settings import ExperimentSettings, load_settings, resolved_dict
from .training import batch_correlation_loss, pairwise_ranking_loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_model(
    settings: ExperimentSettings, checkpoint: Path, device: torch.device
) -> tuple[LGVQSingleMetricOEO16, dict[str, Any]]:
    checkpoint = checkpoint.expanduser().resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise RuntimeError(f"Unsupported checkpoint: {checkpoint}")
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError("Checkpoint/config architecture mismatch")
    if saved.get("target_name") != settings.target_name:
        raise RuntimeError("Spatial and Temporal checkpoints cannot be interchanged")
    model = build_model(settings)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device), saved


def _safe_key(index: int, split: str, sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{split}__{index:05d}__{digest}"


def _select_rows(
    payload: Mapping[str, Any], max_train: int | None, max_test: int | None
) -> list[dict[str, Any]]:
    limits = {"train": max_train, "test": max_test}
    counts: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for index, (sample_id, split, target, video) in enumerate(
        zip(
            payload["sample_ids"],
            payload["splits"],
            payload["targets"],
            payload["video_paths"],
        )
    ):
        split = str(split)
        if split not in limits:
            continue
        if limits[split] is not None and counts[split] >= int(limits[split]):
            continue
        rows.append(
            {
                "order": len(rows),
                "cache_index": index,
                "key": _safe_key(index, split, str(sample_id)),
                "sample_id": str(sample_id),
                "split": split,
                "target": float(target),
                "video_path": str(video),
            }
        )
        counts[split] += 1
    if counts["train"] == 0 or counts["test"] == 0:
        raise RuntimeError("Hardware session requires train and test samples")
    return rows


def _session_rows(
    session_dir: Path,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    *,
    max_train: int | None,
    max_test: int | None,
) -> list[dict[str, Any]]:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "session_manifest.csv"
    identity_path = session_dir / "session_identity.json"
    selected = _select_rows(payload, max_train, max_test)
    identity = {
        "schema_version": 1,
        "target_name": settings.target_name,
        "architecture": settings.architecture_label,
        "manifest_sha256": file_sha256(settings.manifest_path),
        "vision_cache_sha256": file_sha256(settings.vision_cache_path),
        "language_cache_sha256": file_sha256(settings.language_cache_path),
        "selection": {"max_train": max_train, "max_test": max_test},
        "sample_count": len(selected),
    }
    if path.is_file() or identity_path.is_file():
        if not path.is_file() or not identity_path.is_file():
            raise RuntimeError("Incomplete sealed session identity")
        prior = json.loads(identity_path.read_text(encoding="utf-8"))
        if prior != identity:
            raise RuntimeError("Requested cache/config/sample selection differs from sealed session")
        rows = _read_csv(path)
        comparable = [
            {
                **row,
                "order": int(row["order"]),
                "cache_index": int(row["cache_index"]),
                "target": float(row["target"]),
            }
            for row in rows
        ]
        if comparable != selected:
            raise RuntimeError("session_manifest.csv was changed after sealing")
        return selected
    _write_csv(path, selected)
    _json(identity_path, identity)
    return selected


def _stage_dir(session_dir: Path, optical_pass: str) -> Path:
    return session_dir / PASS_DIRECTORIES[optical_pass]


class CcdFolderLoader:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.cache: dict[tuple[str, str], torch.Tensor] = {}

    def __call__(self, optical_pass: str, keys: list[str], device: torch.device) -> torch.Tensor:
        directory = _stage_dir(self.session_dir, optical_pass) / "ccd_captured"
        values = []
        for key in keys:
            cache_key = (optical_pass, key)
            if cache_key not in self.cache:
                matches = [directory / f"{key}{suffix}" for suffix in (".png", ".tif", ".tiff", ".npy")]
                source = next((value for value in matches if value.is_file()), None)
                if source is None:
                    raise FileNotFoundError(f"Missing CCD for {optical_pass}/{key}")
                if source.suffix.lower() == ".npy":
                    array = np.load(source, allow_pickle=False)
                else:
                    with Image.open(source) as image:
                        array = np.asarray(image.convert("L"))
                if array.shape != (478, 478) or not np.isfinite(array).all():
                    raise RuntimeError(f"CCD must be finite canonical 478x478: {source}")
                self.cache[cache_key] = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
            values.append(self.cache[cache_key])
        return torch.stack(values).to(device)


def _batch(payload: Mapping[str, Any], indices: list[int]) -> dict[str, torch.Tensor]:
    result = {
        "vision_tokens": payload["vision_tokens"][indices].float(),
        "quality_tokens": payload["quality_tokens"][indices].float(),
        "language_tokens": payload["language_tokens"][0].float().unsqueeze(0).expand(len(indices), -1, -1),
        "language_mask": payload["language_mask"][0].bool().unsqueeze(0).expand(len(indices), -1),
    }
    for name in ("raw_frames", "vgg_tokens"):
        if name in payload:
            result[name] = payload[name][indices]
    return result


def _phase_geometry(lab_config: Path) -> tuple[tuple[float, float], bool, bool]:
    center, flip_v, flip_h = (980.0, 590.0), True, False
    if lab_config.is_file():
        raw = yaml.safe_load(lab_config.read_text(encoding="utf-8")) or {}
        phase = raw.get("phase_slm", {})
        value = phase.get("center_xy", center)
        center = (float(value[0]), float(value[1]))
        flip_v = bool(phase.get("flip_vertical_before_raster", flip_v))
        flip_h = bool(phase.get("flip_horizontal_before_raster", flip_h))
    return center, flip_v, flip_h


def _write_stage_phase(
    model: LGVQSingleMetricOEO16,
    settings: ExperimentSettings,
    destination: Path,
    optical_pass: str,
    lab_config: Path,
) -> Path:
    plane = build_stage_planes(model, settings)[optical_pass].phase_rad
    encoded = encode_active_phase(plane)
    center, flip_v, flip_h = _phase_geometry(lab_config)
    if flip_v:
        encoded = np.flip(encoded, 0)
    if flip_h:
        encoded = np.flip(encoded, 1)
    compact = destination / "compact_phase"
    physical = destination / "phase_to_play"
    if compact.exists():
        shutil.rmtree(compact)
    if physical.exists():
        shutil.rmtree(physical)
    save_active_png(np.ascontiguousarray(encoded), compact / f"{optical_pass}.png")
    reconstruct_directory(
        compact,
        physical,
        slm_size_wh=(1920, 1200),
        scale_factor=None,
        center_xy=center,
        logical_pixel_pitch_um=17.0,
        slm_pixel_pitch_um=8.0,
    )
    return physical / f"{optical_pass}.bmp"


def validate_capture(
    session_dir: Path, optical_pass: str, rows: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    destination = _stage_dir(session_dir, optical_pass)
    expected = {str(row["key"]) for row in rows}
    amplitudes = {path.stem for path in (destination / "amplitude_to_play").glob("*.bmp")}
    captures = {
        path.stem
        for path in (destination / "ccd_captured").glob("*")
        if path.suffix.lower() in {".png", ".tif", ".tiff", ".npy"}
    }
    phase = list((destination / "phase_to_play").glob("*.bmp"))
    if amplitudes != expected or captures != expected or len(phase) != 1:
        raise RuntimeError(
            f"{optical_pass} incomplete: amplitude={len(amplitudes)}, "
            f"ccd={len(captures)}, expected={len(expected)}, phase={len(phase)}"
        )
    for source in (destination / "ccd_captured").iterdir():
        if source.stem not in expected:
            continue
        if source.suffix.lower() == ".npy":
            array = np.load(source, allow_pickle=False)
            shape = tuple(array.shape)
        else:
            with Image.open(source) as image:
                shape = (image.height, image.width)
        if shape != (478, 478):
            raise RuntimeError(f"CCD is not canonical 478x478: {source}")
    report = {
        "schema_version": 1,
        "optical_pass": optical_pass,
        "sample_count": len(expected),
        "phase_bmp": phase[0].name,
        "phase_bmp_sha256": _sha256(phase[0]),
        "orientation": "canonical_model_xy after detector homography",
        "network_ccd_normalization": "nonnegative -> per-frame mean -> relative clip -> log1p",
    }
    _json(destination / "capture_validation_report.json", report)
    return report


@torch.no_grad()
def export_pass(
    settings: ExperimentSettings,
    checkpoint: Path,
    session_dir: Path,
    optical_pass: str,
    *,
    max_train: int | None,
    max_test: int | None,
    batch_size: int,
    device: torch.device,
    lab_config: Path,
) -> dict[str, Any]:
    if optical_pass not in OPTICAL_PASSES:
        raise ValueError(optical_pass)
    model, saved = _load_model(settings, checkpoint, device)
    model.eval()
    payload = load_single_metric_cache(settings)
    rows = _session_rows(
        session_dir, payload, settings, max_train=max_train, max_test=max_test
    )
    pass_index = OPTICAL_PASSES.index(optical_pass)
    measured_prefix = OPTICAL_PASSES[:pass_index]
    ccd = CcdFolderLoader(session_dir)
    for name in measured_prefix:
        validate_capture(session_dir, name, rows)
    destination = _stage_dir(session_dir, optical_pass)
    compact = destination / "compact_amplitude"
    physical = destination / "amplitude_to_play"
    if compact.exists():
        shutil.rmtree(compact)
    if physical.exists():
        shutil.rmtree(physical)
    phase_path = _write_stage_phase(model, settings, destination, optical_pass, lab_config)
    amplitude_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        indices = [int(row["cache_index"]) for row in subset]
        keys = [str(row["key"]) for row in subset]
        result = forward_hardware(
            model,
            _batch(payload, indices),
            keys,
            measured_passes=measured_prefix,
            measurement_loader=ccd if measured_prefix else None,
            stop_before=optical_pass,
        )
        for row, value in zip(subset, result.amplitudes[optical_pass]):
            encoded, metadata = encode_active_amplitude_with_metadata(value.detach().cpu().numpy())
            path = compact / f"{row['key']}.png"
            save_active_png(encoded, path)
            amplitude_rows.append(
                {
                    "order": row["order"],
                    "key": row["key"],
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "amplitude_file": path.name,
                    "amplitude_sha256": _sha256(path),
                    "encoding_percentile": metadata["percentile"],
                    "encoding_scale": metadata["scale"],
                }
            )
        print(f"[export {optical_pass}] {min(start + len(subset), len(rows))}/{len(rows)}", flush=True)
    _write_csv(destination / "amplitude_manifest.csv", amplitude_rows)
    reconstruction = reconstruct_directory(
        compact,
        physical,
        slm_size_wh=(1024, 1024),
        scale_factor=1,
        center_xy=(512.0, 512.0),
    )
    report = {
        "schema_version": 1,
        "optical_pass": optical_pass,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint.resolve()),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "sample_count": len(rows),
        "measured_upstream_passes": list(measured_prefix),
        "phase_bmp": str(phase_path),
        "phase_bmp_sha256": _sha256(phase_path),
        "amplitude_reconstruction": reconstruction,
    }
    _json(destination / "export_report.json", report)
    return report


class SessionDataset(Dataset[dict[str, Any]]):
    def __init__(self, payload: Mapping[str, Any], rows: list[dict[str, Any]], split: str) -> None:
        self.payload = payload
        self.rows = [row for row in rows if row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = int(row["cache_index"])
        item = {name: value[source] for name, value in self.payload.items() if name in {"vision_tokens", "quality_tokens", "raw_frames", "vgg_tokens"}}
        item.update(
            {
                "language_tokens": self.payload["language_tokens"][0],
                "language_mask": self.payload["language_mask"][0],
                "target": self.payload["targets"][source],
                "key": row["key"],
                "sample_id": row["sample_id"],
            }
        )
        return item


def _set_trainable(model: LGVQSingleMetricOEO16, stage: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "vision_expert":
        modules: list[Any] = [
            model.vision_routes[0], model.parallel_optics.expert_readout, model.fusions[0],
            model.vision_routes[1], model.parallel_optics.width_to_field,
            model.parallel_optics.tokens_to_field, model.parallel_optics.raw_global_phase,
            model.parallel_optics.global_readout, model.fusions[1], model.frame_merger,
            model.language_routes, model.serial_optics, model.serial_router, model.fusions[2:],
            model.readout,
        ]
    elif stage == "vision_global":
        modules = [
            model.vision_routes[1], model.parallel_optics.global_readout, model.fusions[1],
            model.frame_merger, model.language_routes, model.serial_optics, model.serial_router,
            model.fusions[2:], model.readout,
        ]
    elif stage == "language_expert":
        modules = [
            model.language_routes[0], model.serial_optics.expert_readout, model.fusions[2],
            model.language_routes[1], model.serial_optics.raw_global_phase,
            model.serial_optics.global_readout, model.fusions[3], model.readout,
        ]
    elif stage == "language_global":
        modules = [model.language_routes[1], model.serial_optics.global_readout, model.fusions[3], model.readout]
    else:
        raise ValueError(stage)
    for module in modules:
        if isinstance(module, torch.nn.Parameter):
            module.requires_grad_(True)
        else:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _optimizer(model: LGVQSingleMetricOEO16, settings: ExperimentSettings) -> torch.optim.Optimizer:
    electronic, phase, router = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "raw_router_phase" in name:
            router.append(parameter)
        elif "raw_" in name and "phase" in name:
            phase.append(parameter)
        else:
            electronic.append(parameter)
    groups = []
    if electronic:
        groups.append({"params": electronic, "lr": settings.learning_rate, "weight_decay": settings.weight_decay})
    if phase:
        groups.append({"params": phase, "lr": settings.phase_learning_rate, "weight_decay": 0.0})
    if router:
        groups.append({"params": router, "lr": settings.router_phase_learning_rate, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


@torch.no_grad()
def _evaluate(
    model: LGVQSingleMetricOEO16,
    loader: DataLoader,
    ccd: CcdFolderLoader,
    measured: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions, targets, rows = [], [], []
    for batch in loader:
        result = forward_hardware(model, batch, batch["key"], measured_passes=measured, measurement_loader=ccd)
        prediction = result.prediction.detach().cpu()
        target = batch["target"].float().cpu()
        predictions.append(prediction)
        targets.append(target)
        for key, sample_id, observed, expected in zip(batch["key"], batch["sample_id"], prediction, target):
            rows.append({"key": key, "sample_id": sample_id, "target": float(expected), "prediction": float(observed)})
    metrics = regression_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["measured_passes"] = list(measured)
    return metrics, rows


def finetune(
    settings: ExperimentSettings,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    *,
    epochs: int,
    batch_size: int,
    test_interval: int,
    device: torch.device,
) -> dict[str, Any]:
    model, _ = _load_model(settings, checkpoint, device)
    payload = load_single_metric_cache(settings)
    identity = json.loads((session_dir / "session_identity.json").read_text(encoding="utf-8"))
    selection = identity["selection"]
    rows = _session_rows(
        session_dir,
        payload,
        settings,
        max_train=selection["max_train"],
        max_test=selection["max_test"],
    )
    measured = tuple(FUSION_STAGES[stage])
    for name in measured:
        validate_capture(session_dir, name, rows)
    ccd = CcdFolderLoader(session_dir)
    train_loader = DataLoader(SessionDataset(payload, rows, "train"), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(SessionDataset(payload, rows, "test"), batch_size=batch_size, shuffle=False, num_workers=0)
    trainable = _set_trainable(model, stage)
    optimizer = _optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score, best_epoch = float("-inf"), 0
    output_dir = session_dir / "checkpoints"
    best_path = output_dir / f"after_{stage}_best_test.pt"
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        count = 0
        for batch in train_loader:
            target = batch["target"].to(device).float()
            normalized_target = (target - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            result = forward_hardware(model, batch, batch["key"], measured_passes=measured, measurement_loader=ccd)
            prediction = result.normalized_prediction
            regression = F.smooth_l1_loss(prediction, normalized_target)
            ranking = pairwise_ranking_loss(prediction, normalized_target)
            correlation = batch_correlation_loss(prediction, normalized_target)
            loss = regression + settings.ranking_weight * ranking + settings.correlation_weight * correlation
            if result.optical_alignment_loss is not None:
                loss = loss + settings.optical_alignment_weight * result.optical_alignment_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            count += 1
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "loss": totals["loss"] / max(1, count), "test_evaluated": False}
        if epoch == 1 or epoch % test_interval == 0 or epoch == epochs:
            metrics, prediction_rows = _evaluate(model, test_loader, ccd, measured, device)
            score = float(metrics["srcc"])
            row.update({"test_evaluated": True, "test": metrics})
            if score > best_score:
                best_score, best_epoch = score, epoch
                output_dir.mkdir(parents=True, exist_ok=True)
                temporary = best_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "schema_version": 1,
                        "architecture": settings.architecture_label,
                        "target_name": settings.target_name,
                        "prompt": settings.prompt,
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "metrics_optical_hardware": metrics,
                        "settings": resolved_dict(settings),
                        "hardware_finetune": {
                            "stage": stage,
                            "measured_passes": list(measured),
                            "source_checkpoint_sha256": _sha256(checkpoint.resolve()),
                            "trainable_parameter_names": trainable,
                            "test_used_for_selection": True,
                        },
                    },
                    temporary,
                )
                temporary.replace(best_path)
                _json(output_dir / f"after_{stage}_best_metrics.json", metrics)
                _write_csv(output_dir / f"after_{stage}_best_predictions.csv", prediction_rows)
        history.append(row)
        _json(output_dir / f"after_{stage}_history.json", history)
        print(f"[finetune {stage}] epoch={epoch}/{epochs} loss={row['loss']:.6f} best_test_SRCC={best_score:.4f}", flush=True)
    report = {
        "stage": stage,
        "best_epoch": best_epoch,
        "best_test_srcc": best_score,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "test_used_for_selection": True,
    }
    _json(output_dir / f"after_{stage}_report.json", report)
    return report


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export-pass", "validate-capture", "finetune", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--optical-pass", choices=OPTICAL_PASSES)
    parser.add_argument("--stage", choices=tuple(FUSION_STAGES))
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lab-config", default="experiments/lab_lgvq/LAB_CONFIG.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    session = Path(args.session_dir).expanduser().resolve()
    device = _device(args.device)
    if args.action == "export-pass":
        if args.optical_pass is None:
            parser.error("export-pass requires --optical-pass")
        report = export_pass(
            settings,
            checkpoint,
            session,
            args.optical_pass,
            max_train=None if args.all_data else args.max_train,
            max_test=None if args.all_data else args.max_test,
            batch_size=args.batch_size,
            device=device,
            lab_config=Path(args.lab_config),
        )
    else:
        payload = load_single_metric_cache(settings)
        identity_path = session / "session_identity.json"
        if not identity_path.is_file():
            raise FileNotFoundError("Export the first pass before validation/fine-tuning")
        selection = json.loads(identity_path.read_text(encoding="utf-8"))["selection"]
        rows = _session_rows(session, payload, settings, max_train=selection["max_train"], max_test=selection["max_test"])
        if args.action == "validate-capture":
            if args.optical_pass is None:
                parser.error("validate-capture requires --optical-pass")
            report = validate_capture(session, args.optical_pass, rows)
        elif args.action == "finetune":
            if args.stage is None:
                parser.error("finetune requires --stage")
            report = finetune(settings, checkpoint, session, args.stage, epochs=args.epochs, batch_size=args.batch_size, test_interval=args.test_interval, device=device)
        else:
            if args.stage is None:
                parser.error("evaluate requires --stage")
            model, _ = _load_model(settings, checkpoint, device)
            measured = tuple(FUSION_STAGES[args.stage])
            for name in measured:
                validate_capture(session, name, rows)
            loader = DataLoader(SessionDataset(payload, rows, "test"), batch_size=args.batch_size, shuffle=False, num_workers=0)
            metrics, predictions = _evaluate(model, loader, CcdFolderLoader(session), measured, device)
            output = session / "final_evaluation"
            _json(output / f"through_{args.stage}_metrics.json", metrics)
            _write_csv(output / f"through_{args.stage}_predictions.csv", predictions)
            report = metrics
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
