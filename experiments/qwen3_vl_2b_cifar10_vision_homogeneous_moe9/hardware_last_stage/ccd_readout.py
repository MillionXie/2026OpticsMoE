from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ..datasets import CIFAR10_CLASSES
from ..io_utils import resolve_device, set_seed, write_csv, write_json
from ..metrics import metrics_from_logits
from ..modeling import build_head
from ..optics.moe import FullPlaneDetectorReadout
from ..settings import load_settings
from ..visualization import save_confusion_matrix
from .config import HardwareSettings, load_hardware_settings


class CCDReadoutModel(nn.Module):
    """Physical CCD intensity -> unchanged student electronic tail -> logits."""

    def __init__(self, source_settings: Any, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.detector_readout = FullPlaneDetectorReadout(source_settings)
        self.output_adapter = nn.Linear(source_settings.input_adapter_dim, hidden_size)
        self.head = build_head(source_settings, hidden_size, num_classes)

    def forward(self, intensity: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
        readout = self.detector_readout.forward_intensity(intensity)
        features = []
        for index, token_count in enumerate(token_counts.long().tolist()):
            if not 0 < token_count <= readout.shape[1]:
                raise ValueError(f"visual_token_count={token_count} is outside [1,{readout.shape[1]}]")
            hidden = self.output_adapter(readout[index, :token_count, :])
            features.append(hidden.float().mean(0))
        return self.head(torch.stack(features))


def build_ccd_model(settings: HardwareSettings, device: torch.device) -> tuple[CCDReadoutModel, Any]:
    source = load_settings(settings.source_config)
    surrogate_path = settings.source_run_dir / "checkpoints" / f"vision_homogeneous_moe_{settings.checkpoint_tag}.pt"
    head_path = settings.source_run_dir / "checkpoints" / f"student_head_{settings.checkpoint_tag}.pt"
    if not surrogate_path.is_file() or not head_path.is_file():
        raise FileNotFoundError("Source student surrogate/head checkpoints are required")
    surrogate_payload = torch.load(surrogate_path, map_location="cpu", weights_only=True)
    head_payload = torch.load(head_path, map_location="cpu", weights_only=True)
    output_weight = surrogate_payload["state_dict"]["output_adapter.weight"]
    hidden_size = int(output_weight.shape[0])
    model = CCDReadoutModel(source, hidden_size, len(CIFAR10_CLASSES))
    model.output_adapter.load_state_dict({
        "weight": output_weight, "bias": surrogate_payload["state_dict"]["output_adapter.bias"],
    })
    model.head.load_state_dict(head_payload["state_dict"])
    model.to(device)
    return model, source


class CCDDataset(Dataset[Any]):
    def __init__(self, rows: Sequence[dict[str, str]], manifest_path: Path, settings: HardwareSettings) -> None:
        self.rows = list(rows)
        self.manifest_path = manifest_path
        self.settings = settings
        self.background = _load_ccd_image(settings.ccd_background_image, settings) if settings.ccd_background_image else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        raw_path = Path(row["ccd_path"])
        path = raw_path if raw_path.is_absolute() else self.manifest_path.parent / raw_path
        intensity = _load_ccd_image(path, self.settings)
        if self.background is not None:
            intensity = (intensity - self.background).clamp_min(0.0)
        return intensity, int(row["true_label"]), int(row["visual_token_count"]), int(row["sample_index"]), str(path)


def _load_ccd_image(path: Path | None, settings: HardwareSettings) -> torch.Tensor:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"CCD image is missing: {path}")
    if path.suffix.lower() == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value.get("intensity", value.get("image"))
        image = torch.as_tensor(value).float().squeeze()
    elif path.suffix.lower() == ".npy":
        image = torch.from_numpy(np.load(path)).float().squeeze()
    else:
        array = np.asarray(Image.open(path))
        if array.ndim == 3:
            array = array[..., :3].astype(np.float32).mean(-1)
        scale = float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else 1.0
        image = torch.from_numpy(array.astype(np.float32) / scale)
    if image.ndim != 2:
        raise ValueError(f"CCD image must be two-dimensional, got {tuple(image.shape)} from {path}")
    if settings.ccd_crop_xywh is not None:
        x, y, width, height = settings.ccd_crop_xywh
        image = image[y:y + height, x:x + width]
    rotations = settings.ccd_rotate_quadrants % 4
    if rotations:
        image = torch.rot90(image, rotations, (-2, -1))
    if settings.ccd_flip_horizontal:
        image = torch.flip(image, (-1,))
    if settings.ccd_flip_vertical:
        image = torch.flip(image, (-2,))
    target = settings.ccd_image_size
    if tuple(image.shape) != (target, target):
        image = F.interpolate(image[None, None], size=(target, target), mode="bilinear", align_corners=False)[0, 0]
    return image.float().clamp_min(0.0)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_index", "true_label", "visual_token_count", "ccd_path"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise RuntimeError(f"CCD manifest is missing columns: {sorted(missing)}")
    empty = [row["sample_index"] for row in rows if not row.get("ccd_path")]
    if empty:
        raise RuntimeError(f"CCD manifest has empty ccd_path values, including sample {empty[0]}")
    return rows


def _collate(batch: Sequence[Any]):
    images, labels, counts, indices, paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), torch.tensor(counts), torch.tensor(indices), list(paths)


def _split_train_validation(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    explicit_train = [row for row in rows if row.get("split", "").lower() == "train"]
    explicit_validation = [row for row in rows if row.get("split", "").lower() in {"validation", "val"}]
    if explicit_train and explicit_validation:
        return explicit_train, explicit_validation
    candidates = explicit_train or [row for row in rows if row.get("split", "").lower() not in {"test"}]
    if not candidates:
        raise RuntimeError(
            "No CCD training samples. Exported test captures are for evaluation only; add rows with split=train "
            "(and preferably split=validation) before fine-tuning."
        )
    generator = torch.Generator().manual_seed(seed)
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    for label in range(len(CIFAR10_CLASSES)):
        group = [row for row in candidates if int(row["true_label"]) == label]
        order = torch.randperm(len(group), generator=generator).tolist()
        count = min(max(round(len(group) * fraction), 1), len(group) - 1) if len(group) > 1 else 0
        validation_rows.extend(group[index] for index in order[:count])
        train_rows.extend(group[index] for index in order[count:])
    return train_rows, validation_rows


@torch.inference_mode()
def evaluate(model: CCDReadoutModel, dataset: Dataset[Any], batch_size: int, device: torch.device,
             prediction_path: Path | None = None) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    model.eval()
    logits_all, labels_all, prediction_rows = [], [], []
    for intensity, labels, counts, indices, paths in loader:
        logits = model(intensity.to(device), counts.to(device)).cpu()
        logits_all.append(logits)
        labels_all.append(labels)
        predictions = logits.argmax(1)
        for index, label, prediction, values, path in zip(indices.tolist(), labels.tolist(), predictions.tolist(), logits.tolist(), paths):
            row = {
                "sample_index": index, "ccd_path": path, "true_label": label,
                "true_name": CIFAR10_CLASSES[label], "pred_label": prediction,
                "pred_name": CIFAR10_CLASSES[prediction], "correct": label == prediction,
            }
            row.update({f"logit_{name}": value for name, value in zip(CIFAR10_CLASSES, values)})
            prediction_rows.append(row)
    report = metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all), CIFAR10_CLASSES)
    if prediction_path is not None:
        write_csv(prediction_path, prediction_rows, list(prediction_rows[0]))
    return report


def finetune(settings: HardwareSettings, manifest_path: Path, device: torch.device) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    train_rows, validation_rows = _split_train_validation(rows, settings.ccd_validation_fraction, settings.seed)
    model, _source = build_ccd_model(settings, device)
    model.output_adapter.requires_grad_(settings.ccd_train_output_adapter)
    model.head.requires_grad_(settings.ccd_train_head)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings.ccd_learning_rate, weight_decay=settings.ccd_weight_decay)
    train_dataset = CCDDataset(train_rows, manifest_path, settings)
    validation_dataset = CCDDataset(validation_rows, manifest_path, settings)
    loader = DataLoader(train_dataset, batch_size=settings.ccd_batch_size, shuffle=True, collate_fn=_collate,
                        generator=torch.Generator().manual_seed(settings.seed))
    output = settings.ccd_output_dir
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = -1.0
    for epoch in range(1, settings.ccd_epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, seen, correct = 0.0, 0, 0
        for intensity, labels, counts, _indices, _paths in loader:
            intensity, labels, counts = intensity.to(device), labels.to(device), counts.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(intensity, counts)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            correct += int((logits.detach().argmax(1) == labels).sum())
        validation = evaluate(model, validation_dataset, settings.ccd_batch_size, device)
        row = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / seen,
            "train_top1_accuracy": correct / seen, "validation_top1_accuracy": validation["top1_accuracy"],
            "validation_macro_f1": validation["macro_f1"], "epoch_time_sec": time.perf_counter() - started,
        }
        history.append(row)
        write_csv(output / "metrics" / "ccd_training_history.csv", history, list(row))
        write_json(output / "metrics" / "ccd_training_latest.json", row)
        _save_checkpoint(model, output / "checkpoints" / "electronic_readout_last.pt", settings, row)
        if validation["top1_accuracy"] > best:
            best = validation["top1_accuracy"]
            _save_checkpoint(model, output / "checkpoints" / "electronic_readout_best.pt", settings, row)
            write_json(output / "metrics" / "best_validation.json", {**row, **validation})
        print(f"ccd epoch {epoch:03d}/{settings.ccd_epochs} loss={loss_sum/seen:.5f} train={correct/seen:.4f} val={validation['top1_accuracy']:.4f}", flush=True)
    report = {"best_validation_top1": best, "train_samples": len(train_rows), "validation_samples": len(validation_rows)}
    write_json(output / "metrics" / "ccd_training.json", report)
    return report


def inference(settings: HardwareSettings, manifest_path: Path, device: torch.device) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    test_rows = [row for row in rows if row.get("split", "test").lower() == "test"] or rows
    model, _source = build_ccd_model(settings, device)
    checkpoint = settings.ccd_output_dir / "checkpoints" / "electronic_readout_best.pt"
    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    else:
        print("WARNING: no CCD-finetuned checkpoint; evaluating the original simulated student electronic tail", flush=True)
    report = evaluate(
        model, CCDDataset(test_rows, manifest_path, settings), settings.ccd_batch_size, device,
        settings.ccd_output_dir / "metrics" / "ccd_predictions.csv",
    )
    report.update({
        "input": "physical CCD square-law intensity", "source_checkpoint_tag": settings.checkpoint_tag,
        "electronic_path": "AvgPool4 -> non-affine LayerNorm(120x120) -> activation -> first T rows -> Linear(120,1024) -> mean tokens -> normalized linear head",
    })
    write_json(settings.ccd_output_dir / "metrics" / "ccd_inference.json", report)
    matrix = report["confusion_matrix"]
    matrix_rows = [{"true_name": CIFAR10_CLASSES[index], **{f"pred_{name}": value for name, value in zip(CIFAR10_CLASSES, row)}} for index, row in enumerate(matrix)]
    write_csv(settings.ccd_output_dir / "metrics" / "ccd_confusion_matrix.csv", matrix_rows, list(matrix_rows[0]))
    save_confusion_matrix(matrix, CIFAR10_CLASSES, settings.ccd_output_dir / "figures" / "ccd_confusion_matrix.png")
    return report


def _save_checkpoint(model: CCDReadoutModel, path: Path, settings: HardwareSettings, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "metrics": metrics,
        "train_output_adapter": settings.ccd_train_output_adapter, "train_head": settings.ccd_train_head,
    }, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune/evaluate the Qwen vision-MoE electronic tail on physical CCD intensity frames")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("finetune", "inference", "all"), default="all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_hardware_settings(args.config)
    if args.device:
        settings.device = args.device
    if args.manifest:
        settings.ccd_manifest = args.manifest.resolve()
    if settings.ccd_manifest is None:
        raise ValueError("Set ccd_manifest in config or pass --manifest")
    device = resolve_device(settings.device)
    set_seed(settings.seed)
    if args.phase in {"finetune", "all"}:
        finetune(settings, settings.ccd_manifest, device)
    if args.phase in {"inference", "all"}:
        inference(settings, settings.ccd_manifest, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

