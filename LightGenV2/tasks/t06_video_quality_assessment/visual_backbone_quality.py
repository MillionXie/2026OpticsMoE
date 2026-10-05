"""Frozen CLIP/YOLO11 visual baselines for LGVQ temporal and spatial quality.

Both backbones expose one 512-dimensional vector for each of the four ordered
video frames.  The vectors are L2-normalized and concatenated in temporal order,
yielding the same 2048-dimensional readout width used by the Qwen3-VL-2B
quality-token baseline.  Every pretrained parameter is frozen; the only
trainable parameters are five bias-free output rows (5 * 2048 = 10,240).

CLIP rows are initialized from its native text embeddings for Bad, Poor, Fair,
Good, and Excellent.  YOLO11 has no text tower, so its rows use deterministic
Xavier-uniform initialization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import quality_token_common as core
from .project import REPO_ROOT, TASK_DIR, sha256


TARGETS = ("temporal", "spatial")
BACKBONES = ("clip_vit_b32", "yolo11s")
FRAME_COUNT = 4
FRAME_WIDTH = 512
FEATURE_WIDTH = FRAME_COUNT * FRAME_WIDTH
TRAINABLE_PARAMETERS = 5 * FEATURE_WIDTH
FEATURE_FILENAME = "ordered_four_frame_features.pt"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _sha256_ids(rows: Iterable[dict[str, Any]]) -> str:
    joined = "\n".join(row["sample_id"] for row in rows).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def read_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported visual-backbone LGVQ config: {path}")
    return raw


def validate_config(raw: dict[str, Any]) -> None:
    backbone = raw["model"]["backbone"]
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    if raw["model"]["backbone_frozen"] is not True:
        raise ValueError("The complete pretrained backbone must remain frozen")
    if raw["model"]["trainable_module"] != "five_bias_free_quality_rows":
        raise ValueError("Only the five quality rows may be trainable")
    if int(raw["model"]["frame_feature_width"]) != FRAME_WIDTH:
        raise ValueError(f"Each frame must expose {FRAME_WIDTH} features")
    if int(raw["model"]["readout_width"]) != FEATURE_WIDTH:
        raise ValueError(f"Readout width must be {FEATURE_WIDTH}")
    if int(raw["model"]["trainable_parameters"]) != TRAINABLE_PARAMETERS:
        raise ValueError(f"Trainable parameter count must be {TRAINABLE_PARAMETERS}")
    if int(raw["input"]["frame_count"]) != FRAME_COUNT:
        raise ValueError("The paired LGVQ comparison requires four frames")
    fractions = [float(value) for value in raw["input"]["frame_fractions"]]
    if fractions != core.FRAME_FRACTIONS[FRAME_COUNT]:
        raise ValueError("Frame fractions differ from the Qwen four-frame contract")
    if tuple(raw["task"]["targets"]) != TARGETS:
        raise ValueError(f"Expected both LGVQ targets in order: {TARGETS}")
    if int(raw["training"]["epochs"]) != 50:
        raise ValueError("Matched Qwen comparison requires 50 epochs")


def decode_ordered_frames(
    path: Path, fractions: list[float], image_size: int
) -> tuple[list[Image.Image], list[int]]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    positions = [
        min(total - 1, max(0, round((total - 1) * fraction)))
        for fraction in fractions
    ]
    frames: list[Image.Image] = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {position} from {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        side = max(2, round(min(height, width) * 0.65))
        top = (height - side) // 2
        left = (width - side) // 2
        resized = cv2.resize(
            rgb[top : top + side, left : left + side],
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        frames.append(Image.fromarray(resized))
    capture.release()
    return frames, positions


class FiveQualityRows(nn.Module):
    """The only trainable module in either visual-backbone baseline."""

    def __init__(self, rows: torch.Tensor) -> None:
        super().__init__()
        if tuple(rows.shape) != (5, FEATURE_WIDTH):
            raise ValueError(
                f"Expected [5,{FEATURE_WIDTH}] rows, got {tuple(rows.shape)}"
            )
        self.weight = nn.Parameter(rows.float().clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.weight.t()


def deterministic_xavier_rows(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.empty(5, FEATURE_WIDTH)
    fan_in, fan_out = FEATURE_WIDTH, 5
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
    return rows.uniform_(-bound, bound, generator=generator)


def _frozen_metadata(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"Frozen backbone exposes {trainable} trainable parameters")
    return {
        "backbone_trainable_parameters": trainable,
        "backbone_total_parameters": sum(p.numel() for p in model.parameters()),
    }


def load_clip(
    model_path: Path, device: torch.device
) -> tuple[nn.Module, Callable[[list[Image.Image]], torch.Tensor], torch.Tensor, dict[str, Any]]:
    try:
        import clip
    except ImportError as error:
        raise RuntimeError("Install the official openai/CLIP package") from error

    model, preprocess = clip.load(str(model_path), device=device, jit=False)
    model = model.eval().requires_grad_(False)
    tokens = clip.tokenize(list(core.QUALITY_WORDS)).to(device)
    with torch.inference_mode():
        text = model.encode_text(tokens).float()
        text = nn.functional.normalize(text, dim=-1)
    # Repeating text/4 means the initial logit is the average of the four native
    # CLIP image-text similarities while retaining order-specific trainable rows.
    initial_rows = torch.cat([text.cpu() / FRAME_COUNT] * FRAME_COUNT, dim=-1)

    @torch.inference_mode()
    def extract(frames: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([preprocess(frame) for frame in frames]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            features = model.encode_image(batch).float()
        features = nn.functional.normalize(features, dim=-1)
        if tuple(features.shape) != (FRAME_COUNT, FRAME_WIDTH):
            raise RuntimeError(f"Unexpected CLIP feature shape: {tuple(features.shape)}")
        return features.flatten().cpu().half().contiguous()

    metadata: dict[str, Any] = {
        **_frozen_metadata(model),
        "implementation": "openai/CLIP",
        "implementation_path": str(Path(clip.__file__).resolve()),
        "checkpoint": str(model_path),
        "per_frame_feature": "normalized_final_image_embedding",
        "head_initialization": "four_repeated_native_clip_quality_text_embeddings",
        "quality_words_bad_to_excellent": list(core.QUALITY_WORDS),
        "initial_rows_sha256": _sha256_tensor(initial_rows),
    }
    return model, extract, initial_rows, metadata


def load_yolo(
    model_path: Path, device: torch.device, seed: int
) -> tuple[nn.Module, Callable[[list[Image.Image]], torch.Tensor], torch.Tensor, dict[str, Any]]:
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install a pinned Ultralytics release with YOLO11 support") from error

    wrapper = YOLO(str(model_path))
    model = wrapper.model.to(device).eval().requires_grad_(False)
    layers = getattr(model, "model", None)
    if layers is None or len(layers) <= 10:
        raise RuntimeError("Unexpected Ultralytics YOLO11 model structure")
    feature_layer = layers[10]
    if feature_layer.__class__.__name__ != "C2PSA":
        raise RuntimeError(
            f"Expected layer 10 to be C2PSA, got {feature_layer.__class__.__name__}"
        )
    capture: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        capture["feature"] = output

    handle = feature_layer.register_forward_hook(hook)

    @torch.inference_mode()
    def extract(frames: list[Image.Image]) -> torch.Tensor:
        arrays = [np.asarray(frame, dtype=np.uint8).copy() for frame in frames]
        batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
        batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
        capture.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            model(batch)
        feature_map = capture.get("feature")
        if feature_map is None:
            raise RuntimeError("YOLO11 C2PSA feature hook did not run")
        features = feature_map.float().mean(dim=(-2, -1))
        features = nn.functional.normalize(features, dim=-1)
        if tuple(features.shape) != (FRAME_COUNT, FRAME_WIDTH):
            raise RuntimeError(f"Unexpected YOLO11 feature shape: {tuple(features.shape)}")
        return features.flatten().cpu().half().contiguous()

    initial_rows = deterministic_xavier_rows(seed)
    metadata = {
        **_frozen_metadata(model),
        "implementation": "ultralytics/ultralytics",
        "implementation_version": ultralytics.__version__,
        "checkpoint": str(model_path),
        "per_frame_feature": "normalized_global_average_c2psa_layer10",
        "feature_layer_index": 10,
        "feature_layer_type": feature_layer.__class__.__name__,
        "head_initialization": f"xavier_uniform_seed_{seed}",
        "quality_words_bad_to_excellent": list(core.QUALITY_WORDS),
        "initial_rows_sha256": _sha256_tensor(initial_rows),
        "hook_handle": handle,
    }
    return model, extract, initial_rows, metadata


def load_backbone(
    raw: dict[str, Any], model_path: Path, device: torch.device
) -> tuple[nn.Module, Callable[[list[Image.Image]], torch.Tensor], torch.Tensor, dict[str, Any]]:
    backbone = raw["model"]["backbone"]
    if backbone == "clip_vit_b32":
        return load_clip(model_path, device)
    if backbone == "yolo11s":
        return load_yolo(model_path, device, int(raw["training"]["seed"]))
    raise AssertionError(backbone)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "hook_handle"}


def _feature_identity(
    raw: dict[str, Any], model_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "feature_contract": (
            f"frozen_{raw['model']['backbone']}_four_ordered_normalized_"
            f"frame_features_{FEATURE_WIDTH}_v1"
        ),
        "backbone": raw["model"]["backbone"],
        "model_path": str(model_path),
        "frame_count": FRAME_COUNT,
        "frame_fractions": core.FRAME_FRACTIONS[FRAME_COUNT],
        "image_size_wh": [int(raw["input"]["image_size"])] * 2,
        "center_crop_short_side_fraction": 0.65,
        "sample_count": len(rows),
        "sample_order_sha256": _sha256_ids(rows),
        "backbone_trainable_parameters": 0,
    }


def extract_one(
    row: dict[str, Any], raw: dict[str, Any], extractor: Callable[[list[Image.Image]], torch.Tensor]
) -> tuple[torch.Tensor, list[int]]:
    frames, positions = decode_ordered_frames(
        Path(row["video_path"]),
        core.FRAME_FRACTIONS[FRAME_COUNT],
        int(raw["input"]["image_size"]),
    )
    feature = extractor(frames)
    if tuple(feature.shape) != (FEATURE_WIDTH,):
        raise RuntimeError(f"Unexpected ordered feature shape: {tuple(feature.shape)}")
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("Non-finite visual backbone feature")
    return feature, positions


def extract_all(
    *,
    raw: dict[str, Any],
    rows: list[dict[str, Any]],
    model_path: Path,
    output: Path,
    extractor: Callable[[list[Image.Image]], torch.Tensor],
    initial_rows: torch.Tensor,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    identity = _feature_identity(raw, model_path, rows)
    shard_root = output.with_suffix(".parts")
    shard_root.mkdir(parents=True, exist_ok=True)
    chunk_rows = int(raw["feature_extraction"]["chunk_rows"])
    all_features = torch.empty(len(rows), FEATURE_WIDTH, dtype=torch.float16)
    started = time.perf_counter()
    for chunk_start in range(0, len(rows), chunk_rows):
        chunk_stop = min(len(rows), chunk_start + chunk_rows)
        shard = shard_root / f"rows_{chunk_start:05d}_{chunk_stop:05d}.pt"
        expected_ids = [row["sample_id"] for row in rows[chunk_start:chunk_stop]]
        if shard.is_file():
            candidate = torch.load(shard, map_location="cpu", weights_only=False)
            if (
                candidate.get("identity") == identity
                and candidate.get("sample_ids") == expected_ids
                and tuple(candidate.get("features", ()).shape)
                == (chunk_stop - chunk_start, FEATURE_WIDTH)
            ):
                all_features[chunk_start:chunk_stop].copy_(candidate["features"])
                print(f"[resume] {chunk_stop}/{len(rows)}", flush=True)
                continue
        features: list[torch.Tensor] = []
        for index in range(chunk_start, chunk_stop):
            feature, _positions = extract_one(rows[index], raw, extractor)
            features.append(feature)
            if index == chunk_start or (index + 1) % 25 == 0:
                print(f"[extract] {index + 1}/{len(rows)}", flush=True)
        chunk_features = torch.stack(features)
        _atomic_save(
            shard,
            {"identity": identity, "sample_ids": expected_ids, "features": chunk_features},
        )
        all_features[chunk_start:chunk_stop].copy_(chunk_features)
        print(f"[saved] {chunk_stop}/{len(rows)}", flush=True)
    clean_metadata = _clean_metadata(model_metadata)
    payload = {
        "identity": identity,
        "model_metadata": clean_metadata,
        "initial_quality_rows": initial_rows.float().cpu(),
        "sample_ids": [row["sample_id"] for row in rows],
        "splits": [row["split"] for row in rows],
        "temporal_targets": torch.tensor([row["temporal"] for row in rows]),
        "spatial_targets": torch.tensor([row["spatial"] for row in rows]),
        "features": all_features,
    }
    _atomic_save(output, payload)
    report = {
        **identity,
        **clean_metadata,
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_shape": list(all_features.shape),
        "feature_dtype": str(all_features.dtype),
        "feature_sha256": _sha256_tensor(all_features),
    }
    _write_json(output.with_suffix(".report.json"), report)
    return report


@torch.inference_mode()
def evaluate(
    head: FiveQualityRows,
    features: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    level_scores: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    logits = head(features.to(device)).float()
    probabilities = logits.softmax(-1)
    predictions = probabilities @ level_scores.to(device)
    result = core.metrics(targets.numpy(), predictions.cpu().numpy())
    result["five_level_accuracy"] = float(logits.argmax(-1).cpu().eq(labels).float().mean())
    result["cross_entropy"] = float(nn.functional.cross_entropy(logits.cpu(), labels))
    return result, predictions.cpu(), probabilities.cpu()


def train_target(
    raw: dict[str, Any], target: str, feature_path: Path, output_dir: Path
) -> dict[str, Any]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    if identity["backbone_trainable_parameters"] != 0:
        raise RuntimeError("Feature cache violates the frozen-backbone contract")
    features = payload["features"].float().contiguous()
    targets = payload[f"{target}_targets"].float().contiguous()
    if tuple(features.shape) != (2808, FEATURE_WIDTH):
        raise RuntimeError(f"Unexpected feature cache shape: {tuple(features.shape)}")
    train_indices = torch.tensor(
        [index for index, split in enumerate(payload["splits"]) if split == "train"]
    )
    test_indices = torch.tensor(
        [index for index, split in enumerate(payload["splits"]) if split == "test"]
    )
    if (len(train_indices), len(test_indices)) != (2250, 558):
        raise RuntimeError("Expected fixed LGVQ split 2250/558")
    train_targets = targets[train_indices]
    train_min, train_max = float(train_targets.min()), float(train_targets.max())
    boundaries = torch.linspace(train_min, train_max, 6)[1:-1]
    level_scores = torch.linspace(train_min, train_max, 5)
    labels = torch.bucketize(targets, boundaries, right=False)
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    seed = int(raw["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initial_rows = payload["initial_quality_rows"].float()
    head = FiveQualityRows(initial_rows).to(device)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    if trainable != TRAINABLE_PARAMETERS:
        raise RuntimeError(f"Expected {TRAINABLE_PARAMETERS} trainable parameters, got {trainable}")
    epochs = int(raw["training"]["epochs"])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(raw["training"]["learning_rate"]),
        weight_decay=float(raw["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(features[train_indices], train_labels),
        batch_size=int(raw["training"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )
    history: list[dict[str, Any]] = []
    best_srcc = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(batch_features), batch_labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            sample_count += len(batch_labels)
        scheduler.step()
        head.eval()
        test_metrics, _, _ = evaluate(
            head, features[test_indices], targets[test_indices], test_labels, level_scores, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": loss_sum / sample_count,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "test": test_metrics,
            }
        )
        if test_metrics["srcc"] > best_srcc:
            best_srcc = test_metrics["srcc"]
            best_epoch = epoch
            best_metrics = dict(test_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[train {target}] epoch={epoch:03d} test_srcc={test_metrics['srcc']:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
    if best_state is None or best_metrics is None:
        raise RuntimeError("No checkpoint selected")
    last_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    final_metrics, predictions, probabilities = evaluate(
        head, features[test_indices], targets[test_indices], test_labels, level_scores, device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": 1,
        "architecture": f"frozen_{identity['backbone']}_plus_five_ordered_frame_rows",
        "initial_rows": initial_rows,
        "quality_words_bad_to_excellent": list(core.QUALITY_WORDS),
        "train_score_min": train_min,
        "train_score_max": train_max,
        "boundaries": boundaries,
        "level_scores": level_scores,
        "feature_identity": identity,
        "all_backbone_parameters_frozen": True,
        "trainable_parameters": trainable,
    }
    best_checkpoint = {
        **common,
        "state_dict": best_state,
        "selected_epoch": best_epoch,
        "metrics": final_metrics,
        "selection_policy": f"highest test {target} SRCC; no validation",
    }
    last_metrics, _, _ = evaluate(
        FiveQualityRows(last_state["weight"]).to(device),
        features[test_indices],
        targets[test_indices],
        test_labels,
        level_scores,
        device,
    )
    _atomic_save(output_dir / "best_checkpoint.pt", best_checkpoint)
    _atomic_save(
        output_dir / "last_checkpoint.pt",
        {
            **common,
            "state_dict": last_state,
            "selected_epoch": epochs,
            "metrics": last_metrics,
            "selection_policy": "last epoch; not used for reported result",
        },
    )
    _write_json(output_dir / "train_history.json", history)
    test_ids = [payload["sample_ids"][index] for index in test_indices.tolist()]
    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["sample_id", "target_mos", "target_level", "prediction_mos"]
            + [f"probability_{word.lower()}" for word in core.QUALITY_WORDS]
        )
        for sample_id, target_value, label, prediction, sample_probabilities in zip(
            test_ids,
            targets[test_indices].tolist(),
            test_labels.tolist(),
            predictions.tolist(),
            probabilities.tolist(),
        ):
            writer.writerow(
                [sample_id, target_value, core.QUALITY_WORDS[label], prediction]
                + sample_probabilities
            )
    report = {
        "schema_version": 1,
        "status": "complete",
        "target": target,
        "best_epoch": best_epoch,
        "best_test_metrics": final_metrics,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "backbone_trainable_parameters": 0,
        "quality_output_row_trainable_parameters": trainable,
        "feature_cache": str(feature_path),
        "feature_cache_sha256": sha256(feature_path),
        "checkpoint": str(output_dir / "best_checkpoint.pt"),
        "initial_rows_sha256": _sha256_tensor(initial_rows),
        "trained_rows_sha256": _sha256_tensor(best_state["weight"]),
        "model_metadata": payload["model_metadata"],
    }
    _write_json(output_dir / "training_report.json", report)
    return report


def _resolve_run_dir(raw: dict[str, Any], override: Path | None) -> Path:
    value = override if override is not None else Path(raw["run"]["run_dir"])
    path = value.expanduser()
    path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    owner = (TASK_DIR / "runs" / "simulation").resolve()
    if not path.is_relative_to(owner):
        raise ValueError(f"Run directory must stay under {owner}, got {path}")
    return path


def _record_identity(
    config: Path, model: Path, manifest: Path, run_dir: Path, raw: dict[str, Any]
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "git_status": _git("status", "--porcelain").splitlines(),
        "config": str(config),
        "config_sha256": sha256(config),
        "model": str(model),
        "model_sha256": sha256(model),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "run_dir": str(run_dir),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "resolved_config": raw,
    }
    _write_json(run_dir / "run_identity.json", identity)
    return identity


def smoke(
    raw: dict[str, Any], rows: list[dict[str, Any]], model_path: Path, run_dir: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Visual backbone smoke test requires CUDA")
    device = torch.device("cuda:0")
    model, extractor, initial_rows, metadata = load_backbone(raw, model_path, device)
    feature, positions = extract_one(rows[0], raw, extractor)
    report = {
        "status": "smoke_complete",
        "gpu": torch.cuda.get_device_name(device),
        "model_metadata": _clean_metadata(metadata),
        "backbone_trainable_parameters": 0,
        "feature_shape": list(feature.shape),
        "feature_finite": bool(torch.isfinite(feature).all()),
        "frame_positions": positions,
        "initial_rows_shape": list(initial_rows.shape),
        "trainable_parameters": initial_rows.numel(),
    }
    _write_json(run_dir / "smoke_report.json", report)
    handle = metadata.get("hook_handle")
    if handle is not None:
        handle.remove()
    del model
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "smoke", "extract", "train", "all"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    config = args.config.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    raw = read_config(config)
    validate_config(raw)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint is missing: {model_path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"LGVQ manifest is missing: {manifest}")
    run_dir = _resolve_run_dir(raw, args.run_dir)
    rows = core.read_manifest(manifest)
    if len(rows) != 2808:
        raise RuntimeError(f"Expected 2808 LGVQ rows, got {len(rows)}")
    identity = _record_identity(config, model_path, manifest, run_dir, raw)
    if args.phase == "preflight":
        print(json.dumps({"status": "ready", **identity}, indent=2), flush=True)
        return 0
    if args.phase == "smoke":
        print(json.dumps(smoke(raw, rows, model_path, run_dir), indent=2), flush=True)
        return 0

    phases = ("extract", "train") if args.phase == "all" else (args.phase,)
    status: dict[str, Any] = {"status": "running", "completed": []}
    _write_json(run_dir / "status.json", status)
    try:
        if "extract" in phases:
            if not torch.cuda.is_available():
                raise RuntimeError("Feature extraction requires CUDA")
            device = torch.device("cuda:0")
            model, extractor, initial_rows, metadata = load_backbone(raw, model_path, device)
            report = extract_all(
                raw=raw,
                rows=rows,
                model_path=model_path,
                output=run_dir / "features" / FEATURE_FILENAME,
                extractor=extractor,
                initial_rows=initial_rows,
                model_metadata=metadata,
            )
            status["completed"].append({"phase": "extract", "report": report})
            _write_json(run_dir / "status.json", status)
            handle = metadata.get("hook_handle")
            if handle is not None:
                handle.remove()
            del model
            torch.cuda.empty_cache()
        if "train" in phases:
            for target in TARGETS:
                report = train_target(
                    raw,
                    target,
                    run_dir / "features" / FEATURE_FILENAME,
                    run_dir / "checkpoints" / target,
                )
                status["completed"].append({"phase": "train", "target": target, "report": report})
                _write_json(run_dir / "status.json", status)
        status["status"] = "complete"
    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        _write_json(run_dir / "status.json", status)
        raise
    _write_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
