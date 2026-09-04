from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ("spatial", "temporal")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    environment_overrides = {
        "model_path": "LGVQ_MODEL_PATH",
        "manifest_path": "LGVQ_MANIFEST_PATH",
        "artifacts_dir": "LGVQ_ARTIFACTS_DIR",
    }
    for key, environment_name in environment_overrides.items():
        if os.environ.get(environment_name):
            raw[key] = os.environ[environment_name]
    root = path.parent
    for key in ("manifest_path", "artifacts_dir", "model_path"):
        value = Path(raw[key]).expanduser()
        raw[key] = (root / value).resolve() if not value.is_absolute() else value.resolve()
    prompts = raw.get("prompts", {})
    if tuple(prompts) != TARGETS:
        raise ValueError(f"prompts must contain exactly {TARGETS} in that order")
    expected_suffix = "Excellent, Good, Fair, Poor, or Bad."
    if not all(str(prompts[name]).endswith(expected_suffix) for name in TARGETS):
        raise ValueError("Both prompts must preserve the fixed five-level template")
    if set(raw["frame_counts"]) != {4, 16}:
        raise ValueError("Formal baseline must contain exactly the 4-frame and 16-frame runs")
    return raw


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "video_path", "split", "spatial", "temporal"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid LGVQ manifest: {path}")
    result: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] not in {"train", "test"}:
            raise RuntimeError("Baseline permits only the fixed train/test split; no validation split")
        video = Path(row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        result.append(
            {
                "sample_id": row["sample_id"],
                "video_path": str(video.resolve()),
                "split": row["split"],
                "spatial": float(row["spatial"]),
                "temporal": float(row["temporal"]),
            }
        )
    if not {row["split"] for row in result} == {"train", "test"}:
        raise RuntimeError("Both train and test rows are required")
    return result


def frame_fractions(config: dict[str, Any], frame_count: int) -> list[float]:
    configured = config["frame_sampling"][str(frame_count)]
    if configured == "uniform_midpoints":
        return [(index + 0.5) / frame_count for index in range(frame_count)]
    values = [float(value) for value in configured]
    if len(values) != frame_count or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Invalid sampling fractions for {frame_count} frames")
    return values


@dataclass
class DecodedVideo:
    frames: list[Image.Image]
    metadata: Any


def decode_video(path: Path, fractions: list[float], crop_fraction: float) -> DecodedVideo:
    from transformers.video_utils import VideoMetadata

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 24.0
    positions = [min(total - 1, max(0, round((total - 1) * value))) for value in fractions]
    # LGVQ clips contain only about 96 frames. Repeated random seeks are much
    # slower than decoding each short clip once, especially for the 16-frame
    # baseline. This changes no selected index and therefore no model input.
    requested: dict[int, list[int]] = {}
    for output_index, position in enumerate(positions):
        requested.setdefault(position, []).append(output_index)
    frames_by_output: list[Image.Image | None] = [None] * len(positions)
    for frame_index in range(max(positions) + 1):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
        if frame_index not in requested:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        side = max(2, round(min(height, width) * crop_fraction))
        top = (height - side) // 2
        left = (width - side) // 2
        square = cv2.resize(
            rgb[top : top + side, left : left + side],
            (448, 448),
            interpolation=cv2.INTER_AREA,
        )
        image = Image.fromarray(square)
        for output_index in requested[frame_index]:
            frames_by_output[output_index] = image.copy()
    capture.release()
    if any(frame is None for frame in frames_by_output):
        raise RuntimeError(f"Failed to collect every requested frame from {path}")
    frames = [frame for frame in frames_by_output if frame is not None]
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        width=448,
        height=448,
        duration=float(total) / fps,
        frames_indices=positions,
    )
    return DecodedVideo(frames=frames, metadata=metadata)


def prompt_text(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def cache_identity(config: dict[str, Any], rows: list[dict[str, Any]], frame_count: int) -> dict[str, Any]:
    sample_blob = "\n".join(row["sample_id"] for row in rows)
    return {
        "schema_version": 1,
        "feature_contract": "full_qwen3vl_native_video_last_assistant_token_2048_v1",
        "model_path": str(config["model_path"]),
        "frame_count": int(frame_count),
        "frame_fractions": frame_fractions(config, frame_count),
        "sample_count": len(rows),
        "sample_order_sha256": sha256_text(sample_blob),
        "prompts": dict(config["prompts"]),
        "prompt_sha256": sha256_text(json.dumps(config["prompts"], sort_keys=True)),
        "center_crop_short_side_fraction": float(config["center_crop_short_side_fraction"]),
        "processor_pixels": int(config["processor_pixels"]),
        "pooling": "last valid assistant-generation-prefix hidden state after final Qwen norm",
        "native_video_token": True,
        "qwen_lm_head_executed": False,
    }


def cache_path(config: dict[str, Any], frame_count: int) -> Path:
    return Path(config["artifacts_dir"]) / f"frames{frame_count}" / "qwen_prompt_features.pt"


def _valid_final_cache(path: Path, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return payload.get("identity") == identity and tuple(payload.get("features", ()).shape) == (
        identity["sample_count"],
        2,
        2048,
    )


def build_feature_cache(config: dict[str, Any], frame_count: int, *, limit: int | None = None) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    all_rows = read_manifest(Path(config["manifest_path"]))
    rows = all_rows if limit is None else all_rows[:limit]
    identity = cache_identity(config, rows, frame_count)
    output = cache_path(config, frame_count)
    if limit is not None:
        output = output.with_name(f"smoke_{limit}.pt")
    if _valid_final_cache(output, identity):
        report_path = output.with_suffix(".report.json")
        return json.loads(report_path.read_text(encoding="utf-8"))

    device = torch.device("cuda")
    pixels = int(config["processor_pixels"])
    processor = AutoProcessor.from_pretrained(
        str(config["model_path"]),
        min_pixels=pixels,
        max_pixels=pixels,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(config["model_path"]),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    qwen_total = sum(parameter.numel() for parameter in model.parameters())
    qwen_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if qwen_trainable != 0:
        raise RuntimeError("Qwen backbone must be completely frozen")

    texts = {name: prompt_text(processor, config["prompts"][name]) for name in TARGETS}
    fractions = frame_fractions(config, frame_count)
    batch_size = int(config["cache_batch_sizes"][str(frame_count)])
    chunk_rows = int(config["cache_chunk_rows"])
    shard_root = output.with_suffix(".parts")
    shard_root.mkdir(parents=True, exist_ok=True)
    features = torch.empty(len(rows), 2, 2048, dtype=torch.float16)
    total_decode = total_processor = total_forward = 0.0
    total_batches = 0
    peak_memory = 0
    wall_started = time.perf_counter()

    for chunk_start in range(0, len(rows), chunk_rows):
        chunk_stop = min(len(rows), chunk_start + chunk_rows)
        shard = shard_root / f"rows_{chunk_start:05d}_{chunk_stop:05d}.pt"
        expected_ids = [row["sample_id"] for row in rows[chunk_start:chunk_stop]]
        loaded = None
        if shard.is_file():
            candidate = torch.load(shard, map_location="cpu", weights_only=False)
            if (
                candidate.get("identity") == identity
                and candidate.get("sample_ids") == expected_ids
                and tuple(candidate.get("features", ()).shape) == (chunk_stop - chunk_start, 2, 2048)
            ):
                loaded = candidate
        if loaded is not None:
            features[chunk_start:chunk_stop].copy_(loaded["features"])
            timings = loaded["timings"]
            total_decode += float(timings["decode_seconds"])
            total_processor += float(timings["processor_seconds"])
            total_forward += float(timings["qwen_forward_seconds"])
            total_batches += int(timings["batch_count"])
            peak_memory = max(peak_memory, int(timings["peak_cuda_bytes"]))
            print(f"[cache {frame_count}f] resumed {chunk_stop}/{len(rows)}", flush=True)
            continue

        chunk_features: list[torch.Tensor] = []
        chunk_decode = chunk_processor = chunk_forward = 0.0
        chunk_batches = 0
        chunk_peak = 0
        for start in range(chunk_start, chunk_stop, batch_size):
            stop = min(chunk_stop, start + batch_size)
            batch_rows = rows[start:stop]
            tick = time.perf_counter()
            worker_count = min(len(batch_rows), int(config.get("decode_workers", 1)))
            decode_arguments = [
                (Path(row["video_path"]), fractions, float(config["center_crop_short_side_fraction"]))
                for row in batch_rows
            ]
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                decoded = list(executor.map(lambda values: decode_video(*values), decode_arguments))
            chunk_decode += time.perf_counter() - tick
            expanded_texts: list[str] = []
            expanded_videos: list[list[Image.Image]] = []
            expanded_metadata: list[Any] = []
            for item in decoded:
                for target_name in TARGETS:
                    expanded_texts.append(texts[target_name])
                    expanded_videos.append(item.frames)
                    expanded_metadata.append(item.metadata)
            tick = time.perf_counter()
            inputs = processor(
                text=expanded_texts,
                videos=expanded_videos,
                video_metadata=expanded_metadata,
                padding=True,
                return_tensors="pt",
                do_sample_frames=False,
            )
            chunk_processor += time.perf_counter() - tick
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
            torch.cuda.synchronize()
            chunk_forward += time.perf_counter() - tick
            chunk_peak = max(chunk_peak, int(torch.cuda.max_memory_allocated()))
            mask = inputs["attention_mask"].bool()
            positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
            last = positions.masked_fill(~mask, -1).amax(1)
            pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
            pooled = pooled.float().reshape(len(batch_rows), 2, 2048).cpu().half().contiguous()
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("Non-finite Qwen feature encountered")
            chunk_features.append(pooled)
            chunk_batches += 1
            del inputs, hidden, pooled
        chunk_tensor = torch.cat(chunk_features)
        timings = {
            "decode_seconds": chunk_decode,
            "processor_seconds": chunk_processor,
            "qwen_forward_seconds": chunk_forward,
            "batch_count": chunk_batches,
            "peak_cuda_bytes": chunk_peak,
        }
        atomic_torch_save(
            shard,
            {
                "identity": identity,
                "sample_ids": expected_ids,
                "features": chunk_tensor,
                "timings": timings,
            },
        )
        features[chunk_start:chunk_stop].copy_(chunk_tensor)
        total_decode += chunk_decode
        total_processor += chunk_processor
        total_forward += chunk_forward
        total_batches += chunk_batches
        peak_memory = max(peak_memory, chunk_peak)
        print(f"[cache {frame_count}f] saved {chunk_stop}/{len(rows)}", flush=True)

    splits = [row["split"] for row in rows]
    targets = torch.tensor([[row[name] for name in TARGETS] for row in rows], dtype=torch.float32)
    payload = {
        "identity": identity,
        "sample_ids": [row["sample_id"] for row in rows],
        "splits": splits,
        "targets": targets,
        "features": features,
    }
    atomic_torch_save(output, payload)
    wall_seconds = time.perf_counter() - wall_started
    report = {
        **identity,
        "output": str(output),
        "qwen_total_parameters": qwen_total,
        "qwen_trainable_parameters": qwen_trainable,
        "linear_head_not_executed_during_cache": True,
        "batch_size_videos": batch_size,
        "prompt_examples_per_video": 2,
        "batch_count": total_batches,
        "decode_seconds": total_decode,
        "processor_seconds": total_processor,
        "qwen_forward_seconds": total_forward,
        "wall_seconds_this_invocation": wall_seconds,
        "mean_qwen_forward_ms_per_prompt_video": 1000.0 * total_forward / (2 * len(rows)),
        "mean_pipeline_ms_per_unique_video_for_both_prompts": 1000.0 * (total_decode + total_processor + total_forward) / len(rows),
        "peak_cuda_bytes": peak_memory,
        "peak_cuda_gib": peak_memory / (1024**3),
        "split_counts": {name: splits.count(name) for name in ("train", "test")},
    }
    atomic_json(output.with_suffix(".report.json"), report)
    return report


class SharedScalarLinearHead(nn.Module):
    """The only trainable module permitted by the formal baseline."""

    def __init__(self, width: int = 2048) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value).squeeze(-1)


def safe_correlation(callable_: Any, target: np.ndarray, predicted: np.ndarray) -> float:
    result = callable_(target, predicted)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    difference = predicted - target
    return {
        "srcc": safe_correlation(spearmanr, target, predicted),
        "krcc": safe_correlation(kendalltau, target, predicted),
        "plcc": safe_correlation(pearsonr, target, predicted),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
    }


@torch.inference_mode()
def predict(
    head: SharedScalarLinearHead,
    features: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    samples, tasks, width = features.shape
    flat = features.float().reshape(samples * tasks, width).to(device)
    normalized = head(flat).reshape(samples, tasks).cpu()
    return normalized * target_std + target_mean


def evaluate(
    head: SharedScalarLinearHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    prediction = predict(head, features, target_mean, target_std, device)
    report = {
        name: metrics(targets[:, index].numpy(), prediction[:, index].numpy())
        for index, name in enumerate(TARGETS)
    }
    report["selection_mean_srcc"] = float(np.mean([report[name]["srcc"] for name in TARGETS]))
    return report, prediction


def write_predictions(
    path: Path,
    sample_ids: list[str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "spatial_target", "spatial_prediction", "temporal_target", "temporal_prediction"])
        for sample_id, target, prediction in zip(sample_ids, targets.tolist(), predictions.tolist()):
            writer.writerow([sample_id, target[0], prediction[0], target[1], prediction[1]])


def train_linear_head(config: dict[str, Any], frame_count: int) -> dict[str, Any]:
    source = cache_path(config, frame_count)
    if not source.is_file():
        raise FileNotFoundError(f"Feature cache is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    features = payload["features"].float()
    targets = payload["targets"].float()
    splits = payload["splits"]
    train_indices = torch.tensor([i for i, value in enumerate(splits) if value == "train"], dtype=torch.long)
    test_indices = torch.tensor([i for i, value in enumerate(splits) if value == "test"], dtype=torch.long)
    train_targets = targets[train_indices]
    target_mean = train_targets.mean(0)
    target_std = train_targets.std(0, unbiased=False).clamp_min(1.0e-6)
    train_features = features[train_indices].reshape(-1, features.shape[-1])
    train_normalized_targets = ((train_targets - target_mean) / target_std).reshape(-1)

    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    head = SharedScalarLinearHead(features.shape[-1]).to(device)
    trainable_names = [name for name, parameter in head.named_parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    if trainable_names != ["linear.weight", "linear.bias"] or trainable_parameters != 2049:
        raise RuntimeError("Baseline contract violation: exactly Linear(2048,1) must be trainable")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_normalized_targets),
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )
    output_dir = Path(config["artifacts_dir"]) / f"frames{frame_count}" / "linear_head"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        head.train()
        loss_sum = 0.0
        batches = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(batch_features)
            loss = torch.nn.functional.mse_loss(prediction, batch_targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite linear baseline loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        scheduler.step()
        torch.cuda.synchronize()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_mse": loss_sum / max(1, batches),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "test_evaluated": False,
        }
        if epoch == 1 or epoch % int(config["test_interval_epochs"]) == 0 or epoch == epochs:
            head.eval()
            test_metrics, _ = evaluate(
                head,
                features[test_indices],
                targets[test_indices],
                target_mean,
                target_std,
                device,
            )
            row["test_evaluated"] = True
            row["test"] = test_metrics
            score = float(test_metrics["selection_mean_srcc"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                best_metrics = test_metrics
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            status = row.get("test", {})
            print(
                f"[train {frame_count}f] epoch={epoch:03d} mse={row['train_mse']:.6f} "
                f"spatial_srcc={status.get('spatial', {}).get('srcc', float('nan')):.4f} "
                f"temporal_srcc={status.get('temporal', {}).get('srcc', float('nan')):.4f} "
                f"best={best_epoch}",
                flush=True,
            )
    if best_state is None or best_metrics is None:
        raise RuntimeError("No baseline checkpoint was selected")
    training_seconds = time.perf_counter() - total_started
    head.load_state_dict(best_state)
    head.eval()
    final_metrics, final_prediction = evaluate(
        head,
        features[test_indices],
        targets[test_indices],
        target_mean,
        target_std,
        device,
    )

    flat_test = features[test_indices].reshape(-1, features.shape[-1]).to(device)
    for _ in range(20):
        _ = head(flat_test)
    torch.cuda.synchronize()
    repetitions = 200
    tick = time.perf_counter()
    for _ in range(repetitions):
        _ = head(flat_test)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - tick
    head_latency_ms_per_prompt_video = 1000.0 * elapsed / (repetitions * flat_test.shape[0])

    checkpoint = {
        "schema_version": 1,
        "architecture": "frozen_qwen3vl_2b_instruct_plus_one_shared_linear_2048_to_1",
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "target_mean": target_mean,
        "target_std": target_std,
        "metrics": final_metrics,
        "feature_identity": payload["identity"],
        "selection_policy": "highest test mean(spatial SRCC, temporal SRCC); no validation; test leakage accepted",
    }
    atomic_torch_save(output_dir / "best_test_checkpoint.pt", checkpoint)
    atomic_json(output_dir / "metrics_best_test.json", final_metrics)
    atomic_json(output_dir / "train_history.json", history)
    write_predictions(
        output_dir / "test_predictions.csv",
        [payload["sample_ids"][index] for index in test_indices.tolist()],
        targets[test_indices],
        final_prediction,
    )
    report = {
        "frame_count": frame_count,
        "epochs_requested": epochs,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "metrics": final_metrics,
        "training_seconds": training_seconds,
        "mean_epoch_seconds": training_seconds / epochs,
        "head_latency_ms_per_prompt_video": head_latency_ms_per_prompt_video,
        "qwen_backbone_trainable_parameters": 0,
        "head_type": "one shared nn.Linear(2048,1)",
        "head_trainable_parameters": trainable_parameters,
        "trainable_parameter_names": trainable_names,
        "loss": "mean squared error on per-target standardized MOS",
        "test_used_for_selection": True,
        "validation_used": False,
    }
    atomic_json(output_dir / "training_report.json", report)
    return report


def combined_report(config: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for frames in config["frame_counts"]:
        frame_root = Path(config["artifacts_dir"]) / f"frames{frames}"
        results[str(frames)] = {
            "cache": json.loads((frame_root / "qwen_prompt_features.report.json").read_text(encoding="utf-8")),
            "training": json.loads((frame_root / "linear_head" / "training_report.json").read_text(encoding="utf-8")),
        }
    report = {
        "schema_version": 1,
        "experiment": "Qwen3-VL-2B-Instruct frozen baseline with one shared scalar linear head",
        "results": results,
    }
    root = Path(config["artifacts_dir"])
    atomic_json(root / "comparison.json", report)
    lines = [
        "# Qwen3-VL LGVQ baseline results",
        "",
        "Only `Linear(2048,1)` is trainable. Qwen3-VL is frozen. Spatial and temporal are selected by prompt and reported separately.",
        "",
        "| Frames | Target | SRCC | KRCC | PLCC | RMSE | MAE |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for frames in config["frame_counts"]:
        values = results[str(frames)]["training"]["metrics"]
        for target in TARGETS:
            metric = values[target]
            lines.append(
                f"| {frames} | {target} | {metric['srcc']:.4f} | {metric['krcc']:.4f} | "
                f"{metric['plcc']:.4f} | {metric['rmse']:.4f} | {metric['mae']:.4f} |"
            )
    lines.extend(
        [
            "",
            "| Frames | Qwen ms / prompt-video | Pipeline ms / video for two prompts | Peak GPU GiB | Head train seconds | Head ms / prompt-video |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for frames in config["frame_counts"]:
        cache = results[str(frames)]["cache"]
        training = results[str(frames)]["training"]
        lines.append(
            f"| {frames} | {cache['mean_qwen_forward_ms_per_prompt_video']:.3f} | "
            f"{cache['mean_pipeline_ms_per_unique_video_for_both_prompts']:.3f} | "
            f"{cache['peak_cuda_gib']:.3f} | {training['training_seconds']:.3f} | "
            f"{training['head_latency_ms_per_prompt_video']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Checkpoint selection uses the periodically observed fixed test split, with no validation split, as explicitly requested for this project.",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    rows = read_manifest(Path(config["manifest_path"]))
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(str(config["model_path"]), local_files_only=True, trust_remote_code=True)
    text_config = getattr(model_config, "text_config", model_config)
    hidden_size = int(getattr(text_config, "hidden_size"))
    if hidden_size != 2048:
        raise RuntimeError(f"Expected Qwen hidden size 2048, got {hidden_size}")
    audit = {
        "status": "ready",
        "project_root": str(project_root()),
        "model_path": str(config["model_path"]),
        "manifest_path": str(config["manifest_path"]),
        "sample_count": len(rows),
        "split_counts": {name: sum(row["split"] == name for row in rows) for name in ("train", "test")},
        "targets": list(TARGETS),
        "alignment_target_used": False,
        "prompts": config["prompts"],
        "frame_counts": config["frame_counts"],
        "qwen_hidden_size": hidden_size,
        "qwen_frozen": True,
        "only_trainable_module": "nn.Linear(2048,1)",
        "only_trainable_parameter_count": 2049,
        "native_qwen_video_tokens": True,
        "qwen_vision_tower_changed": False,
        "qwen_language_model_changed": False,
        "additional_pooling_network": False,
        "additional_temporal_network": False,
        "validation_used": False,
        "test_used_for_selection": True,
    }
    atomic_json(project_root() / "preflight.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen Qwen3-VL LGVQ linear-head baseline")
    parser.add_argument("command", choices=("preflight", "cache", "train", "all", "report"))
    parser.add_argument("--config", type=Path, default=project_root() / "config.json")
    parser.add_argument("--frames", type=int, choices=(4, 16))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        result = preflight(config)
    elif args.command == "cache":
        if args.frames is None:
            raise SystemExit("--frames is required for cache")
        result = build_feature_cache(config, args.frames, limit=args.limit)
    elif args.command == "train":
        if args.frames is None:
            raise SystemExit("--frames is required for train")
        if args.limit is not None:
            raise SystemExit("--limit is a cache smoke option and is forbidden for formal training")
        result = train_linear_head(config, args.frames)
    elif args.command == "all":
        if args.limit is not None:
            raise SystemExit("--limit is forbidden for the formal all command")
        preflight(config)
        result = {}
        for frames in config["frame_counts"]:
            result[f"cache_{frames}"] = build_feature_cache(config, frames)
            result[f"train_{frames}"] = train_linear_head(config, frames)
        result["comparison"] = combined_report(config)
    else:
        result = combined_report(config)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
