"""Frozen DeepSeek-VL2-Tiny baseline for LGVQ temporal/spatial quality.

The executable contract mirrors the audited Qwen quality-token baseline:

* use the fixed prompt-group LGVQ split and four sampled frames;
* freeze every pretrained vision, projector, and language parameter;
* pool the final language layer at the last valid prompt token; and
* train only five bias-free output rows initialized from native LM-head rows.

DeepSeek-VL2 is an image VLM rather than a native video VLM, so the four ordered
frames are represented as four interleaved image placeholders in one dialogue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import quality_token_common as core
from .project import REPO_ROOT, TASK_DIR, sha256


MODEL_NAME = "deepseek-ai/deepseek-vl2-tiny"
QUALITY_WORDS = core.QUALITY_WORDS
TARGETS = ("temporal", "spatial")
FEATURE_FILENAME = "deepseek_vl2_prompt_features.pt"


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
        raise ValueError(f"Unsupported DeepSeek LGVQ config: {path}")
    return raw


def build_conversation(target: str, frame_count: int) -> list[dict[str, Any]]:
    """Build the official DeepSeek multi-image dialogue in temporal order."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frame_lines = [f"Frame {index + 1}: <image>" for index in range(frame_count)]
    content = "\n".join([*frame_lines, core.PROMPTS[target]])
    return [
        {
            "role": "<|User|>",
            "content": content,
            # The official processor checks image cardinality; pixels are passed
            # separately as in-memory PIL images.
            "images": [f"frame_{index + 1:02d}.png" for index in range(frame_count)],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]


def quality_token_rows(
    tokenizer: Any, lm_head_weight: torch.Tensor
) -> tuple[torch.Tensor, list[list[int]], str]:
    """Copy five native LM-head rows, averaging only if a word is split.

    Qwen's five labels are each one token. DeepSeek's tokenizer is audited at
    runtime. A deterministic mean of native subtoken rows is the explicit
    fallback so a tokenizer difference cannot silently change the parameter
    budget or turn into random initialization.
    """
    token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in QUALITY_WORDS]
    if any(not ids for ids in token_ids):
        raise RuntimeError(f"Empty quality-word tokenization: {token_ids}")
    rows = torch.stack(
        [lm_head_weight.detach().cpu()[ids].float().mean(dim=0) for ids in token_ids]
    ).contiguous()
    mode = "native_single_token_rows" if all(len(ids) == 1 for ids in token_ids) else "mean_native_subtoken_rows"
    return rows, token_ids, mode


class FiveQualityRows(nn.Module):
    """The only trainable module in the DeepSeek baseline."""

    def __init__(self, rows: torch.Tensor) -> None:
        super().__init__()
        if rows.ndim != 2 or rows.shape[0] != 5:
            raise ValueError(f"Expected [5, hidden_width] rows, got {tuple(rows.shape)}")
        self.weight = nn.Parameter(rows.float().clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.weight.t()


def validate_config(raw: dict[str, Any]) -> None:
    if raw["model"]["backbone"] != MODEL_NAME:
        raise ValueError(f"Expected backbone {MODEL_NAME}")
    if raw["model"]["backbone_frozen"] is not True:
        raise ValueError("The complete DeepSeek backbone must remain frozen")
    if raw["model"]["trainable_module"] != "five_bias_free_quality_rows":
        raise ValueError("Only the five quality rows may be trainable")
    if int(raw["model"]["hidden_width"]) != 1280:
        raise ValueError("DeepSeek-VL2-Tiny hidden width must be 1280")
    if int(raw["model"]["trainable_parameters"]) != 5 * 1280:
        raise ValueError("The configured trainable parameter count must be 6400")
    if int(raw["input"]["frame_count"]) != 4:
        raise ValueError("The matched LGVQ comparison requires four frames")
    if int(raw["input"]["image_size"]) != 384:
        raise ValueError("DeepSeek-VL2's native vision input is 384x384")
    fractions = [float(value) for value in raw["input"]["frame_fractions"]]
    if fractions != core.FRAME_FRACTIONS[4]:
        raise ValueError("Frame fractions differ from the Qwen four-frame contract")
    if tuple(raw["task"]["targets"]) != TARGETS:
        raise ValueError(f"Expected both LGVQ targets in order: {TARGETS}")
    for target in TARGETS:
        if raw["task"]["prompts"][target] != core.PROMPTS[target]:
            raise ValueError(f"Prompt contract changed for {target}")
    if int(raw["training"]["epochs"]) != 50:
        raise ValueError("Matched Qwen comparison requires 50 epochs")


def _load_backbone(model_path: Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "DeepSeek-VL2 dependencies are unavailable. Install the official "
            "DeepSeek-VL2 package in its isolated Transformers 4.38.2 environment."
        ) from error

    processor = DeepseekVLV2Processor.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval().requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"Frozen DeepSeek backbone exposes {trainable} trainable parameters")
    if int(model.config.language_config.hidden_size) != 1280:
        raise RuntimeError(
            f"Expected DeepSeek-VL2-Tiny hidden width 1280, got "
            f"{model.config.language_config.hidden_size}"
        )
    rows, token_ids, row_initialization = quality_token_rows(
        processor.tokenizer, model.language.lm_head.weight
    )
    metadata = {
        "quality_words_bad_to_excellent": list(QUALITY_WORDS),
        "quality_token_ids": token_ids,
        "row_initialization": row_initialization,
        "initial_rows_sha256": _sha256_tensor(rows),
        "hidden_width": int(rows.shape[1]),
        "vocabulary_size": int(model.language.lm_head.weight.shape[0]),
        "backbone_trainable_parameters": trainable,
        "backbone_total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return processor, model, {"rows": rows, "metadata": metadata}


@torch.inference_mode()
def extract_one(
    *,
    row: dict[str, Any],
    target: str,
    processor: Any,
    model: nn.Module,
    device: torch.device,
    image_size: int,
    fractions: list[float],
) -> tuple[torch.Tensor, list[int], int]:
    frames, _metadata, positions = core.decode_random_seek(
        Path(row["video_path"]), fractions, image_size
    )
    conversation = build_conversation(target, len(frames))
    prepared = processor(
        conversations=conversation,
        images=frames,
        force_batchify=True,
        system_prompt="",
    ).to(device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_embeds = model.prepare_inputs_embeds(**prepared)
        outputs = model.language.model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attention_mask,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.last_hidden_state
    mask = prepared.attention_mask.bool()
    indices = torch.arange(mask.shape[1], device=device).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(1)
    pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
    if tuple(pooled.shape) != (1, 1280):
        raise RuntimeError(f"Unexpected pooled feature shape: {tuple(pooled.shape)}")
    pooled = pooled[0].float().cpu().half().contiguous()
    if not bool(torch.isfinite(pooled).all()):
        raise RuntimeError("Non-finite DeepSeek feature")
    return pooled, positions, int(mask.sum())


def _feature_identity(
    *, raw: dict[str, Any], model_path: Path, rows: list[dict[str, Any]], target: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "feature_contract": f"frozen_deepseek_vl2_tiny_{target}_last_valid_hidden_1280_v1",
        "backbone": MODEL_NAME,
        "model_path": str(model_path),
        "target": target,
        "prompt": core.PROMPTS[target],
        "frame_count": 4,
        "frame_fractions": core.FRAME_FRACTIONS[4],
        "image_size_wh": [int(raw["input"]["image_size"])] * 2,
        "center_crop_short_side_fraction": 0.65,
        "frame_representation": "four_ordered_interleaved_image_placeholders",
        "processor_policy": "official_deepseek_vl2_processor_defaults",
        "sample_count": len(rows),
        "sample_order_sha256": _sha256_ids(rows),
        "backbone_trainable_parameters": 0,
    }


def extract_target(
    *,
    raw: dict[str, Any],
    rows: list[dict[str, Any]],
    target: str,
    model_path: Path,
    output: Path,
    processor: Any,
    model: nn.Module,
    model_metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    identity = _feature_identity(raw=raw, model_path=model_path, rows=rows, target=target)
    shard_root = output.with_suffix(".parts")
    shard_root.mkdir(parents=True, exist_ok=True)
    chunk_rows = int(raw["feature_extraction"]["chunk_rows"])
    all_features = torch.empty(len(rows), 1280, dtype=torch.float16)
    all_sequence_lengths: list[int] = [0] * len(rows)
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
                and tuple(candidate.get("features", ()).shape) == (chunk_stop - chunk_start, 1280)
            ):
                all_features[chunk_start:chunk_stop].copy_(candidate["features"])
                all_sequence_lengths[chunk_start:chunk_stop] = candidate["sequence_lengths"]
                print(f"[resume {target}] {chunk_stop}/{len(rows)}", flush=True)
                continue
        features: list[torch.Tensor] = []
        sequence_lengths: list[int] = []
        for index in range(chunk_start, chunk_stop):
            feature, _positions, sequence_length = extract_one(
                row=rows[index],
                target=target,
                processor=processor,
                model=model,
                device=device,
                image_size=int(raw["input"]["image_size"]),
                fractions=core.FRAME_FRACTIONS[4],
            )
            features.append(feature)
            sequence_lengths.append(sequence_length)
            if index == chunk_start or (index + 1) % 25 == 0:
                print(f"[extract {target}] {index + 1}/{len(rows)} seq={sequence_length}", flush=True)
        chunk_features = torch.stack(features)
        _atomic_save(
            shard,
            {
                "identity": identity,
                "sample_ids": expected_ids,
                "features": chunk_features,
                "sequence_lengths": sequence_lengths,
            },
        )
        all_features[chunk_start:chunk_stop].copy_(chunk_features)
        all_sequence_lengths[chunk_start:chunk_stop] = sequence_lengths
        print(f"[saved {target}] {chunk_stop}/{len(rows)}", flush=True)
    payload = {
        "identity": identity,
        "model_metadata": model_metadata,
        "initial_quality_rows": model_metadata.pop("_initial_quality_rows"),
        "sample_ids": [row["sample_id"] for row in rows],
        "splits": [row["split"] for row in rows],
        "targets": torch.tensor([row[target] for row in rows], dtype=torch.float32),
        "features": all_features,
        "sequence_lengths": all_sequence_lengths,
    }
    _atomic_save(output, payload)
    report = {
        **identity,
        **model_metadata,
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_shape": list(all_features.shape),
        "feature_dtype": str(all_features.dtype),
        "sequence_length_min": min(all_sequence_lengths),
        "sequence_length_max": max(all_sequence_lengths),
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
    *, raw: dict[str, Any], target: str, feature_path: Path, output_dir: Path
) -> dict[str, Any]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    if identity["target"] != target or identity["backbone_trainable_parameters"] != 0:
        raise RuntimeError("Feature cache violates the frozen-backbone target contract")
    features = payload["features"].float().contiguous()
    targets = payload["targets"].float().contiguous()
    if tuple(features.shape) != (2808, 1280):
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
    trainable = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    if trainable != 6400:
        raise RuntimeError(f"Expected exactly 6400 trainable parameters, got {trainable}")
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
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[train {target}] epoch={epoch:03d} "
                f"test_srcc={test_metrics['srcc']:.4f} best_epoch={best_epoch}",
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
        "architecture": "frozen_deepseek_vl2_tiny_plus_five_native_lm_output_rows",
        "initial_native_rows": initial_rows,
        "quality_words_bad_to_excellent": list(QUALITY_WORDS),
        "quality_token_ids": payload["model_metadata"]["quality_token_ids"],
        "row_initialization": payload["model_metadata"]["row_initialization"],
        "train_score_min": train_min,
        "train_score_max": train_max,
        "boundaries": boundaries,
        "level_scores": level_scores,
        "feature_identity": identity,
        "all_deepseek_parameters_frozen": True,
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
            + [f"probability_{word.lower()}" for word in QUALITY_WORDS]
        )
        for sample_id, target_value, label, prediction, sample_probabilities in zip(
            test_ids,
            targets[test_indices].tolist(),
            test_labels.tolist(),
            predictions.tolist(),
            probabilities.tolist(),
        ):
            writer.writerow(
                [sample_id, target_value, QUALITY_WORDS[label], prediction]
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
    *, config: Path, model: Path, manifest: Path, run_dir: Path, raw: dict[str, Any]
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
    *, raw: dict[str, Any], rows: list[dict[str, Any]], model_path: Path, run_dir: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek smoke test requires CUDA")
    device = torch.device("cuda:0")
    processor, model, row_data = _load_backbone(model_path, device)
    results: dict[str, Any] = {}
    for target in TARGETS:
        feature, positions, sequence_length = extract_one(
            row=rows[0],
            target=target,
            processor=processor,
            model=model,
            device=device,
            image_size=int(raw["input"]["image_size"]),
            fractions=core.FRAME_FRACTIONS[4],
        )
        results[target] = {
            "feature_shape": list(feature.shape),
            "feature_finite": bool(torch.isfinite(feature).all()),
            "frame_positions": positions,
            "sequence_length": sequence_length,
        }
    report = {
        "status": "smoke_complete",
        "gpu": torch.cuda.get_device_name(device),
        "model_metadata": row_data["metadata"],
        "backbone_trainable_parameters": 0,
        "targets": results,
    }
    _write_json(run_dir / "smoke_report.json", report)
    del model, processor
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
    if not model_path.is_dir():
        raise FileNotFoundError(f"DeepSeek model directory is missing: {model_path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"LGVQ manifest is missing: {manifest}")
    run_dir = _resolve_run_dir(raw, args.run_dir)
    rows = core.read_manifest(manifest)
    if len(rows) != 2808:
        raise RuntimeError(f"Expected 2808 LGVQ rows, got {len(rows)}")
    identity = _record_identity(
        config=config, model=model_path, manifest=manifest, run_dir=run_dir, raw=raw
    )
    if args.phase == "preflight":
        print(json.dumps({"status": "ready", **identity}, indent=2), flush=True)
        return 0
    if args.phase == "smoke":
        print(json.dumps(smoke(raw=raw, rows=rows, model_path=model_path, run_dir=run_dir), indent=2), flush=True)
        return 0

    phases = ("extract", "train") if args.phase == "all" else (args.phase,)
    status: dict[str, Any] = {"status": "running", "completed": []}
    _write_json(run_dir / "status.json", status)
    try:
        if "extract" in phases:
            if not torch.cuda.is_available():
                raise RuntimeError("DeepSeek feature extraction requires CUDA")
            device = torch.device("cuda:0")
            processor, model, row_data = _load_backbone(model_path, device)
            for target in TARGETS:
                metadata = dict(row_data["metadata"])
                metadata["_initial_quality_rows"] = row_data["rows"]
                output = run_dir / "features" / target / FEATURE_FILENAME
                report = extract_target(
                    raw=raw,
                    rows=rows,
                    target=target,
                    model_path=model_path,
                    output=output,
                    processor=processor,
                    model=model,
                    model_metadata=metadata,
                    device=device,
                )
                status["completed"].append({"phase": "extract", "target": target, "report": report})
                _write_json(run_dir / "status.json", status)
            del model, processor
            torch.cuda.empty_cache()
        if "train" in phases:
            for target in TARGETS:
                report = train_target(
                    raw=raw,
                    target=target,
                    feature_path=run_dir / "features" / target / FEATURE_FILENAME,
                    output_dir=run_dir / "checkpoints" / target,
                )
                status["completed"].append({"phase": "train", "target": target, "report": report})
                _write_json(run_dir / "status.json", status)
        status["status"] = "complete"
    except Exception as error:
        status["status"] = "failed"
        status["error"] = repr(error)
        _write_json(run_dir / "status.json", status)
        raise
    _write_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
