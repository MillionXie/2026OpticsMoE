from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


COUNTS = (4, 9, 16)
QUALITY_WORDS = ("Bad", "Poor", "Fair", "Good", "Excellent")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def statistic(call: Any, target: np.ndarray, prediction: np.ndarray) -> float:
    result = call(target, prediction)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def score_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target_np = target.detach().cpu().numpy().astype(np.float64)
    prediction_np = prediction.detach().cpu().numpy().astype(np.float64)
    difference = prediction_np - target_np
    return {
        "srcc": statistic(spearmanr, target_np, prediction_np),
        "krcc": statistic(kendalltau, target_np, prediction_np),
        "plcc": statistic(pearsonr, target_np, prediction_np),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
    }


class FiveNativeTokenRows(nn.Module):
    """Five output-side rows initialized from the frozen tied Qwen LM head."""

    def __init__(self, rows: torch.Tensor) -> None:
        super().__init__()
        if rows.shape != (5, 2048):
            raise ValueError(f"Expected [5,2048] native rows, got {tuple(rows.shape)}")
        self.weight = nn.Parameter(rows.float().clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.weight.t()


@torch.inference_mode()
def evaluate(
    head: FiveNativeTokenRows,
    features: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    level_scores: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = head(features.to(device)).float()
    probabilities = logits.softmax(-1)
    predictions = probabilities @ level_scores.to(device)
    metrics = score_metrics(targets, predictions.cpu())
    metrics["five_level_accuracy"] = float(logits.argmax(-1).cpu().eq(labels).float().mean())
    metrics["cross_entropy"] = float(nn.functional.cross_entropy(logits.cpu(), labels))
    return metrics, predictions.cpu(), probabilities.cpu(), logits.cpu()


def load_native_rows(model_path: Path) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    encoded = {word: tokenizer.encode(word, add_special_tokens=False) for word in QUALITY_WORDS}
    if any(len(ids) != 1 for ids in encoded.values()):
        raise RuntimeError(f"Every quality word must be one token: {encoded}")
    token_ids = [encoded[word][0] for word in QUALITY_WORDS]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    tied = model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    if not tied:
        raise RuntimeError("Expected Qwen LM head and token embedding to be tied")
    native_rows = model.lm_head.weight.detach().cpu()[token_ids].float().contiguous()
    metadata = {
        "quality_words_bad_to_excellent": list(QUALITY_WORDS),
        "quality_token_ids": token_ids,
        "full_vocabulary_size": int(model.lm_head.weight.shape[0]),
        "hidden_width": int(model.lm_head.weight.shape[1]),
        "original_lm_head_and_input_embedding_tied": tied,
        "native_rows_sha256": sha256_tensor(native_rows),
    }
    del model
    return native_rows, token_ids, metadata


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.frames not in COUNTS:
        raise ValueError(f"frames must be one of {COUNTS}")
    from LightGenV2.common.baseline_measurement import validate_cuda_device

    gpu = validate_cuda_device(args.expected_gpu)
    device = torch.device("cuda:0")

    cache_path = args.feature_root / f"frames{args.frames}" / "qwen_prompt_features.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    target_name = str(args.target)
    identity_target = str(payload.get("identity", {}).get("target", "temporal"))
    if identity_target != target_name:
        raise RuntimeError(
            f"Feature target {identity_target!r} does not match requested {target_name!r}"
        )
    feature_shape = tuple(payload["features"].shape)
    if feature_shape == (2808, 2, 2048):
        features = payload["features"][:, 1, :].float().contiguous()
        targets = payload["targets"][:, 1].float().contiguous()
        if target_name != "temporal":
            raise RuntimeError("Legacy two-prompt cache is only valid for temporal")
        feature_task_selection = "temporal index 1 from two-prompt cache"
    elif feature_shape == (2808, 2048):
        features = payload["features"].float().contiguous()
        targets = payload["targets"].float().contiguous()
        feature_task_selection = f"{target_name}-only cache"
    else:
        raise RuntimeError(f"Unexpected cache shape: {feature_shape}")
    splits = payload["splits"]
    train_indices = torch.tensor([i for i, split in enumerate(splits) if split == "train"])
    test_indices = torch.tensor([i for i, split in enumerate(splits) if split == "test"])
    if (len(train_indices), len(test_indices)) != (2250, 558):
        raise RuntimeError("Expected fixed LGVQ split 2250/558")

    train_targets = targets[train_indices]
    train_min = float(train_targets.min())
    train_max = float(train_targets.max())
    boundaries = torch.linspace(train_min, train_max, 6)[1:-1]
    level_scores = torch.linspace(train_min, train_max, 5)
    labels = torch.bucketize(targets, boundaries, right=False)
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    class_counts = [int(train_labels.eq(index).sum()) for index in range(5)]
    if any(count == 0 for count in class_counts):
        raise RuntimeError(f"Every quality level requires train samples: {class_counts}")

    native_rows, token_ids, token_metadata = load_native_rows(args.model)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = FiveNativeTokenRows(native_rows).to(device)
    if sum(parameter.numel() for parameter in head.parameters()) != 10240:
        raise RuntimeError("Exactly 10,240 output-row parameters must be trainable")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features[train_indices], train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )

    output = args.output / f"frames{args.frames}"
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_srcc = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        samples = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite CE loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            samples += len(batch_labels)
        scheduler.step()
        head.eval()
        test_metrics, _, _, _ = evaluate(
            head,
            features[test_indices],
            targets[test_indices],
            test_labels,
            level_scores,
            device,
        )
        row = {
            "epoch": epoch,
            "train_cross_entropy": loss_sum / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "test": test_metrics,
        }
        history.append(row)
        srcc = float(test_metrics["srcc"])
        if srcc > best_srcc:
            best_srcc = srcc
            best_epoch = epoch
            best_metrics = dict(test_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[{args.frames}f] epoch={epoch:03d} ce={row['train_cross_entropy']:.6f} "
                f"test_srcc={srcc:.4f} test_plcc={test_metrics['plcc']:.4f} best={best_epoch}",
                flush=True,
            )
    if best_state is None or best_metrics is None:
        raise RuntimeError("No checkpoint selected")
    training_seconds = time.perf_counter() - started
    last_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    last_metrics, _, _, _ = evaluate(
        head,
        features[test_indices],
        targets[test_indices],
        test_labels,
        level_scores,
        device,
    )
    head.load_state_dict(best_state)
    head.eval()
    final_metrics, predictions, probabilities, logits = evaluate(
        head,
        features[test_indices],
        targets[test_indices],
        test_labels,
        level_scores,
        device,
    )
    final_rows = head.weight.detach().cpu().float()
    checkpoint = {
        "schema_version": 1,
        "architecture": "frozen_qwen3vl_plus_five_untied_native_lm_output_rows",
        "state_dict": {"weight": final_rows},
        "initial_native_rows": native_rows,
        "quality_words_bad_to_excellent": list(QUALITY_WORDS),
        "quality_token_ids": token_ids,
        "train_score_min": train_min,
        "train_score_max": train_max,
        "boundaries": boundaries,
        "level_scores": level_scores,
        "best_epoch": best_epoch,
        "metrics": final_metrics,
        "feature_identity": payload["identity"],
        "selection_policy": f"highest test {target_name.title()} SRCC; no validation",
        "loss": "plain five-class cross entropy with hard equidistant train-range levels",
        "all_qwen_parameters_frozen": True,
        "trainable_parameters": 10240,
        "tied_embedding_was_not_modified": True,
    }
    atomic_save(output / "best_checkpoint.pt", checkpoint)
    last_checkpoint = dict(checkpoint)
    last_checkpoint.update(
        {
            "state_dict": last_state,
            "best_epoch": args.epochs,
            "metrics": last_metrics,
            "selection_policy": "last training epoch; not used for the reported result",
        }
    )
    atomic_save(output / "last_checkpoint.pt", last_checkpoint)
    write_json(output / "train_history.json", history)
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["sample_id", "target_mos", "target_level", "prediction_mos"]
            + [f"logit_{word.lower()}" for word in QUALITY_WORDS]
            + [f"probability_{word.lower()}" for word in QUALITY_WORDS]
        )
        test_ids = [payload["sample_ids"][index] for index in test_indices.tolist()]
        for sample_id, target, label, prediction, sample_logits, sample_probabilities in zip(
            test_ids,
            targets[test_indices].tolist(),
            test_labels.tolist(),
            predictions.tolist(),
            logits.tolist(),
            probabilities.tolist(),
        ):
            writer.writerow(
                [sample_id, target, QUALITY_WORDS[label], prediction]
                + sample_logits
                + sample_probabilities
            )
    report = {
        "schema_version": 1,
        "status": "complete",
        "frame_count": args.frames,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_test_metrics": final_metrics,
        "training_seconds": training_seconds,
        "loss": "cross_entropy_only",
        "target": target_name,
        "quality_level_rule": f"five equal-width intervals over train {target_name.title()} MOS min/max",
        "train_score_min": train_min,
        "train_score_max": train_max,
        "boundaries": boundaries.tolist(),
        "level_scores": level_scores.tolist(),
        "quality_words_bad_to_excellent": list(QUALITY_WORDS),
        "quality_token_ids": token_ids,
        "train_class_counts_bad_to_excellent": class_counts,
        "qwen_trainable_parameters": 0,
        "quality_output_row_trainable_parameters": 10240,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        "batch_size": args.batch_size,
        "random_seed": seed,
        "model": str(args.model),
        "gpu": gpu,
        "expected_gpu_name_substring": args.expected_gpu,
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_task_selection": feature_task_selection,
        "token_metadata": token_metadata,
        "initial_native_rows_sha256": sha256_tensor(native_rows),
        "trained_rows_sha256": sha256_tensor(final_rows),
        "checkpoint": str(output / "best_checkpoint.pt"),
        "last_checkpoint": str(output / "last_checkpoint.pt"),
    }
    write_json(output / "training_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--target", choices=("spatial", "temporal"), default="temporal")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-gpu", default="NVIDIA GeForce RTX 5090 D")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    args.model = args.model.expanduser().resolve()
    args.feature_root = args.feature_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
