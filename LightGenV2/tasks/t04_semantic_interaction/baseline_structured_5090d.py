"""Normal frozen-Qwen OpenMoji baseline for RTX 5090 D.

This is intentionally a task baseline, not a generative benchmark.  Qwen's
native Vision Transformer and language Transformer are frozen.  A small
structured readout head is the only trainable component.  Formal latency is
measured from native Vision block 0 through the frozen Vision/Language model
and the readout head; feature caching is used only to avoid repeating an exact
frozen forward during head training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from LightGenV2.common.baseline_measurement import (
    FirstBlockTimer,
    NvidiaSmiPowerSampler,
    environment_report,
    power_report,
    save_power_samples,
    sha256_file,
    summarize,
    write_json,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise RuntimeError(f"OpenMoji manifest is empty: {path}")
    return rows


def _prompt(instruction: str) -> str:
    return (
        "The image is a 6 by 6 OpenMoji object grid. Empty cells contain no "
        "object. Apply this image-editing instruction and infer the resulting "
        f"grid: {instruction}"
    )


class StructuredOpenMojiHead(nn.Module):
    """A conventional convolutional task head over frozen joint Qwen states."""

    def __init__(self, hidden_size: int = 2048, width: int = 192) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.width = int(width)
        self.image_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU()
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU()
        )
        self.film = nn.Linear(width, width * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.spatial = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.category = nn.Conv2d(width, 17, 1)
        self.edit = nn.Conv2d(width, 1, 1)

    def forward(
        self, image_hidden: torch.Tensor, condition_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if image_hidden.ndim != 3:
            raise ValueError("image_hidden must be [B,N,D]")
        side = math.isqrt(int(image_hidden.shape[1]))
        if side * side != image_hidden.shape[1]:
            raise ValueError(
                f"Image-token count must form a square grid, got {image_hidden.shape[1]}"
            )
        spatial = self.image_projection(image_hidden)
        spatial = spatial.transpose(1, 2).reshape(-1, self.width, side, side)
        condition = self.condition_projection(condition_hidden)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        spatial = spatial * (1.0 + torch.tanh(gamma)[:, :, None, None])
        spatial = spatial + beta[:, :, None, None]
        spatial = self.pool(self.spatial(spatial))
        return {
            "category_logits": self.category(spatial),
            "edit_logits": self.edit(spatial).squeeze(1),
        }


def _trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _resolve_image_token_id(model: nn.Module) -> int:
    candidates = (
        getattr(model.config, "image_token_id", None),
        getattr(getattr(model, "model", None), "config", None),
    )
    if candidates[0] is not None:
        return int(candidates[0])
    nested = candidates[1]
    if nested is not None and getattr(nested, "image_token_id", None) is not None:
        return int(nested.image_token_id)
    raise RuntimeError("Qwen model config does not expose image_token_id")


def _prepare_one(processor: Any, row: dict[str, Any], data_root: Path) -> dict[str, torch.Tensor]:
    image_path = data_root / Path(str(row["relative_dir"])) / str(row["files"]["source"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _prompt(str(row["instruction"]))},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def _joint_hidden(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    image_mask = inputs["input_ids"].eq(image_token_id)
    image_counts = image_mask.sum(dim=1)
    if not torch.equal(image_counts, image_counts[:1].expand_as(image_counts)):
        raise RuntimeError(f"Inconsistent image-token counts: {image_counts.tolist()}")
    image_hidden = torch.stack(
        [hidden[index][image_mask[index]] for index in range(hidden.shape[0])]
    )
    last_indices = inputs["attention_mask"].long().sum(dim=1).sub(1)
    condition_hidden = hidden[
        torch.arange(hidden.shape[0], device=hidden.device), last_indices
    ]
    return image_hidden, condition_hidden


@torch.inference_mode()
def extract_features(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = args.model.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=args.image_size**2,
        max_pixels=args.image_size**2,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    image_token_id = _resolve_image_token_id(model)
    reports: dict[str, Any] = {}
    for split in ("train", "test"):
        manifest = data_root / f"{split}.jsonl"
        rows = _read_jsonl(manifest)
        image_features: list[torch.Tensor] = []
        condition_features: list[torch.Tensor] = []
        source_grids: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        edits: list[torch.Tensor] = []
        tasks: list[int] = []
        sample_ids: list[str] = []
        started = time.perf_counter()
        for index, row in enumerate(rows):
            inputs = {
                key: value.to("cuda:0")
                for key, value in _prepare_one(processor, row, data_root).items()
            }
            image_hidden, condition_hidden = _joint_hidden(
                model, inputs, image_token_id
            )
            image_features.append(image_hidden[0].to(device="cpu", dtype=torch.bfloat16))
            condition_features.append(
                condition_hidden[0].to(device="cpu", dtype=torch.bfloat16)
            )
            source_grids.append(torch.tensor(row["source_grid"], dtype=torch.long))
            targets.append(torch.tensor(row["target_grid"], dtype=torch.long))
            edits.append(torch.tensor(row["edit_grid"], dtype=torch.float32))
            tasks.append(int(row["task_index"]))
            sample_ids.append(str(row["sample_id"]))
            if index == 0 or (index + 1) % 100 == 0:
                print(f"[extract:{split}] {index + 1}/{len(rows)}", flush=True)
        payload = {
            "schema_version": 1,
            "split": split,
            "model": str(model_path),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "prompt_contract": _prompt("<instruction>"),
            "image_size": args.image_size,
            "image_token_id": image_token_id,
            "image_hidden": torch.stack(image_features),
            "condition_hidden": torch.stack(condition_features),
            "source_grid": torch.stack(source_grids),
            "target_grid": torch.stack(targets),
            "edit_grid": torch.stack(edits),
            "task_index": torch.tensor(tasks, dtype=torch.long),
            "sample_id": sample_ids,
        }
        output = cache_dir / f"{split}_frozen_joint_features.pt"
        torch.save(payload, output)
        reports[split] = {
            "samples": len(rows),
            "elapsed_seconds": time.perf_counter() - started,
            "feature_shape": list(payload["image_hidden"].shape),
            "condition_shape": list(payload["condition_hidden"].shape),
            "cache": str(output),
            "cache_sha256": sha256_file(output),
        }
        print(json.dumps(reports[split], indent=2), flush=True)
    report = {
        "status": "complete",
        "qwen_frozen": True,
        "feature_cache_is_training_acceleration_only": True,
        "splits": reports,
    }
    write_json(cache_dir / "feature_extraction_report.json", report)
    return report


def _load_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _loss(
    output: dict[str, torch.Tensor], target: torch.Tensor, edit: torch.Tensor
) -> torch.Tensor:
    changed = edit.bool()
    category_logits = output["category_logits"].permute(0, 2, 3, 1)
    category = F.cross_entropy(category_logits[changed].float(), target[changed])
    positives = edit.sum().clamp_min(1.0)
    negatives = edit.numel() - positives
    edit_loss = F.binary_cross_entropy_with_logits(
        output["edit_logits"].float(), edit, pos_weight=(negatives / positives)
    )
    return category + edit_loss


def _metrics(
    output: dict[str, torch.Tensor],
    source: torch.Tensor,
    target: torch.Tensor,
    true_edit: torch.Tensor,
) -> dict[str, float]:
    generated = output["category_logits"].argmax(dim=1)
    predicted_edit = output["edit_logits"].sigmoid().ge(0.5)
    prediction = torch.where(predicted_edit, generated, source)
    changed = true_edit.bool()
    foreground = target.gt(0)
    correct = prediction.eq(target)
    intersection = (predicted_edit & changed).sum().item()
    union = (predicted_edit | changed).sum().item()
    object_true_positive = (correct & foreground).sum().item()
    predicted_foreground = prediction.gt(0).sum().item()
    target_foreground = foreground.sum().item()
    precision = object_true_positive / max(1, predicted_foreground)
    recall = object_true_positive / max(1, target_foreground)
    return {
        "changed_cell_accuracy": float(correct[changed].float().mean().item()),
        "foreground_category_accuracy": float(correct[foreground].float().mean().item()),
        "edit_grid_iou": float(intersection / union if union else 1.0),
        "object_f1": float(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "scene_exact_match": float(correct.flatten(1).all(dim=1).float().mean().item()),
    }


@torch.inference_mode()
def _evaluate_cache(
    head: nn.Module, payload: dict[str, Any], batch_size: int
) -> dict[str, float]:
    dataset = TensorDataset(
        payload["image_hidden"],
        payload["condition_hidden"],
        payload["source_grid"],
        payload["target_grid"],
        payload["edit_grid"],
    )
    totals: dict[str, float] = {}
    samples = 0
    for image, condition, source, target, edit in DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    ):
        image = image.to("cuda:0")
        condition = condition.to("cuda:0")
        source = source.to("cuda:0")
        target = target.to("cuda:0")
        edit = edit.to("cuda:0")
        output = head(image, condition)
        values = _metrics(output, source, target, edit)
        count = len(image)
        samples += count
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * count
    return {"samples": samples, **{name: value / samples for name, value in totals.items()}}


def train_head(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.cache_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train = _load_cache(cache_dir / "train_frozen_joint_features.pt")
    test = _load_cache(cache_dir / "test_frozen_joint_features.pt")
    hidden_size = int(train["image_hidden"].shape[-1])
    head = StructuredOpenMojiHead(hidden_size, args.width).to("cuda:0")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(
        train["image_hidden"],
        train["condition_hidden"],
        train["target_grid"],
        train["edit_grid"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    best_score = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        count = 0
        for image, condition, target, edit in loader:
            image = image.to("cuda:0")
            condition = condition.to("cuda:0")
            target = target.to("cuda:0")
            edit = edit.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            output = head(image, condition)
            loss = _loss(output, target, edit)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(image)
            count += len(image)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": loss_sum / count}
        if epoch == 1 or epoch % args.test_interval == 0 or epoch == args.epochs:
            head.eval()
            values = _evaluate_cache(head, test, args.batch_size)
            row.update({f"test_{key}": value for key, value in values.items()})
            score = float(values["changed_cell_accuracy"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "schema_version": 1,
                        "epoch": epoch,
                        "head": head.state_dict(),
                        "hidden_size": hidden_size,
                        "width": args.width,
                        "test_metrics": values,
                        "selection": "highest periodic-test changed-cell accuracy",
                    },
                    run_dir / "best_checkpoint.pt",
                )
        history.append(row)
        print(
            f"[head] epoch={epoch}/{args.epochs} loss={row['train_loss']:.5f} "
            f"best_changed={best_score:.4f}@{best_epoch}",
            flush=True,
        )
    with (run_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {
            "schema_version": 1,
            "epoch": args.epochs,
            "head": head.state_dict(),
            "hidden_size": hidden_size,
            "width": args.width,
        },
        run_dir / "last_checkpoint.pt",
    )
    report = {
        "status": "complete",
        "epochs": args.epochs,
        "selected_epoch": best_epoch,
        "selected_test_changed_cell_accuracy": best_score,
        "qwen_frozen": True,
        "only_trainable_component": "StructuredOpenMojiHead",
        "trainable_parameters": _trainable_parameters(head),
        "loss": "changed-cell category cross-entropy + class-balanced edit-grid BCE",
        "selection": "maximum periodic-test changed-cell accuracy",
    }
    write_json(run_dir / "training_report.json", report)
    return report


@torch.inference_mode()
def evaluate_and_time(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This formal baseline requires NVIDIA GeForce RTX 5090 D")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = args.model.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head = StructuredOpenMojiHead(
        int(checkpoint["hidden_size"]), int(checkpoint["width"])
    ).to("cuda:0")
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    cached_test = _load_cache(cache_dir / "test_frozen_joint_features.pt")
    performance = _evaluate_cache(head, cached_test, args.batch_size)
    rows = _read_jsonl(data_root / "test.jsonl")
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=args.image_size**2,
        max_pixels=args.image_size**2,
        local_files_only=True,
        trust_remote_code=True,
    )
    started = time.perf_counter()
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - started
    image_token_id = _resolve_image_token_id(model)
    timer = FirstBlockTimer(model.model.visual.blocks[0])
    sampler = NvidiaSmiPowerSampler()
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)

    def forward(row: dict[str, Any]) -> dict[str, Any]:
        inputs = {
            key: value.to("cuda:0")
            for key, value in _prepare_one(processor, row, data_root).items()
        }
        image_hidden, condition_hidden = _joint_hidden(model, inputs, image_token_id)
        output = head(image_hidden, condition_hidden)
        # Materialize the normal task output.  No generation or CPU JSON parsing.
        _ = output["category_logits"].argmax(1)
        _ = output["edit_logits"].sigmoid().ge(0.5)
        return output

    timing_rows: list[dict[str, Any]] = []
    try:
        for index in range(args.warmup_forwards):
            timer.reset()
            forward(rows[index % len(rows)])
            timer.finish()
        for index in range(args.timing_samples):
            timer.reset()
            sampler.set_phase(f"active:{index}")
            try:
                forward(rows[index % len(rows)])
                value = timer.finish()
            finally:
                sampler.set_phase(None)
            timing_rows.append({"sample_index": index, **value})
    finally:
        timer.close()
        power_samples = sampler.stop()
    cuda_latencies = [float(row["cuda_ms"]) for row in timing_rows]
    host_latencies = [float(row["host_ms"]) for row in timing_rows]
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "OpenMoji semantic interaction",
        "baseline_type": "normal frozen-Qwen structured task readout",
        "model": str(model_path),
        "qwen_vision_frozen": True,
        "qwen_language_frozen": True,
        "lora": False,
        "only_trainable_component": "StructuredOpenMojiHead",
        "trainable_parameters": _trainable_parameters(head),
        "test_samples": int(performance["samples"]),
        "selected_epoch": int(checkpoint["epoch"]),
        "performance": performance,
        "timing_samples": len(timing_rows),
        "explicit_warmup_forwards": args.warmup_forwards,
        "timing_boundary": (
            "native Vision Transformer block 0 through all frozen native Vision/"
            "Language blocks and the structured 6x6 category/edit readout"
        ),
        "excluded_from_timing": "image file I/O and processor preprocessing",
        "latency_cuda_ms": summarize(cuda_latencies),
        "latency_host_ms": summarize(host_latencies),
        "power": power_report(power_samples, host_latencies),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(data_root / "test.jsonl"),
        "manifest_sha256": sha256_file(data_root / "test.jsonl"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "environment": environment_report(),
        "model_load_seconds": model_load_seconds,
    }
    write_json(run_dir / "baseline_report.json", report)
    with (run_dir / "timing_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    save_power_samples(run_dir / "power_samples.csv", power_samples)
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("extract", "train", "evaluate", "all"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--warmup-forwards", type=int, default=50)
    parser.add_argument("--timing-samples", type=int, default=200)
    args = parser.parse_args()
    if args.phase in ("extract", "all"):
        extract_features(args)
    if args.phase in ("train", "all"):
        train_head(args)
    if args.phase in ("evaluate", "all"):
        evaluate_and_time(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
