"""Frozen Qwen3-VL-2B-Instruct OpenMoji baseline on RTX 5090 D."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

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
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.assets import (
    ICON_SPECS,
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
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 OpenMoji test rows, got {len(rows)}")
    return rows


def _prompt(instruction: str) -> str:
    categories = ", ".join(f"{item.index}={item.name}" for item in ICON_SPECS)
    return (
        "The image is a 6 by 6 object grid. Empty cells are category 0. "
        f"The only object categories are: {categories}. "
        f"Apply this instruction: {instruction} "
        "Return only compact JSON with exactly two row-major arrays of 36 integers: "
        '{"target":[...],"edit":[...]}. target uses category ids 0 through 16; '
        "edit is 1 exactly where source and target differ, otherwise 0."
    )


def _parse_array(value: Any, *, binary: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.size != 36:
        raise ValueError("grid must contain exactly 36 integers")
    array = array.reshape(6, 6)
    if binary:
        if not np.isin(array, [0, 1]).all():
            raise ValueError("edit grid must be binary")
    elif not ((0 <= array) & (array <= 16)).all():
        raise ValueError("target category id must be between 0 and 16")
    return array


def _parse(text: str) -> tuple[np.ndarray, np.ndarray]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("no JSON object")
    value = json.loads(match.group(0))
    return _parse_array(value["target"], binary=False), _parse_array(
        value["edit"], binary=True
    )


def _sample_metrics(
    prediction: np.ndarray,
    predicted_edit: np.ndarray,
    target: np.ndarray,
    true_edit: np.ndarray,
) -> dict[str, float]:
    correct = prediction == target
    foreground = target > 0
    changed = true_edit.astype(bool)
    intersection = np.logical_and(predicted_edit, true_edit).sum()
    union = np.logical_or(predicted_edit, true_edit).sum()
    predicted_objects = {
        (int(prediction[row, col]), int(row), int(col))
        for row, col in zip(*np.nonzero(prediction))
    }
    target_objects = {
        (int(target[row, col]), int(row), int(col))
        for row, col in zip(*np.nonzero(target))
    }
    matched = len(predicted_objects & target_objects)
    precision = matched / max(1, len(predicted_objects))
    recall = matched / max(1, len(target_objects))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "changed_cell_accuracy": float(correct[changed].mean()),
        "foreground_category_accuracy": float(correct[foreground].mean()),
        "edit_grid_iou": float(intersection / union if union else 1.0),
        "object_f1": float(f1),
        "scene_exact_match": float(np.array_equal(prediction, target)),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This formal baseline requires NVIDIA GeForce RTX 5090 D")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = args.model.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_root / "test.jsonl"
    rows = _read_jsonl(manifest)
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
    timer = FirstBlockTimer(model.model.visual.blocks[0])
    sampler = NvidiaSmiPowerSampler()
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)
    records: list[dict[str, Any]] = []
    try:
        for index in range(args.warmup_forwards):
            row = rows[index % len(rows)]
            relative = Path(str(row["relative_dir"]))
            image_path = data_root / relative / str(row["files"]["source"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _prompt(str(row["instruction"]))},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            timer.reset()
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            try:
                _parse(response)
            except Exception:
                pass
            timer.finish()
        for index, row in enumerate(rows):
            relative = Path(str(row["relative_dir"]))
            image_path = data_root / relative / str(row["files"]["source"])
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
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            torch.cuda.synchronize()
            timer.reset()
            sampler.set_phase(f"active:{index}")
            try:
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
                new_tokens = generated[:, inputs["input_ids"].shape[1] :]
                response = processor.batch_decode(
                    new_tokens, skip_special_tokens=True
                )[0]
                parse_error = None
                try:
                    prediction, predicted_edit = _parse(response)
                except Exception as error:
                    parse_error = f"{type(error).__name__}: {error}"
                    prediction = np.zeros((6, 6), dtype=np.int64)
                    predicted_edit = np.zeros((6, 6), dtype=np.int64)
                timing = timer.finish()
            finally:
                sampler.set_phase(None)
            target = np.asarray(row["target_grid"], dtype=np.int64)
            true_edit = np.asarray(row["edit_grid"], dtype=np.int64)
            metrics = _sample_metrics(prediction, predicted_edit, target, true_edit)
            records.append(
                {
                    "sample_index": index,
                    "sample_id": row["sample_id"],
                    "task": row["task"],
                    "instruction": row["instruction"],
                    "parse_failed": parse_error is not None,
                    "parse_error": parse_error or "",
                    "response": response,
                    "cuda_ms": timing["cuda_ms"],
                    "host_ms": timing["host_ms"],
                    "generated_tokens": int(new_tokens.shape[1]),
                    **metrics,
                }
            )
            if index == 0 or (index + 1) % 25 == 0:
                print(
                    f"[OpenMoji] {index + 1}/{len(rows)} parse_failed={parse_error is not None} "
                    f"host={timing['host_ms']:.1f}ms tokens={new_tokens.shape[1]}",
                    flush=True,
                )
    finally:
        timer.close()
        power_samples = sampler.stop()

    metric_names = (
        "changed_cell_accuracy",
        "foreground_category_accuracy",
        "edit_grid_iou",
        "object_f1",
        "scene_exact_match",
    )
    performance = {
        "samples": len(records),
        "parse_failure_rate": float(
            np.mean([bool(row["parse_failed"]) for row in records])
        ),
        **{
            name: float(np.mean([float(row[name]) for row in records]))
            for name in metric_names
        },
    }
    cuda_latencies = [float(row["cuda_ms"]) for row in records]
    host_latencies = [float(row["host_ms"]) for row in records]
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "OpenMoji semantic interaction",
        "model": str(model_path),
        "qwen_frozen": True,
        "trainable_parameters": 0,
        "test_samples": len(records),
        "timing_samples": len(records),
        "explicit_warmup_forwards": args.warmup_forwards,
        "first_test_sample_included": False,
        "timing_boundary": (
            "input to native Vision Transformer block 0 through all native Vision/"
            "Language blocks, autoregressive compact-grid generation and CPU JSON parse"
        ),
        "primary_latency": "host_ms because the endpoint includes CPU JSON parsing",
        "performance": performance,
        "latency_cuda_ms": summarize(cuda_latencies),
        "latency_host_ms": summarize(host_latencies),
        "power": power_report(power_samples, host_latencies),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "environment": environment_report(),
        "model_load_seconds": model_load_seconds,
        "max_new_tokens": args.max_new_tokens,
    }
    write_json(run_dir / "baseline_report.json", report)
    _write_rows(run_dir / "predictions_timing.csv", records)
    save_power_samples(run_dir / "power_samples.csv", power_samples)
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--warmup-forwards", type=int, default=50)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
