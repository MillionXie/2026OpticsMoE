from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from experiments.qwen3_vl_2b_spaq_multitask_iqa.io_utils import write_csv, write_json
from experiments.qwen3_vl_2b_spaq_multitask_iqa.metrics import regression_metrics

from . import TASK_NAMES
from .settings import Settings


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LABELED_SCORE = re.compile(
    rf"(?:score|rating)(?:\s+is)?\s*[:=]?\s*({NUMBER_PATTERN})",
    flags=re.IGNORECASE,
)
ANY_NUMBER = re.compile(NUMBER_PATTERN)
PREDICTION_FIELDS = [
    "sample_index", "image_name", "image_path", "task", "true_score",
    "predicted_score", "absolute_error", "parse_valid", "parse_strategy",
    "raw_response", "image_grid_thw", "generation_time_sec",
]


def parse_score(response: str) -> tuple[float | None, str]:
    """Parse one explicit 0--100 score without silently clipping invalid text."""

    text = str(response).strip()
    labeled = LABELED_SCORE.search(text)
    candidates: list[tuple[str, str]] = []
    if labeled:
        candidates.append((labeled.group(1), "labeled_score"))
    stripped = text.strip().rstrip(".。,%")
    if re.fullmatch(NUMBER_PATTERN, stripped):
        candidates.insert(0, (stripped, "number_only"))
    if not candidates:
        numbers = ANY_NUMBER.findall(text)
        if len(numbers) == 1:
            candidates.append((numbers[0], "single_number_in_text"))
        elif numbers:
            candidates.append((numbers[0], "first_number_in_text"))
    if not candidates:
        return None, "no_number"
    raw, strategy = candidates[0]
    try:
        value = float(raw)
    except ValueError:
        return None, "invalid_number"
    if not math.isfinite(value):
        return None, "non_finite"
    if not 0.0 <= value <= 100.0:
        return None, "out_of_range"
    return value, strategy


def preprocess_for_generation(
    processor: Any,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    system_prompt: str,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    texts = [
        _chat_text(processor, image, prompt, system_prompt)
        for image, prompt in zip(images, prompts)
    ]
    values = processor(
        text=texts,
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Qwen processor did not return: {missing}")
    inputs = {
        name: value
        for name, value in values.items()
        if torch.is_tensor(value) and name not in {"token_type_ids", "mm_token_type_ids"}
    }
    return inputs, texts


def generate_scores(
    model: torch.nn.Module,
    processor: Any,
    inputs: Mapping[str, torch.Tensor],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[list[str], list[list[int]], float]:
    if model.training:
        raise RuntimeError("Zero-shot Qwen must remain in eval mode")
    gpu_inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
    input_length = gpu_inputs["input_ids"].shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(
            **gpu_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    completion_ids = sequences[:, input_length:]
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        decoder = processor.tokenizer.batch_decode
    responses = [
        value.strip()
        for value in decoder(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]
    return responses, inputs["image_grid_thw"].tolist(), elapsed


def evaluate_zeroshot(
    model: torch.nn.Module,
    processor: Any,
    loader: Any,
    settings: Settings,
    expected_metadata: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = settings.output_dir / "generation_metadata.json"
    jsonl_path = settings.output_dir / "zeroshot_predictions.jsonl"
    csv_path = settings.output_dir / "zeroshot_predictions.csv"
    _validate_or_create_metadata(metadata_path, expected_metadata)
    saved = _load_jsonl(jsonl_path)
    by_index = {int(row["sample_index"]): row for row in saved}
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, 1):
        pending = [
            offset
            for offset, sample_index in enumerate(batch["sample_indices"].tolist())
            if int(sample_index) not in by_index
        ]
        if not pending:
            continue
        images = [batch["images"][index] for index in pending]
        prompts = [batch["prompts"][index] for index in pending]
        inputs, _ = preprocess_for_generation(
            processor, images, prompts, settings.system_prompt
        )
        responses, grids, elapsed = generate_scores(
            model, processor, inputs, device, settings.max_new_tokens
        )
        per_sample_time = elapsed / max(len(pending), 1)
        new_rows = []
        for local_index, batch_offset in enumerate(pending):
            response = responses[local_index]
            prediction, strategy = parse_score(response)
            truth = float(batch["scores"][batch_offset])
            row = {
                "sample_index": int(batch["sample_indices"][batch_offset]),
                "image_name": batch["image_names"][batch_offset],
                "image_path": batch["image_paths"][batch_offset],
                "task": batch["tasks"][batch_offset],
                "true_score": truth,
                "predicted_score": prediction,
                "absolute_error": abs(prediction - truth) if prediction is not None else None,
                "parse_valid": prediction is not None,
                "parse_strategy": strategy,
                "raw_response": response,
                "image_grid_thw": grids[local_index],
                "generation_time_sec": per_sample_time,
            }
            by_index[row["sample_index"]] = row
            new_rows.append(row)
        _append_jsonl(jsonl_path, new_rows)
        if batch_index % settings.save_interval_batches == 0:
            _write_prediction_csv(csv_path, list(by_index.values()))
        if batch_index % settings.log_interval_batches == 0 or batch_index == total_batches:
            valid = sum(bool(row["parse_valid"]) for row in by_index.values())
            print(
                f"zero-shot batch={batch_index}/{total_batches} "
                f"completed={len(by_index)} valid={valid} "
                f"parse_rate={valid / max(len(by_index), 1):.2%}"
            )
    rows = sorted(by_index.values(), key=lambda row: int(row["sample_index"]))
    _write_prediction_csv(csv_path, rows)
    metrics = zeroshot_metrics(rows)
    metrics.update(
        {
            "model_id": settings.model_id,
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": settings.max_new_tokens,
                "thinking_disable_requested": True,
                "thinking_template_fallback_if_unsupported": True,
            },
            "training_or_finetuning_used": False,
            "checkpoint_trained_for_spaq": False,
        }
    )
    write_json(settings.output_dir / "zeroshot_metrics.json", metrics)
    return rows, metrics


def zeroshot_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for task in TASK_NAMES:
        all_task = [row for row in rows if row["task"] == task]
        valid = [row for row in all_task if bool(row["parse_valid"])]
        if valid:
            result: dict[str, Any] = regression_metrics(
                [float(row["true_score"]) for row in valid],
                [float(row["predicted_score"]) for row in valid],
            )
            errors = np.asarray(
                [abs(float(row["predicted_score"]) - float(row["true_score"])) for row in valid]
            )
            result.update(
                {
                    "within_5_accuracy": float(np.mean(errors <= 5.0)),
                    "within_10_accuracy": float(np.mean(errors <= 10.0)),
                    "within_15_accuracy": float(np.mean(errors <= 15.0)),
                }
            )
        else:
            result = {
                "mae": None, "srcc": None, "plcc": None, "samples": 0,
                "within_5_accuracy": None, "within_10_accuracy": None,
                "within_15_accuracy": None,
            }
        result.update(
            {
                "requested_samples": len(all_task),
                "parse_failures": len(all_task) - len(valid),
                "parse_rate": len(valid) / max(len(all_task), 1),
            }
        )
        by_task[task] = result
    macro = {
        metric: float(np.mean([by_task[task][metric] for task in TASK_NAMES]))
        if all(by_task[task][metric] is not None for task in TASK_NAMES) else None
        for metric in (
            "mae", "srcc", "plcc", "within_5_accuracy",
            "within_10_accuracy", "within_15_accuracy",
        )
    }
    valid_count = sum(bool(row["parse_valid"]) for row in rows)
    return {
        "score_scale": [0.0, 100.0],
        "tasks": by_task,
        "macro_average": macro,
        "requested_task_samples": len(rows),
        "valid_task_samples": valid_count,
        "parse_failures": len(rows) - valid_count,
        "parse_rate": valid_count / max(len(rows), 1),
        "original_test_images": len({row["image_name"] for row in rows}),
    }


def _chat_text(
    processor: Any,
    image: Image.Image,
    prompt: str,
    system_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _validate_or_create_metadata(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"saved": saved.get(key), "current": value}
            for key, value in expected.items()
            if saved.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Zero-shot generation cache metadata mismatch: {mismatches}. "
                "Use a new output_dir or remove the incompatible generation cache."
            )
    else:
        write_json(path, dict(expected))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def _write_prediction_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
    serializable = []
    for row in ordered:
        value = dict(row)
        value["image_grid_thw"] = json.dumps(value["image_grid_thw"])
        serializable.append(value)
    write_csv(path, serializable, PREDICTION_FIELDS)
