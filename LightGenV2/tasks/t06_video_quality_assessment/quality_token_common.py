from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn


COUNTS = (4, 9, 16)
QUALITY_WORDS = ("Bad", "Poor", "Fair", "Good", "Excellent")
FRAME_FRACTIONS = {
    4: [0.10, 0.37, 0.63, 0.90],
    9: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    16: np.linspace(0.10, 0.90, 16).tolist(),
    25: np.linspace(0.10, 0.90, 25).tolist(),
    36: np.linspace(0.10, 0.90, 36).tolist(),
    49: np.linspace(0.10, 0.90, 49).tolist(),
}
PROMPTS = {
    "temporal": (
        "Please evaluate the temporal quality of this video and rate it using one of "
        "the following five levels: Excellent, Good, Fair, Poor, or Bad."
    ),
    "spatial": (
        "Please evaluate the spatial quality of this video and rate it using one of "
        "the following five levels: Excellent, Good, Fair, Poor, or Bad."
    ),
}
PROMPT = PROMPTS["temporal"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: list[dict[str, Any]] = []
    for row in rows:
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
    return result


def decode_random_seek(
    path: Path, fractions: Sequence[float], image_size: int
) -> tuple[list[Image.Image], Any, list[int]]:
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
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        width=image_size,
        height=image_size,
        duration=float(total) / fps,
        frames_indices=positions,
    )
    return frames, metadata, positions


def render_prompt(processor: Any, target: str = "temporal") -> str:
    if target not in PROMPTS:
        raise ValueError(f"target must be one of {sorted(PROMPTS)}, got {target!r}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": PROMPTS[target]},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def safe_statistic(result: Any) -> float:
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    difference = prediction - target
    return {
        "srcc": safe_statistic(spearmanr(target, prediction)),
        "krcc": safe_statistic(kendalltau(target, prediction)),
        "plcc": safe_statistic(pearsonr(target, prediction)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


class FiveNativeTokenRows(nn.Module):
    """Independent output-only copies of the five native LM-head rows."""

    def __init__(self, rows: torch.Tensor) -> None:
        super().__init__()
        if rows.shape != (5, 2048):
            raise ValueError(f"Expected [5,2048] quality rows, got {tuple(rows.shape)}")
        self.weight = nn.Parameter(rows.float().clone(), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.weight.t()


class BoundaryTimer:
    """CUDA-event and synchronized host timing from internal module hooks."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.state: dict[str, Any] = {}
        self.handles = [
            model.model.visual.blocks[0].register_forward_pre_hook(self._vision_start),
            model.model.language_model.layers[0].register_forward_pre_hook(self._language_start),
        ]

    def reset(self, expected_calls: int = 1) -> None:
        self.state = {
            "vision_event": torch.cuda.Event(enable_timing=True),
            "language_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "vision_calls": 0,
            "language_calls": 0,
            "expected_calls": int(expected_calls),
        }

    def _record(self, name: str, args: tuple[Any, ...]) -> None:
        self.state[f"{name}_calls"] += 1
        if self.state[f"{name}_calls"] == 1:
            self.state[f"{name}_host_started"] = time.perf_counter()
            self.state[f"{name}_event"].record()
            if args and isinstance(args[0], torch.Tensor):
                self.state[f"{name}_input_shape"] = list(args[0].shape)

    def _vision_start(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        self._record("vision", args)

    def _language_start(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        self._record("language", args)

    def finish(self) -> dict[str, Any]:
        self.state["end_event"].record()
        self.state["end_event"].synchronize()
        ended = time.perf_counter()
        for name in ("vision", "language"):
            if self.state[f"{name}_calls"] != self.state["expected_calls"]:
                raise RuntimeError(
                    f"Expected {self.state['expected_calls']} {name} block-0 calls, "
                    f"got {self.state[f'{name}_calls']}"
                )
        return {
            "vision_first_block_to_score_cuda_ms": float(
                self.state["vision_event"].elapsed_time(self.state["end_event"])
            ),
            "vision_first_block_to_score_host_ms": 1000.0
            * (ended - self.state["vision_host_started"]),
            "language_first_block_to_score_cuda_ms": float(
                self.state["language_event"].elapsed_time(self.state["end_event"])
            ),
            "language_first_block_to_score_host_ms": 1000.0
            * (ended - self.state["language_host_started"]),
            "vision_first_block_input_shape": self.state.get("vision_input_shape"),
            "language_first_block_input_shape": self.state.get("language_input_shape"),
            "complete_forward_calls": self.state["expected_calls"],
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def prepare_inputs(
    row: dict[str, Any],
    fractions: Sequence[float],
    processor: Any,
    prompt_text: str,
    device: torch.device,
    image_size: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    frames, metadata, positions = decode_random_seek(
        Path(row["video_path"]), fractions, image_size
    )
    inputs = processor(
        text=[prompt_text],
        videos=[frames],
        video_metadata=[metadata],
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )
    return {key: value.to(device) for key, value in inputs.items()}, positions
