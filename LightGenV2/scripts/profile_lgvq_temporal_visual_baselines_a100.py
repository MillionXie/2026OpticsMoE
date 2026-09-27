"""A100 timing/power demo for frozen CLIP and YOLO11s LGVQ baselines.

The first 200 test videos are evaluated in manifest order.  Four deterministic
frames are decoded per video.  Decode, resize, normalization and host-to-device
copy are performed before the timed region.  The synchronized-wall and CUDA
boundaries cover the native backbone (from its first block) and the five-row
temporal-quality readout.  There is no explicit warm-up and call one is kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


FRAME_FRACTIONS = (0.10, 0.37, 0.63, 0.90)
FRAME_COUNT = 4
HOST_POWER_W = 338.2
PERFORMANCE_SRCC = {
    "clip_vit_b32": 0.7251568,
    "deepseek_vl2_tiny": 0.5688169,
    "yolo11s": 0.7015822,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(ordered.mean()),
        "median": float(np.median(ordered)),
        "std": float(ordered.std()),
        "p05": float(np.percentile(ordered, 5)),
        "p95": float(np.percentile(ordered, 95)),
        "minimum": float(ordered.min()),
        "maximum": float(ordered.max()),
    }


def read_test_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("split") == "test"]
    if len(rows) < 200:
        raise RuntimeError(f"Need at least 200 test rows, found {len(rows)}")
    return rows[:200]


def video_path(row: dict[str, str]) -> Path:
    for key in ("video_path", "path", "file_path"):
        if row.get(key):
            return Path(row[key])
    raise KeyError(f"No video path column in {sorted(row)}")


def decode_frames(path: Path, image_size: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Unreadable video: {path}")
    frames: list[Image.Image] = []
    for fraction in FRAME_FRACTIONS:
        position = min(total - 1, max(0, round((total - 1) * fraction)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {position}: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        side = max(2, round(min(height, width) * 0.65))
        top, left = (height - side) // 2, (width - side) // 2
        rgb = cv2.resize(
            rgb[top : top + side, left : left + side],
            (image_size, image_size), interpolation=cv2.INTER_AREA,
        )
        frames.append(Image.fromarray(rgb))
    capture.release()
    return frames


class Sampler:
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.active = False
        self.stop_event = threading.Event()
        self.rows: list[dict[str, float]] = []
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 "--query-gpu=power.draw,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, capture_output=True,
            )
            if result.returncode == 0 and self.active:
                values = [float(item.strip()) for item in result.stdout.strip().split(",")]
                if len(values) == 3:
                    self.rows.append({
                        "host_time_unix": time.time(), "power_w": values[0],
                        "utilization_percent": values[1], "memory_used_mib": values[2],
                    })
            self.stop_event.wait(0.05)

    def stop(self) -> list[dict[str, float]]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        return self.rows


def process_audit(gpu_index: int) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}",
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


class BoundaryTimer:
    def __init__(self, first_block: nn.Module) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)
        self.wall_start = 0.0
        self.handle = first_block.register_forward_pre_hook(self._start)

    def _start(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        torch.cuda.synchronize()
        self.wall_start = time.perf_counter()
        self.start_event.record()

    def finish(self) -> tuple[float, float]:
        self.stop_event.record()
        torch.cuda.synchronize()
        return (
            float(self.start_event.elapsed_time(self.stop_event)),
            1000.0 * (time.perf_counter() - self.wall_start),
        )

    def close(self) -> None:
        self.handle.remove()


def load_clip(model_path: Path, device: torch.device):
    import clip

    model, preprocess = clip.load(str(model_path), device=device, jit=False)
    model.eval().requires_grad_(False)
    head = nn.Linear(2048, 5, bias=False, device=device).eval().requires_grad_(False)
    first_block = model.visual.transformer.resblocks[0]

    def prepare(frames: list[Image.Image]) -> torch.Tensor:
        return torch.stack([preprocess(frame) for frame in frames])

    def forward(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            feature = model.encode_image(batch).float()
        feature = nn.functional.normalize(feature, dim=-1).reshape(-1, 2048)
        return head(feature)

    return model, head, first_block, prepare, forward, 224


def load_yolo(model_path: Path, device: torch.device):
    from ultralytics import YOLO

    model = YOLO(str(model_path)).model.to(device).eval().requires_grad_(False)
    capture: dict[str, torch.Tensor] = {}
    feature_layer = model.model[10]
    feature_handle = feature_layer.register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__("feature", output)
    )
    head = nn.Linear(2048, 5, bias=False, device=device).eval().requires_grad_(False)

    def prepare(frames: list[Image.Image]) -> torch.Tensor:
        arrays = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])
        return torch.from_numpy(arrays.copy()).permute(0, 3, 1, 2).float().div_(255.0)

    def forward(batch: torch.Tensor) -> torch.Tensor:
        capture.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            model(batch)
        feature = capture["feature"].float().mean(dim=(-2, -1))
        feature = nn.functional.normalize(feature, dim=-1).reshape(-1, 2048)
        return head(feature)

    return model, head, model.model[0], prepare, forward, 640, feature_handle


def load_deepseek(model_path: Path, device: torch.device):
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor

    processor = DeepseekVLV2Processor.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval().requires_grad_(False)
    hidden_size = int(model.config.language_config.hidden_size)
    head = nn.Linear(hidden_size, 5, bias=False, device=device).eval().requires_grad_(False)
    prompt = (
        "Please evaluate the temporal quality of this video and rate it using one of "
        "the following five levels: Excellent, Good, Fair, Poor, or Bad."
    )

    def prepare(frames: list[Image.Image]):
        samples = []
        for start in range(0, len(frames), FRAME_COUNT):
            selected = frames[start : start + FRAME_COUNT]
            content = "\n".join(f"Frame {index + 1}: <image>" for index in range(FRAME_COUNT))
            conversation = [
                {"role": "<|User|>", "content": content + "\n" + prompt, "images": selected},
                {"role": "<|Assistant|>", "content": ""},
            ]
            samples.append(processor.process_one(conversations=conversation, images=selected))
        return processor.batchify(samples)

    def forward(batch):
        inputs_embeds = model.prepare_inputs_embeds(**batch)
        outputs = model.language.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        lengths = batch.attention_mask.long().sum(dim=-1).sub(1)
        hidden = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0], device=device), lengths
        ].float()
        return head(hidden)

    return model, head, model.vision.blocks[0], prepare, forward, 384


def run_batch_size(
    *, rows: list[dict[str, str]], batch_size: int, prepare, forward,
    image_size: int, first_block: nn.Module, device: torch.device,
    gpu_index: int, output: Path,
) -> dict[str, Any]:
    cpu_batches: list[torch.Tensor] = []
    sample_ids: list[list[str]] = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        if len(selected) != batch_size:
            break
        frames: list[Image.Image] = []
        for row in selected:
            frames.extend(decode_frames(video_path(row), image_size))
        prepared = prepare(frames)
        cpu_batches.append(prepared.contiguous() if isinstance(prepared, torch.Tensor) else prepared)
        sample_ids.append([row.get("sample_id", str(start + i)) for i, row in enumerate(selected)])

    timer = BoundaryTimer(first_block)
    timing_rows: list[dict[str, Any]] = []
    gpu_batches: list[torch.Tensor] = []
    try:
        for index, (cpu_batch, ids) in enumerate(zip(cpu_batches, sample_ids)):
            gpu_batch = cpu_batch.to(device)
            gpu_batches.append(gpu_batch)
            cuda_ms, wall_ms = timer.finish() if False else (0.0, 0.0)
            output_tensor = forward(gpu_batch)
            cuda_ms, wall_ms = timer.finish()
            timing_rows.append({
                "call_index": index, "batch_size_videos": batch_size,
                "sample_ids": json.dumps(ids, ensure_ascii=False),
                "cuda_event_ms": cuda_ms, "synchronized_wall_ms": wall_ms,
                "output_shape": json.dumps(list(output_tensor.shape)),
            })
    finally:
        timer.close()

    sampler = Sampler(gpu_index)
    sampler.start()
    power_started = time.perf_counter()
    sampler.active = True
    index = 0
    try:
        while time.perf_counter() - power_started < 15.0:
            with torch.inference_mode():
                forward(gpu_batches[index % len(gpu_batches)])
            index += 1
        torch.cuda.synchronize()
    finally:
        sampler.active = False
        power_rows = sampler.stop()
    if not power_rows:
        raise RuntimeError("No active power samples")

    cuda = summary([float(row["cuda_event_ms"]) for row in timing_rows])
    wall = summary([float(row["synchronized_wall_ms"]) for row in timing_rows])
    power = summary([float(row["power_w"]) for row in power_rows])
    calls_for_16 = 16 // batch_size
    time_16_ms = wall["mean"] * calls_for_16
    board_energy_16_j = power["mean"] * time_16_ms / 1000.0
    chassis_board_energy_16_j = (HOST_POWER_W + power["mean"]) * time_16_ms / 1000.0
    output.mkdir(parents=True, exist_ok=True)
    with (output / "timing_per_call.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader(); writer.writerows(timing_rows)
    with (output / "power_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(power_rows[0]))
        writer.writeheader(); writer.writerows(power_rows)
    return {
        "batch_size_videos": batch_size,
        "measured_videos": len(timing_rows) * batch_size,
        "measured_calls": len(timing_rows),
        "explicit_warmup_calls": 0,
        "first_call_included": True,
        "cuda_event_ms_per_batch": cuda,
        "synchronized_wall_ms_per_batch": wall,
        "active_a100_board_power_w": power,
        "active_power_samples": len(power_rows),
        "calls_for_16_videos": calls_for_16,
        "equivalent_16_video_time_ms": time_16_ms,
        "equivalent_16_video_board_energy_j": board_energy_16_j,
        "equivalent_16_video_host_plus_board_energy_j": chassis_board_energy_16_j,
        "videos_per_j_host_plus_board": 16.0 / chassis_board_energy_16_j,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=tuple(PERFORMANCE_SRCC), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=6)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if process_audit(args.gpu_index):
        raise RuntimeError(f"A100 has foreign compute processes: {process_audit(args.gpu_index)}")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"A100 required, got {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    device = torch.device("cuda:0")
    rows = read_test_rows(args.manifest.resolve())
    extra_handle = None
    if args.backbone == "clip_vit_b32":
        model, head, first_block, prepare, forward, image_size = load_clip(args.model, device)
    elif args.backbone == "yolo11s":
        model, head, first_block, prepare, forward, image_size, extra_handle = load_yolo(args.model, device)
    else:
        model, head, first_block, prepare, forward, image_size = load_deepseek(args.model, device)
    results = []
    try:
        for batch_size in args.batch_sizes:
            print(f"[{args.backbone}] batch={batch_size}", flush=True)
            results.append(run_batch_size(
                rows=rows, batch_size=batch_size, prepare=prepare, forward=forward,
                image_size=image_size, first_block=first_block, device=device,
                gpu_index=args.gpu_index, output=args.output / f"batch_{batch_size:02d}",
            ))
    finally:
        if extra_handle is not None:
            extra_handle.remove()
    report = {
        "schema_version": 1,
        "backbone": args.backbone,
        "performance_temporal_srcc": PERFORMANCE_SRCC[args.backbone],
        "performance_source": "fixed LGVQ 2250/558 split, frozen backbone, best test-SRCC checkpoint",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model.resolve()) if args.model.is_file() else None,
        "gpu": torch.cuda.get_device_name(0),
        "timing_boundary": "first native backbone block to five-row temporal-quality score ready on GPU",
        "excluded": ["decode", "crop/resize", "normalization", "host-to-device transfer", "model load"],
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
