from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
CONTRACT = "qwen3vl_lgvq_temporal_raw_to_scalar_random_seek_v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_path", "manifest_path", "output_dir"):
        value = Path(raw[key]).expanduser()
        raw[key] = value.resolve() if value.is_absolute() else (path.parent / value).resolve()
    counts = [int(value) for value in raw["frame_counts"]]
    if counts != [4, 9, 16]:
        raise ValueError("The formal audit must test frame counts [4, 9, 16]")
    if sorted({int(value) for order in raw["measurement_orders"] for value in order}) != counts:
        raise ValueError("Every measurement order must use only 4, 9, and 16")
    if any(sorted(map(int, order)) != counts for order in raw["measurement_orders"]):
        raise ValueError("Each trial must contain every frame count exactly once")
    for count in counts:
        fractions = [float(value) for value in raw["frame_sampling"][str(count)]]
        if len(fractions) != count or any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError(f"Invalid frame sampling contract for {count} frames")
    if int(raw["processor_pixels"]) != 448 * 448:
        raise ValueError("The audited project contract requires 448x448 Qwen inputs")
    return raw


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"sample_id", "video_path", "split"}.issubset(rows[0]):
        raise RuntimeError(f"Invalid LGVQ manifest: {path}")
    for row in rows:
        if row["split"] not in {"train", "test"}:
            raise RuntimeError(f"Unexpected split {row['split']!r}")
        video = Path(row["video_path"]).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        row["video_path"] = str(video)
    return rows


@dataclass(frozen=True)
class DecodedVideo:
    frames: list[Image.Image]
    metadata: Any
    frame_total: int
    frame_positions: list[int]


def decode_video_random_seek(
    path: Path,
    fractions: Sequence[float],
    crop_fraction: float,
    resize_wh: tuple[int, int],
) -> DecodedVideo:
    """Faithfully reproduce the three LGVQ projects' per-frame random seeking.

    This function intentionally does not use the faster sequential decoder,
    parallel workers, a decoded-frame cache, or a feature cache.  Every sampled
    position executes CAP_PROP_POS_FRAMES followed by read(), as the original
    experiment loaders do.
    """

    from transformers.video_utils import VideoMetadata

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 24.0
    positions = [
        min(total - 1, max(0, round((total - 1) * float(fraction))))
        for fraction in fractions
    ]
    frames: list[Image.Image] = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode frame {position} from {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            side = max(2, round(min(height, width) * crop_fraction))
            top, left = (height - side) // 2, (width - side) // 2
            square = cv2.resize(
                rgb[top : top + side, left : left + side],
                resize_wh,
                interpolation=cv2.INTER_AREA,
            )
            frames.append(Image.fromarray(square))
    finally:
        capture.release()
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        width=resize_wh[0],
        height=resize_wh[1],
        duration=float(total) / fps,
        frames_indices=positions,
    )
    return DecodedVideo(frames, metadata, total, positions)


def render_temporal_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class ScalarLinearReadout(nn.Module):
    """The baseline's only post-Qwen operation: Linear(2048, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2048, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value).squeeze(-1)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty measurement")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "p05": percentile(values, 5),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def synchronize() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def measure_one(
    *,
    row: dict[str, str],
    frame_count: int,
    fractions: Sequence[float],
    crop_fraction: float,
    resize_wh: tuple[int, int],
    processor: Any,
    prompt_text: str,
    model: nn.Module,
    readout: ScalarLinearReadout,
    device: torch.device,
) -> dict[str, Any]:
    synchronize()
    total_started = time.perf_counter()

    started = time.perf_counter()
    decoded = decode_video_random_seek(
        Path(row["video_path"]), fractions, crop_fraction, resize_wh
    )
    decode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    inputs = processor(
        text=[prompt_text],
        videos=[decoded.frames],
        video_metadata=[decoded.metadata],
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )
    processor_seconds = time.perf_counter() - started

    input_shapes = {
        key: list(value.shape)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    started = time.perf_counter()
    inputs = {key: value.to(device) for key, value in inputs.items()}
    synchronize()
    transfer_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            **inputs, return_dict=True, use_cache=False
        ).last_hidden_state
    synchronize()
    qwen_seconds = time.perf_counter() - started

    started = time.perf_counter()
    mask = inputs["attention_mask"].bool()
    indices = torch.arange(mask.shape[1], device=device).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(1)
    pooled = hidden[torch.arange(hidden.shape[0], device=device), last].float()
    scalar = readout(pooled)
    score = float(scalar.cpu().item())
    synchronize()
    readout_seconds = time.perf_counter() - started
    total_seconds = time.perf_counter() - total_started

    return {
        "sample_id": row["sample_id"],
        "frame_count": frame_count,
        "video_total_frames": decoded.frame_total,
        "selected_frame_positions": decoded.frame_positions,
        "unique_selected_frame_positions": len(set(decoded.frame_positions)),
        "decode_ms": 1000.0 * decode_seconds,
        "processor_ms": 1000.0 * processor_seconds,
        "host_to_device_ms": 1000.0 * transfer_seconds,
        "qwen_backbone_ms": 1000.0 * qwen_seconds,
        "pool_and_linear_ms": 1000.0 * readout_seconds,
        "total_raw_video_to_scalar_ms": 1000.0 * total_seconds,
        "sequence_length": int(inputs["input_ids"].shape[1]),
        "input_shapes": input_shapes,
        "ignored_timing_score": score,
    }


def write_measurements_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "trial",
        "order_position",
        "sample_index",
        "sample_id",
        "frame_count",
        "video_total_frames",
        "selected_frame_positions",
        "unique_selected_frame_positions",
        "decode_ms",
        "processor_ms",
        "host_to_device_ms",
        "qwen_backbone_ms",
        "pool_and_linear_ms",
        "total_raw_video_to_scalar_ms",
        "sequence_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in columns}
            row["selected_frame_positions"] = json.dumps(
                row["selected_frame_positions"], separators=(",", ":")
            )
            writer.writerow(row)


def build_summary(
    config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    model_load_seconds: float,
    model_total_parameters: int,
    model_trainable_parameters: int,
    head_parameters: int,
    attention_implementation: str,
    gpu_name: str,
    peak_memory_by_frame_count: dict[int, int],
) -> dict[str, Any]:
    component_keys = (
        "decode_ms",
        "processor_ms",
        "host_to_device_ms",
        "qwen_backbone_ms",
        "pool_and_linear_ms",
        "total_raw_video_to_scalar_ms",
    )
    by_count: dict[str, Any] = {}
    for frame_count in config["frame_counts"]:
        subset = [row for row in records if row["frame_count"] == frame_count]
        sequence_lengths = sorted({row["sequence_length"] for row in subset})
        by_count[str(frame_count)] = {
            "measured_video_evaluations": len(subset),
            "unique_videos": len({row["sample_id"] for row in subset}),
            "trials": len({row["trial"] for row in subset}),
            "frame_fractions": config["frame_sampling"][str(frame_count)],
            "sequence_lengths": sequence_lengths,
            "mean_source_video_frames": float(
                statistics.fmean(row["video_total_frames"] for row in subset)
            ),
            "source_video_frame_count_distribution": {
                str(value): sum(row["video_total_frames"] == value for row in subset)
                for value in sorted({row["video_total_frames"] for row in subset})
            },
            "mean_unique_selected_frame_positions": float(
                statistics.fmean(row["unique_selected_frame_positions"] for row in subset)
            ),
            "latency_ms_per_video": {
                key: summarize([float(row[key]) for row in subset])
                for key in component_keys
            },
            "videos_per_second_from_raw_video_to_scalar": 1000.0
            / statistics.fmean(
                float(row["total_raw_video_to_scalar_ms"]) for row in subset
            ),
            "peak_cuda_allocated_bytes": peak_memory_by_frame_count[frame_count],
            "peak_cuda_allocated_gib": peak_memory_by_frame_count[frame_count]
            / (1024**3),
            "example_selected_positions": subset[0]["selected_frame_positions"],
            "example_input_shapes": subset[0]["input_shapes"],
        }
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "temporal prompt only; one raw video produces one scalar",
        "timed_boundary": (
            "open MP4 -> repeated random seeks for all requested frames -> 65% center "
            "crop -> 448x448 resize -> official Qwen processor/tokenizer -> host-to-device "
            "copy -> complete frozen Qwen3-VL vision+merger+language backbone -> final-token "
            "pool -> Linear(2048,1) -> CPU scalar"
        ),
        "excluded_from_per_video_latency": [
            "Python/model loading",
            "one-time prompt chat-template rendering",
            "warmup evaluations",
        ],
        "deliberately_disabled_accelerations": [
            "sequential full-clip decoding",
            "parallel video decoding",
            "decoded-frame cache",
            "Qwen feature cache",
            "torch.compile",
            "quantization",
            "FlashAttention override",
        ],
        "decode_contract": (
            "one cv2.VideoCapture per evaluation; CAP_PROP_POS_FRAMES + read for every "
            "sampled position, matching the original LGVQ project loaders"
        ),
        "batch_size_videos": 1,
        "temporal_prompt": config["temporal_prompt"],
        "measurement_orders": config["measurement_orders"],
        "warmup_videos_per_frame_count": config["warmup_videos_per_frame_count"],
        "model_path": str(config["model_path"]),
        "model_load_seconds": model_load_seconds,
        "model_total_parameters": model_total_parameters,
        "model_trainable_parameters": model_trainable_parameters,
        "linear_readout_parameters": head_parameters,
        "model_dtype": "bfloat16",
        "attention_implementation": attention_implementation,
        "gpu": gpu_name,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "opencv": cv2.__version__,
        },
        "results": by_count,
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    columns = [
        "frames",
        "evaluations",
        "mean_total_ms_per_video",
        "median_total_ms_per_video",
        "p95_total_ms_per_video",
        "mean_decode_ms",
        "mean_processor_ms",
        "mean_host_to_device_ms",
        "mean_qwen_backbone_ms",
        "mean_pool_and_linear_ms",
        "videos_per_second",
        "peak_cuda_gib",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame_count, result in summary["results"].items():
            latency = result["latency_ms_per_video"]
            writer.writerow(
                {
                    "frames": frame_count,
                    "evaluations": result["measured_video_evaluations"],
                    "mean_total_ms_per_video": latency["total_raw_video_to_scalar_ms"]["mean"],
                    "median_total_ms_per_video": latency["total_raw_video_to_scalar_ms"]["median"],
                    "p95_total_ms_per_video": latency["total_raw_video_to_scalar_ms"]["p95"],
                    "mean_decode_ms": latency["decode_ms"]["mean"],
                    "mean_processor_ms": latency["processor_ms"]["mean"],
                    "mean_host_to_device_ms": latency["host_to_device_ms"]["mean"],
                    "mean_qwen_backbone_ms": latency["qwen_backbone_ms"]["mean"],
                    "mean_pool_and_linear_ms": latency["pool_and_linear_ms"]["mean"],
                    "videos_per_second": result["videos_per_second_from_raw_video_to_scalar"],
                    "peak_cuda_gib": result["peak_cuda_allocated_gib"],
                }
            )


def write_results_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-VL temporal frame-count timing result",
        "",
        "The reported total is for **one video and one Temporal prompt**. It starts",
        "before opening the MP4 and stops after the scalar prediction reaches CPU.",
        "Model loading and warmup are recorded separately and excluded.",
        "",
        "| Frames | Total mean (ms/video) | Total median | Total p95 | Decode | Processor | H2D | Qwen backbone | Pool+Linear | video/s | Peak GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frame_count in ("4", "9", "16"):
        result = summary["results"][frame_count]
        latency = result["latency_ms_per_video"]
        lines.append(
            "| {f} | {tm:.3f} | {td:.3f} | {tp:.3f} | {d:.3f} | {p:.3f} | "
            "{h:.3f} | {q:.3f} | {r:.3f} | {v:.3f} | {g:.3f} |".format(
                f=frame_count,
                tm=latency["total_raw_video_to_scalar_ms"]["mean"],
                td=latency["total_raw_video_to_scalar_ms"]["median"],
                tp=latency["total_raw_video_to_scalar_ms"]["p95"],
                d=latency["decode_ms"]["mean"],
                p=latency["processor_ms"]["mean"],
                h=latency["host_to_device_ms"]["mean"],
                q=latency["qwen_backbone_ms"]["mean"],
                r=latency["pool_and_linear_ms"]["mean"],
                v=result["videos_per_second_from_raw_video_to_scalar"],
                g=result["peak_cuda_allocated_gib"],
            )
        )
    lines.extend(
        [
            "",
            f"- GPU: `{summary['gpu']}`",
            f"- Model load time: `{summary['model_load_seconds']:.3f} s`",
            "- Batch size: `1 video`.",
            "- Decode: original repeated random-seek implementation; no sequential-decode optimization.",
            "- Qwen feature caching, frame caching, quantization and `torch.compile` are disabled.",
            "- The linear layer is present only to preserve the baseline output boundary; its numerical score is ignored in this timing audit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    config = load_config(config_path)
    rows = read_rows(Path(config["manifest_path"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    random.Random(int(config["seed"])).shuffle(train_rows)
    random.Random(int(config["seed"])).shuffle(test_rows)
    warmup_count = int(config["warmup_videos_per_frame_count"])
    if not bool(config.get("measure_all_test_videos", False)):
        raise RuntimeError("Formal timing requires every test video")
    if len(train_rows) < warmup_count or not test_rows:
        raise RuntimeError("The manifest does not contain enough train/test videos")
    warmup_rows = train_rows[:warmup_count]
    order_count = len(config["measurement_orders"])
    measured_partitions = [test_rows[index::order_count] for index in range(order_count)]

    device = torch.device("cuda")
    pixels = int(config["processor_pixels"])
    processor = AutoProcessor.from_pretrained(
        str(config["model_path"]),
        min_pixels=pixels,
        max_pixels=pixels,
        local_files_only=True,
        trust_remote_code=True,
    )
    load_started = time.perf_counter()
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(config["model_path"]),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    synchronize()
    model_load_seconds = time.perf_counter() - load_started
    torch.manual_seed(int(config["seed"]))
    readout = ScalarLinearReadout().to(device).eval().requires_grad_(False)
    prompt = render_temporal_prompt(processor, config["temporal_prompt"])
    resize_wh = tuple(map(int, config["resize_wh"]))
    crop_fraction = float(config["center_crop_short_side_fraction"])

    # Shape-specific warmup is explicit and excluded from every reported number.
    for frame_count in config["frame_counts"]:
        fractions = config["frame_sampling"][str(frame_count)]
        for row in warmup_rows:
            measure_one(
                row=row,
                frame_count=frame_count,
                fractions=fractions,
                crop_fraction=crop_fraction,
                resize_wh=resize_wh,
                processor=processor,
                prompt_text=prompt,
                model=model,
                readout=readout,
                device=device,
            )

    records: list[dict[str, Any]] = []
    peak_by_count = {int(count): 0 for count in config["frame_counts"]}
    for trial, (order, measured_rows) in enumerate(
        zip(config["measurement_orders"], measured_partitions), start=1
    ):
        for order_position, raw_count in enumerate(order, start=1):
            frame_count = int(raw_count)
            torch.cuda.reset_peak_memory_stats()
            for sample_index, row in enumerate(measured_rows, start=1):
                record = measure_one(
                    row=row,
                    frame_count=frame_count,
                    fractions=config["frame_sampling"][str(frame_count)],
                    crop_fraction=crop_fraction,
                    resize_wh=resize_wh,
                    processor=processor,
                    prompt_text=prompt,
                    model=model,
                    readout=readout,
                    device=device,
                )
                record.update(
                    {
                        "trial": trial,
                        "order_position": order_position,
                        "sample_index": sample_index,
                    }
                )
                records.append(record)
                print(
                    f"[trial {trial} position {order_position}] {frame_count}f "
                    f"{sample_index}/{len(measured_rows)} "
                    f"total={record['total_raw_video_to_scalar_ms']:.1f}ms",
                    flush=True,
                )
            peak_by_count[frame_count] = max(
                peak_by_count[frame_count], int(torch.cuda.max_memory_allocated())
            )

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    attention = str(getattr(model.config, "_attn_implementation", "unknown"))
    summary = build_summary(
        config,
        records,
        model_load_seconds=model_load_seconds,
        model_total_parameters=model_parameters,
        model_trainable_parameters=trainable_parameters,
        head_parameters=sum(parameter.numel() for parameter in readout.parameters()),
        attention_implementation=attention,
        gpu_name=torch.cuda.get_device_name(device),
        peak_memory_by_frame_count=peak_by_count,
    )
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_measurements_csv(output / "per_video_measurements.csv", records)
    write_json(output / "timing_summary.json", summary)
    write_summary_csv(output / "timing_summary.csv", summary)
    write_results_markdown(output / "RESULTS.md", summary)
    write_json(
        output / "run_identity.json",
        {
            "contract": CONTRACT,
            "config": str(config_path.expanduser().resolve()),
            "config_sha256": sha256_file(config_path.expanduser().resolve()),
            "manifest": str(config["manifest_path"]),
            "manifest_sha256": sha256_file(Path(config["manifest_path"])),
            "measured_sample_ids": [row["sample_id"] for row in test_rows],
            "warmup_sample_ids": [row["sample_id"] for row in warmup_rows],
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config.json"
    )
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
