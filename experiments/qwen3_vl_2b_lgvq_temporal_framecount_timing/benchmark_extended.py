from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

try:
    from . import benchmark as base
except ImportError:
    import benchmark as base


PROJECT_ROOT = Path(__file__).resolve().parent
EXTENDED_CONTRACT = "qwen3vl_lgvq_temporal_36_then_conditional_49_random_seek_v1"


def load_extended_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    extended = json.loads(path.read_text(encoding="utf-8"))
    base_path = Path(extended["base_config"]).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    standard = base.load_config(base_path)
    if int(extended["primary_frame_count"]) != 36:
        raise ValueError("The primary extended timing contract must test 36 frames")
    if int(extended["conditional_frame_count"]) != 49:
        raise ValueError("The conditional timing contract must test 49 frames")
    if float(extended["conditional_threshold_mean_total_ms"]) != 910.0:
        raise ValueError("The user-specified conditional threshold is exactly 910 ms")
    if not bool(extended.get("measure_all_test_videos", False)):
        raise ValueError("The formal extension must measure all test videos")
    return extended, standard


def central_fractions(count: int, start: float, stop: float) -> list[float]:
    if count < 2 or not 0.0 <= start < stop <= 1.0:
        raise ValueError("Invalid central-time sampling contract")
    return [start + index * (stop - start) / (count - 1) for index in range(count)]


def write_count_csv(path: Path, records: list[dict[str, Any]]) -> None:
    base.write_measurements_csv(path, records)


def run_count(
    *,
    frame_count: int,
    rows: list[dict[str, str]],
    warmup_rows: list[dict[str, str]],
    fractions: list[float],
    standard: dict[str, Any],
    processor: Any,
    prompt: str,
    model: torch.nn.Module,
    readout: base.ScalarLinearReadout,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    resize_wh = tuple(map(int, standard["resize_wh"]))
    crop_fraction = float(standard["center_crop_short_side_fraction"])
    for row in warmup_rows:
        base.measure_one(
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

    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows, start=1):
        record = base.measure_one(
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
        record.update(
            {"trial": 1, "order_position": 1, "sample_index": sample_index}
        )
        records.append(record)
        print(
            f"[{frame_count}f] {sample_index}/{len(rows)} "
            f"total={record['total_raw_video_to_scalar_ms']:.1f}ms",
            flush=True,
        )
    peak = int(torch.cuda.max_memory_allocated())
    write_count_csv(output_dir / f"per_video_measurements_{frame_count}.csv", records)
    config_for_summary = {
        "frame_counts": [frame_count],
        "frame_sampling": {str(frame_count): fractions},
        "measurement_orders": [[frame_count]],
        "warmup_videos_per_frame_count": len(warmup_rows),
        "model_path": standard["model_path"],
        "temporal_prompt": standard["temporal_prompt"],
    }
    summary = base.build_summary(
        config_for_summary,
        records,
        model_load_seconds=0.0,
        model_total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        model_trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        head_parameters=sum(parameter.numel() for parameter in readout.parameters()),
        attention_implementation=str(
            getattr(model.config, "_attn_implementation", "unknown")
        ),
        gpu_name=torch.cuda.get_device_name(device),
        peak_memory_by_frame_count={frame_count: peak},
    )
    base.write_json(output_dir / f"timing_summary_{frame_count}.json", summary)
    return summary, records, peak


def write_combined_csv(path: Path, summary: dict[str, Any]) -> None:
    columns = [
        "frames",
        "sample_videos",
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
        "mean_unique_sampled_positions",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame_count in sorted(map(int, summary["results"])):
            result = summary["results"][str(frame_count)]
            latency = result["latency_ms_per_video"]
            writer.writerow(
                {
                    "frames": frame_count,
                    "sample_videos": result["measured_video_evaluations"],
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
                    "mean_unique_sampled_positions": result["mean_unique_selected_frame_positions"],
                }
            )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-VL Temporal 4/9/16/36/49-frame timing report",
        "",
        "## Main result",
        "",
        "Every reported latency is for one complete video and one Temporal prompt,",
        "not for one frame. N is the number of unique LGVQ test videos.",
        "",
        "| Frames | N | Mean total ms/video | Median | P95 | Decode | Processor | H2D | Qwen | Linear | video/s | Peak GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in sorted(map(int, summary["results"])):
        result = summary["results"][str(count)]
        latency = result["latency_ms_per_video"]
        lines.append(
            "| {count} | {n} | {mean:.3f} | {median:.3f} | {p95:.3f} | "
            "{decode:.3f} | {processor:.3f} | {h2d:.3f} | {qwen:.3f} | "
            "{linear:.3f} | {throughput:.3f} | {memory:.3f} |".format(
                count=count,
                n=result["measured_video_evaluations"],
                mean=latency["total_raw_video_to_scalar_ms"]["mean"],
                median=latency["total_raw_video_to_scalar_ms"]["median"],
                p95=latency["total_raw_video_to_scalar_ms"]["p95"],
                decode=latency["decode_ms"]["mean"],
                processor=latency["processor_ms"]["mean"],
                h2d=latency["host_to_device_ms"]["mean"],
                qwen=latency["qwen_backbone_ms"]["mean"],
                linear=latency["pool_and_linear_ms"]["mean"],
                throughput=result["videos_per_second_from_raw_video_to_scalar"],
                memory=result["peak_cuda_allocated_gib"],
            )
        )
    decision = summary["conditional_49_decision"]
    lines.extend(
        [
            "",
            "## Conditional 49-frame rule",
            "",
            f"The predeclared threshold was mean total latency <= {decision['threshold_mean_total_ms']:.1f} ms/video.",
            f"The measured 36-frame mean was {decision['observed_36_mean_total_ms']:.3f} ms/video.",
            f"49-frame status: **{decision['status']}**.",
            "",
            "## Statistical definitions",
            "",
            "- `N`: unique test videos measured once at that frame count.",
            "- `Mean`: arithmetic mean across all N videos; it is the threshold metric.",
            "- `Median`: the middle latency; half the videos are faster and half slower.",
            "- `P95`: the 95th percentile; 95% of videos finish no slower than this value, while the slowest 5% exceed it.",
            "- `Decode`: MP4 open plus one random seek/read for every requested frame, including crop and resize.",
            "- `Total`: Decode + processor/tokenizer + H2D + full frozen Qwen + pool/linear + small Python glue overhead.",
            "",
            "The benchmark uses batch size 1, no sequential decode, no parallel decode,",
            "no frame/feature cache, no quantization, and no torch.compile.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    extended, standard = load_extended_config(config_path)
    rows = base.read_rows(Path(standard["manifest_path"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    random.Random(int(extended["seed"])).shuffle(train_rows)
    random.Random(int(extended["seed"])).shuffle(test_rows)
    warmup_count = int(extended["warmup_videos_per_frame_count"])
    warmup_rows = train_rows[:warmup_count]
    if len(test_rows) != 558:
        raise RuntimeError(f"Expected all 558 test videos, got {len(test_rows)}")

    device = torch.device("cuda")
    pixels = int(standard["processor_pixels"])
    processor = AutoProcessor.from_pretrained(
        str(standard["model_path"]),
        min_pixels=pixels,
        max_pixels=pixels,
        local_files_only=True,
        trust_remote_code=True,
    )
    load_started = time.perf_counter()
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(standard["model_path"]),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    base.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    torch.manual_seed(int(extended["seed"]))
    readout = base.ScalarLinearReadout().to(device).eval().requires_grad_(False)
    prompt = base.render_temporal_prompt(processor, standard["temporal_prompt"])
    output_dir = Path(standard["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start = float(extended["sampling_start_fraction"])
    stop = float(extended["sampling_end_fraction"])
    primary = int(extended["primary_frame_count"])
    primary_fractions = central_fractions(primary, start, stop)
    summary36, _, _ = run_count(
        frame_count=primary,
        rows=test_rows,
        warmup_rows=warmup_rows,
        fractions=primary_fractions,
        standard=standard,
        processor=processor,
        prompt=prompt,
        model=model,
        readout=readout,
        device=device,
        output_dir=output_dir,
    )
    mean36 = summary36["results"][str(primary)]["latency_ms_per_video"][
        "total_raw_video_to_scalar_ms"
    ]["mean"]
    threshold = float(extended["conditional_threshold_mean_total_ms"])
    extra_results = dict(summary36["results"])
    conditional = int(extended["conditional_frame_count"])
    if mean36 <= threshold:
        conditional_fractions = central_fractions(conditional, start, stop)
        summary49, _, _ = run_count(
            frame_count=conditional,
            rows=test_rows,
            warmup_rows=warmup_rows,
            fractions=conditional_fractions,
            standard=standard,
            processor=processor,
            prompt=prompt,
            model=model,
            readout=readout,
            device=device,
            output_dir=output_dir,
        )
        extra_results.update(summary49["results"])
        status = "measured because the 36-frame mean did not exceed 910 ms"
    else:
        status = "skipped because the 36-frame mean exceeded 910 ms"

    existing_path = output_dir / "timing_summary.json"
    if not existing_path.is_file():
        raise FileNotFoundError(
            "The audited 4/9/16-frame timing_summary.json is required for combination"
        )
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    combined = {
        "schema_version": 1,
        "contract": EXTENDED_CONTRACT,
        "scope": "one raw video, one Temporal prompt, one scalar",
        "model_path": str(standard["model_path"]),
        "model_load_seconds_this_extension": model_load_seconds,
        "model_total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_trainable_parameters": 0,
        "linear_readout_parameters": sum(parameter.numel() for parameter in readout.parameters()),
        "batch_size_videos": 1,
        "results": {**existing["results"], **extra_results},
        "conditional_49_decision": {
            "threshold_metric": "mean total raw-video-to-scalar latency",
            "threshold_mean_total_ms": threshold,
            "observed_36_mean_total_ms": mean36,
            "condition": "measure 49 frames only when observed_36_mean_total_ms <= threshold",
            "status": status,
        },
        "sample_count_definition": (
            "N=558 unique fixed-split LGVQ test videos, measured once per executed frame count"
        ),
        "p95_definition": (
            "95th percentile across the 558 videos; 95% finish at or below this latency"
        ),
        "decode_contract": existing["decode_contract"],
        "deliberately_disabled_accelerations": existing[
            "deliberately_disabled_accelerations"
        ],
    }
    base.write_json(output_dir / "timing_summary_extended.json", combined)
    write_combined_csv(output_dir / "timing_summary_extended.csv", combined)
    write_markdown(output_dir / "RESULTS_EXTENDED.md", combined)
    base.write_json(
        output_dir / "extended_run_identity.json",
        {
            "contract": EXTENDED_CONTRACT,
            "config": str(config_path.expanduser().resolve()),
            "config_sha256": base.sha256_file(config_path.expanduser().resolve()),
            "base_config_sha256": base.sha256_file(PROJECT_ROOT / "config.json"),
            "manifest_sha256": base.sha256_file(Path(standard["manifest_path"])),
            "test_sample_ids": [row["sample_id"] for row in test_rows],
            "warmup_sample_ids": [row["sample_id"] for row in warmup_rows],
        },
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False), flush=True)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "extended_config.json"
    )
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

