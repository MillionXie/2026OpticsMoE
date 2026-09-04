from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

try:
    from . import benchmark as base
    from . import benchmark_extended as extended
except ImportError:
    import benchmark as base
    import benchmark_extended as extended


PROJECT_ROOT = Path(__file__).resolve().parent
CONTRACT = "qwen3vl_lgvq_temporal_explicit_25_49_random_seek_v1"


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    additional = json.loads(path.read_text(encoding="utf-8"))
    base_path = Path(additional["base_config"]).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    standard = base.load_config(base_path)
    if [int(value) for value in additional["frame_counts"]] != [25, 49]:
        raise ValueError("This explicit follow-up contract must test 25 and 49 frames")
    if not bool(additional.get("measure_all_test_videos", False)):
        raise ValueError("The formal follow-up must measure all test videos")
    return additional, standard


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-VL Temporal 4/9/16/25/36/49-frame timing report",
        "",
        "Every latency is for one complete video and one Temporal prompt, not one frame.",
        "N is the number of unique fixed-split LGVQ test videos.",
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
    lines.extend(
        [
            "",
            "The earlier conditional rule skipped 49 frames because the 36-frame mean exceeded",
            "910 ms/video. This follow-up measures 49 frames because the user explicitly requested it.",
            "",
            "The benchmark retains batch size 1 and the original per-position random seek/read.",
            "No sequential decode, parallel decode, frame/feature cache, quantization,",
            "torch.compile, or manual FlashAttention override is used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    additional, standard = load_config(config_path)
    rows = base.read_rows(Path(standard["manifest_path"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    random.Random(int(additional["seed"])).shuffle(train_rows)
    random.Random(int(additional["seed"])).shuffle(test_rows)
    if len(test_rows) != 558:
        raise RuntimeError(f"Expected all 558 test videos, got {len(test_rows)}")
    warmup_rows = train_rows[: int(additional["warmup_videos_per_frame_count"])]

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
    torch.manual_seed(int(additional["seed"]))
    readout = base.ScalarLinearReadout().to(device).eval().requires_grad_(False)
    prompt = base.render_temporal_prompt(processor, standard["temporal_prompt"])
    output_dir = Path(standard["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start = float(additional["sampling_start_fraction"])
    stop = float(additional["sampling_end_fraction"])
    added_results: dict[str, Any] = {}
    for count in map(int, additional["frame_counts"]):
        summary, _, _ = extended.run_count(
            frame_count=count,
            rows=test_rows,
            warmup_rows=warmup_rows,
            fractions=extended.central_fractions(count, start, stop),
            standard=standard,
            processor=processor,
            prompt=prompt,
            model=model,
            readout=readout,
            device=device,
            output_dir=output_dir,
        )
        added_results.update(summary["results"])

    prior_path = output_dir / "timing_summary_extended.json"
    if not prior_path.is_file():
        raise FileNotFoundError("The audited 4/9/16/36 summary is required")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    combined = {
        "schema_version": 1,
        "contract": CONTRACT,
        "scope": "one raw video, one Temporal prompt, one scalar",
        "model_path": str(standard["model_path"]),
        "model_load_seconds_this_followup": model_load_seconds,
        "model_total_parameters": sum(p.numel() for p in model.parameters()),
        "model_trainable_parameters": 0,
        "linear_readout_parameters": sum(p.numel() for p in readout.parameters()),
        "batch_size_videos": 1,
        "results": {**prior["results"], **added_results},
        "historical_conditional_49_decision": prior["conditional_49_decision"],
        "explicit_followup": "The user subsequently requested both 25 and 49 frames",
        "sample_count_definition": (
            "N=558 unique fixed-split LGVQ test videos, measured once per frame count"
        ),
        "p95_definition": (
            "95th percentile across the 558 videos; 95% finish at or below this latency"
        ),
        "decode_contract": prior["decode_contract"],
        "deliberately_disabled_accelerations": prior[
            "deliberately_disabled_accelerations"
        ],
    }
    base.write_json(output_dir / "timing_summary_all.json", combined)
    extended.write_combined_csv(output_dir / "timing_summary_all.csv", combined)
    write_markdown(output_dir / "RESULTS_ALL.md", combined)
    base.write_json(
        output_dir / "additional_run_identity.json",
        {
            "contract": CONTRACT,
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
        "--config", type=Path, default=PROJECT_ROOT / "additional_config.json"
    )
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
