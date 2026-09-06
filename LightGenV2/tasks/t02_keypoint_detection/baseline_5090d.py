"""Frozen Qwen Vision plus the historical lightweight LSP pose head on 5090 D."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

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
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    DatasetBundle,
    LSPPoseDataset,
    _read_source,
    split_standard_protocol,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import (
    build_teacher,
    load_vision_backbone,
    preprocess_vision,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.training import (
    build_loaders,
    evaluate_model,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


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
    settings = load_settings(args.config.expanduser().resolve())
    settings.data_root = args.data_root.expanduser().resolve()
    settings.model_id = str(args.model.expanduser().resolve())
    settings.cache_dir = None
    settings.local_files_only = True
    settings.download = False
    settings.output_dir = args.run_dir.expanduser().resolve()
    settings.inference_batch_size = int(args.performance_batch_size)
    settings.num_workers = int(args.num_workers)
    settings.visualization_sample_count = 0
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    lsp_root = settings.data_root / "lsp_dataset"
    lsp = _read_source(lsp_root, "lsp", settings.visibility_policy)
    if len(lsp) != 2000:
        raise RuntimeError(f"Expected 2000 official LSP samples, got {len(lsp)}")
    train, test = split_standard_protocol(lsp, [])
    bundle = DatasetBundle(
        train=train,
        test=test,
        metadata={
            "dataset": "LSP",
            "protocol": "official first-1000 train / last-1000 test",
            "test_samples": len(test),
        },
    )
    loaded = load_vision_backbone(settings, torch.device("cuda:0"))
    model = build_teacher(loaded, settings)
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.head.load_state_dict(payload["head"], strict=True)
    model.eval()
    _, test_loader = build_loaders(bundle, settings, training=False)
    performance, _ = evaluate_model(
        model,
        "teacher",
        test_loader,
        loaded.processor,
        loaded.device,
        settings,
        phase="qwen_5090d_baseline",
        epoch=int(payload.get("epoch", -1)),
        save_outputs=False,
    )

    dataset = LSPPoseDataset(test, settings, training=False)
    timer = FirstBlockTimer(loaded.visual.blocks[0])
    sampler = NvidiaSmiPowerSampler()
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)
    measurements: list[dict[str, Any]] = []
    try:
        for index in range(args.warmup_forwards):
            item = dataset[index % len(dataset)]
            inputs = preprocess_vision(loaded.processor, [item["image"]], loaded.device)
            timer.reset()
            heatmaps, _spatial = model(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            _result = heatmaps.argmax(dim=-1)
            timer.finish()
        for index in range(min(args.timing_samples, len(dataset))):
            item = dataset[index]
            inputs = preprocess_vision(loaded.processor, [item["image"]], loaded.device)
            torch.cuda.synchronize()
            timer.reset()
            sampler.set_phase(f"active:{index}")
            try:
                heatmaps, spatial = model(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
                _result = heatmaps.argmax(dim=-1)
                timing = timer.finish()
            finally:
                sampler.set_phase(None)
            measurements.append(
                {
                    "sample_index": index,
                    "sample_id": item["sample_id"],
                    "cuda_ms": timing["cuda_ms"],
                    "host_ms": timing["host_ms"],
                    "first_block_input_shape": json.dumps(
                        timing["first_block_input_shape"]
                    ),
                    "spatial_shape": json.dumps(list(spatial.shape)),
                    "output_shape": json.dumps(list(heatmaps.shape)),
                }
            )
    finally:
        timer.close()
        power_samples = sampler.stop()
        model.close()

    latencies = [row["cuda_ms"] for row in measurements]
    trainable = sum(parameter.numel() for parameter in model.head.parameters())
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "LSP keypoint detection",
        "model": settings.model_id,
        "qwen_frozen": True,
        "readout": "LightweightPoseHead -> 14x56x56 heatmaps",
        "readout_trainable_parameters": trainable,
        "test_samples": len(test),
        "timing_samples": len(measurements),
        "explicit_warmup_forwards": args.warmup_forwards,
        "first_test_sample_included": False,
        "timing_boundary": (
            "input to native Vision Transformer block 0 through all native Vision "
            "blocks and the trained lightweight pose head to 14 heatmaps"
        ),
        "performance": performance,
        "latency_cuda_ms": summarize(latencies),
        "latency_host_ms": summarize([row["host_ms"] for row in measurements]),
        "power": power_report(power_samples, latencies),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "environment": environment_report(),
        "model_load_seconds": loaded.load_time_sec,
    }
    write_json(settings.output_dir / "baseline_report.json", report)
    _write_rows(settings.output_dir / "timing_per_sample.csv", measurements)
    save_power_samples(settings.output_dir / "power_samples.csv", power_samples)
    (settings.output_dir / "command.txt").write_text(
        " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timing-samples", type=int, default=200)
    parser.add_argument("--warmup-forwards", type=int, default=50)
    parser.add_argument("--performance-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
