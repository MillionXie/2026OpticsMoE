"""Benchmark the matched non-optical OpenMoji control on one visible GPU."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from LightGenV2.common.baseline_measurement import (
    NvidiaSmiPowerSampler,
    environment_report,
    power_report,
    save_power_samples,
    sha256_file,
    summarize,
    validate_cuda_device,
    write_json,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
    OpenMojiEditingDataset,
    collate_samples,
    load_prompt_cache,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.modeling import (
    build_model,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.settings import (
    load_settings,
)


def _load_selected_model(
    config: Path, checkpoint: Path | None, device: torch.device
) -> tuple[Any, torch.nn.Module, Path, dict[str, Any]]:
    settings = load_settings(config)
    if settings.optical_enabled:
        raise RuntimeError("Matched electronic benchmark requires model.optical_enabled=false")
    selected = (
        checkpoint.expanduser().resolve()
        if checkpoint is not None
        else settings.output_dir / "checkpoints" / "best_test_changed.pt"
    )
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    model = build_model(settings, device)
    state = model.state_dict()
    for name, value in payload.get("ema_model", payload["model"]).items():
        if name in state:
            state[name] = value.to(dtype=state[name].dtype)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return settings, model, selected, payload


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    gpu = validate_cuda_device(args.expected_gpu)
    device = torch.device("cuda:0")
    settings, model, checkpoint, payload = _load_selected_model(
        args.config, args.checkpoint, device
    )
    prompts = load_prompt_cache(settings.prompt_cache_path)
    dataset = OpenMojiEditingDataset(settings.test_manifest, settings, prompts)
    sample_count = min(int(args.samples), len(dataset))
    if sample_count <= 0:
        raise ValueError("--samples must be positive")

    sampler = NvidiaSmiPowerSampler()
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)
    records: list[dict[str, float | int]] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    try:
        for index in range(sample_count):
            # Dataset decode and host-to-device transfer are outside the model
            # boundary, matching the formal Qwen baseline convention.
            batch = collate_samples([dataset[index]])
            image = batch["source_image"].to(device)
            prompt = [value.to(device) for value in batch["prompt_hidden"]]
            torch.cuda.synchronize()
            sampler.set_phase(f"active:{index}")
            started = time.perf_counter_ns()
            start_event.record()
            try:
                output = model(image, prompt)
                _ = output["category_logits"].argmax(1)
                _ = output["edit_logits"].sigmoid().ge(0.5)
                end_event.record()
                end_event.synchronize()
                host_ms = (time.perf_counter_ns() - started) / 1.0e6
                cuda_ms = float(start_event.elapsed_time(end_event))
            finally:
                sampler.set_phase(None)
            records.append(
                {"sample_index": index, "cuda_ms": cuda_ms, "host_ms": host_ms}
            )
    finally:
        power_samples = sampler.stop()

    host = [float(row["host_ms"]) for row in records]
    cuda = [float(row["cuda_ms"]) for row in records]
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "OpenMoji semantic interaction",
        "baseline_type": "matched non-optical four-block electronic control",
        "not_full_qwen3vl_baseline": True,
        "timing_boundary": (
            "normalized source tensor plus precomputed frozen-Qwen instruction hidden "
            "through Qwen patch/position stem, four electronic task blocks, and 6x6 readout"
        ),
        "excluded_from_timing": (
            "image file I/O, prompt-cache lookup, Qwen instruction-hidden precomputation, "
            "and host-to-device transfer"
        ),
        "test_samples": sample_count,
        "explicit_warmup_forwards": 0,
        "first_test_sample_included": True,
        "selected_epoch": int(payload["epoch"]),
        "latency_cuda_ms": summarize(cuda),
        "latency_host_ms": summarize(host),
        "power": power_report(power_samples, host),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(settings.test_manifest),
        "manifest_sha256": sha256_file(settings.test_manifest),
        "environment": environment_report(),
        "gpu": gpu,
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "benchmark_report.json", report)
    with (output / "timing_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_power_samples(output / "power_samples.csv", power_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--expected-gpu", default="A100")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
