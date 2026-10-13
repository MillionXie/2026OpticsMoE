"""Frozen LGVQ batch evaluation; audit preprocessing is NOT a cost component.

One fresh process per model/batch, full 558-video metrics and first-200-video
costs. CPU preparation and inference are interleaved, as in the nine-task B1
measurement. No warmup, training, cache fitting or replacement of old records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

HOST_W = 338.2
MANIFEST_SHA = "607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc"
A100_UUID = "GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55"
CHECKPOINT_SHA = {
    "qwen": "e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2",
    "clip": "f364f6458eff4ba57015564ba1187eb0c5d59371140190ec825dcaddb2f89089a",
    "yolo": "8ac8f0965f6ad011392dd3e1e98e14f8ddb3d49ffbb3dfeb108059d30ee47c51",
    "deepseek": "6540addb760cc8bf61ff3f83af75d162fa2dd2ebbf29f938a0762767d2b0aef6",
}
BACKBONE_SHA = {
    "clip": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "yolo": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "deepseek": "cc1e5047280253e224b299677bea16b7960c2436ec676d28911ebd9de3bb0074",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def system_cost(mean_batch_wall_ms: float, active_w: float, batch: int) -> dict[str, float]:
    """Only inference wall time enters final latency AND energy."""
    if batch not in (1, 2, 4, 8) or mean_batch_wall_ms <= 0 or active_w <= 0:
        raise ValueError("Invalid measured batch cost")
    time16 = mean_batch_wall_ms * (16 // batch)
    return {
        "latency_16_videos_ms": time16,
        "latency_per_video_ms": time16 / 16,
        "energy_16_videos_j": (HOST_W + active_w) * time16 / 1000,
        "energy_per_video_j": (HOST_W + active_w) * time16 / 16000,
        "throughput_videos_per_second": 16000 / time16,
        "host_proxy_w": HOST_W,
        "measured_active_board_w": active_w,
    }


def backbone_file(baseline: str, model: Path) -> Path:
    if baseline != "deepseek":
        return model
    direct = model / "model.safetensors"
    if direct.is_file():
        return direct
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    filenames = set(index["weight_map"].values())
    if len(filenames) != 1:
        raise ValueError("Expected the original single-shard DeepSeek backbone")
    filename = filenames.pop()
    if Path(filename).name != filename:
        raise ValueError("Invalid DeepSeek weight index path")
    return model / filename


def preflight_files(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    owner = Path(__file__).resolve().parent / "runs" / "simulation"
    if not args.output.is_relative_to(owner.resolve()):
        raise ValueError(f"Output must stay in canonical task runs: {owner}")
    expected_checkpoint = CHECKPOINT_SHA[args.baseline]
    receipt_path = getattr(args, "reproduction_receipt", None)
    if receipt_path is not None:
        receipt = validate_reproduction_receipt(receipt_path, args.baseline, args.checkpoint)
        expected_checkpoint = receipt["artifacts"]["checkpoints/temporal/best_checkpoint.pt"]
    for path, expected in ((args.manifest, MANIFEST_SHA), (args.checkpoint, expected_checkpoint)):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or non-formal asset: {path}")
    if args.baseline in BACKBONE_SHA:
        weight = backbone_file(args.baseline, args.model)
        if not weight.is_file() or sha256(weight) != BACKBONE_SHA[args.baseline]:
            raise ValueError(f"Backbone SHA mismatch: {weight}")


def validate_reproduction_receipt(path: Path, baseline: str, checkpoint: Path) -> dict[str, Any]:
    """Explicit provenance for NEW heads; historical SHA constants stay unchanged."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (baseline not in BACKBONE_SHA or receipt.get("baseline") != baseline
            or receipt.get("status") != "complete"
            or receipt.get("origin") != "retrained_original_protocol"
            or receipt.get("manifest_sha256") != MANIFEST_SHA
            or receipt.get("backbone_sha256") != BACKBONE_SHA[baseline]
            or receipt.get("backbone_sha256_after") != BACKBONE_SHA[baseline]
            or receipt.get("training_contract") != {
                "target": "temporal", "epochs": 50, "batch_size": 512,
                "seed": 42, "learning_rate": 0.001, "weight_decay": 0.0,
                "loss": "cross_entropy_only", "scheduler": "cosine",
                "train_samples": 2250, "test_samples": 558,
                "selection": "highest_TEST_SRCC_development_no_VAL",
            }):
        raise ValueError("Invalid original-protocol reproduction receipt")
    artifacts = receipt.get("artifacts", {})
    relative = "checkpoints/temporal/best_checkpoint.pt"
    if checkpoint.resolve() != (path.parent / relative).resolve():
        raise ValueError("Reproduction checkpoint is not bound to its receipt")
    for name in (relative, "checkpoints/temporal/training_report.json",
                 "checkpoints/temporal/train_history.json", "cpu_verification.json"):
        asset = path.parent / name
        if not asset.is_file() or sha256(asset) != artifacts.get(name):
            raise ValueError(f"Reproduction artifact SHA mismatch: {name}")
    return receipt


def process_audit() -> list[dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    records = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) == 3 and fields[1].strip() == A100_UUID:
            records.append(dict(zip(("pid", "gpu_uuid", "process"), map(str.strip, fields))))
    foreign = [row for row in records if row["pid"] != str(os.getpid())]
    if foreign:
        raise RuntimeError(f"A100 occupied by other compute processes: {foreign}")
    return records


def load_runtime(args: argparse.Namespace):
    import torch
    from . import quality_token_common as core
    from LightGenV2.common.baseline_measurement import FirstBlockTimer

    device = torch.device("cuda:0")
    if args.baseline == "qwen":
        from . import quality_token_batch_benchmark_200 as original
        args.image_size = 448
        processor, model, head, scores, rows, prompt = original.load_runtime(args)
        head.requires_grad_(False)
        timer = core.BoundaryTimer(model)

        def prepare(selected):
            return original.prepare_batch(selected, processor, prompt, torch.device("cpu"), 448)

        def transfer(inputs):
            return {key: value.to(device) for key, value in inputs.items()}

        def forward(inputs):
            _, predictions, timing = original.forward_batch(inputs, model, head, scores, timer)
            return predictions, timing["vision_first_block_to_score_host_ms"], timing["vision_first_block_to_score_cuda_ms"]

        return model, head, rows, prepare, transfer, forward, timer, []

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_arch = {
        "clip": "frozen_openai/clip-vit-base-patch32_plus_five_ordered_frame_rows",
        "yolo": "frozen_ultralytics/yolo11s_plus_five_ordered_frame_rows",
        "deepseek": "frozen_deepseek_vl2_tiny_plus_five_native_lm_output_rows",
    }
    # Exact PT SHA is authoritative; additionally reject mismatched head shapes.
    weight = payload["state_dict"]["weight"].float().to(device)
    width = 1280 if args.baseline == "deepseek" else 2048
    if tuple(weight.shape) != (5, width) or len(payload["level_scores"]) != 5:
        raise ValueError("Formal trained quality rows have an unexpected shape")
    if args.baseline == "deepseek" and payload["architecture"] != expected_arch[args.baseline]:
        raise ValueError("Not the formal DeepSeek quality checkpoint")
    scores = payload["level_scores"].float().to(device)
    head = torch.nn.Linear(width, 5, bias=False, device=device).eval().requires_grad_(False)
    head.weight.copy_(weight)
    rows = [row for row in core.read_manifest(args.manifest) if row["split"] == "test"]
    handles = []
    if args.baseline == "clip":
        import clip
        model, transform = clip.load(str(args.model), device=device, jit=False)
        model.eval().requires_grad_(False)
        image_size = 224
        first_block = model.visual.transformer.resblocks[0]

        def encode(frames):
            return torch.stack([transform(frame) for frame in frames])

        def features(inputs):
            with torch.autocast("cuda", dtype=torch.float16):
                feature = model.encode_image(inputs).float()
            return torch.nn.functional.normalize(feature, dim=-1).reshape(-1, 2048).half().float()

    elif args.baseline == "yolo":
        from ultralytics import YOLO
        import numpy as np
        model = YOLO(str(args.model)).model.to(device).eval().requires_grad_(False)
        captured = {}
        if model.model[10].__class__.__name__ != "C2PSA":
            raise RuntimeError("YOLO native feature layer changed")
        handles.append(model.model[10].register_forward_hook(lambda _m, _a, out: captured.__setitem__("feature", out)))
        first_block, image_size = model.model[0], 640

        def encode(frames):
            array = np.stack([np.asarray(frame, dtype=np.uint8).copy() for frame in frames])
            return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float().div_(255)

        def features(inputs):
            captured.clear()
            with torch.autocast("cuda", dtype=torch.float16):
                model(inputs)
            feature = captured["feature"].float().mean((-2, -1))
            return torch.nn.functional.normalize(feature, dim=-1).reshape(-1, 2048).half().float()

    else:
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor
        processor = DeepseekVLV2Processor.from_pretrained(str(args.model))
        model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
            trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval().requires_grad_(False)
        first_block, image_size = model.vision.blocks[0], 384

        def encode(frames):
            prepared = []
            for start in range(0, len(frames), 4):
                selected = frames[start:start + 4]
                content = "\n".join([*[f"Frame {i + 1}: <image>" for i in range(4)], core.PROMPTS["temporal"]])
                conversation = [
                    {"role": "<|User|>", "content": content, "images": [f"frame_{i + 1:02d}.png" for i in range(4)]},
                    {"role": "<|Assistant|>", "content": ""},
                ]
                prepared.append(processor.process_one(conversations=conversation, images=selected, system_prompt=""))
            return processor.batchify(prepared)

        def features(inputs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model.prepare_inputs_embeds(**inputs)
                hidden = model.language.model(inputs_embeds=embeddings, attention_mask=inputs.attention_mask,
                    use_cache=False, return_dict=True).last_hidden_state
            mask = inputs.attention_mask.bool()
            indices = torch.arange(mask.shape[1], device=device).expand_as(mask)
            last = indices.masked_fill(~mask, -1).amax(1)
            return hidden[torch.arange(hidden.shape[0], device=device), last].float().half().float()

    from .visual_backbone_quality import decode_ordered_frames
    timer = FirstBlockTimer(first_block)

    def prepare(selected):
        frames, positions = [], []
        for row in selected:
            decoded, where = decode_ordered_frames(Path(row["video_path"]), core.FRAME_FRACTIONS[4], image_size)
            frames.extend(decoded)
            positions.append(where)
        return encode(frames), positions

    def transfer(inputs):
        if args.baseline == "deepseek":
            return inputs.to(device, dtype=torch.bfloat16)
        return inputs.to(device)

    def forward(inputs):
        torch.cuda.synchronize()
        timer.reset()
        # FP16 feature storage is part of the original trained-head contract.
        feature = features(inputs)
        prediction = (feature @ head.weight.t()).softmax(-1) @ scores
        timing = timer.finish()
        return prediction.detach().cpu().tolist(), timing["host_ms"], timing["cuda_ms"]

    return model, head, rows, prepare, transfer, forward, timer, handles


def run(args: argparse.Namespace) -> None:
    import torch
    import numpy as np
    from . import quality_token_common as core
    from LightGenV2.common import baseline_measurement as measurement

    if os.environ.get("CUDA_VISIBLE_DEVICES") != A100_UUID:
        raise RuntimeError(f"Bind the sole physical A100 by UUID: {A100_UUID}")
    before = process_audit()
    gpu_name = measurement.validate_cuda_device("A100")
    rated_w = measurement.gpu_power_limit_w(A100_UUID)
    identity = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(Path(__file__)), "pid": os.getpid(), "gpu_uuid": A100_UUID,
        "gpu_name": gpu_name, "actual_power_limit_w": rated_w,
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "model": str(args.model), "baseline": args.baseline, "batch_size_videos": args.batch,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "process_audit_before_load": before,
        "checkpoint_origin": ("retrained_original_protocol" if getattr(args, "reproduction_receipt", None)
                              else "historical_exact_SHA"),
        "reproduction_receipt_sha256": (sha256(args.reproduction_receipt)
            if getattr(args, "reproduction_receipt", None) else None),
        "model_file_sha256": (
            {path.name: sha256(path) for path in sorted(args.model.iterdir())
             if path.is_file() and (path.suffix in (".safetensors", ".bin")
                                   or path.name in ("config.json", "model.safetensors.index.json"))}
            if args.model.is_dir() else {args.model.name: sha256(args.model)}
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "status.json", {"status": "loading", **identity})
    sampler, timer, handles = None, None, []
    try:
        with torch.inference_mode():
            model, head, rows, prepare, transfer, forward, timer, handles = load_runtime(args)
            if len(rows) != 558 or any(p.requires_grad for module in (model, head) for p in module.parameters()):
                raise RuntimeError("Full 558 TEST / fully frozen model and head required")
            sampler = measurement.NvidiaSmiPowerSampler(A100_UUID)
            sampler.start()
            sampler.set_phase("idle")
            time.sleep(2)
            sampler.set_phase(None)
            predictions, timings, audits = [], [], []
            with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as pred_stream, \
                 (args.output / "batch_timing.csv").open("w", encoding="utf-8", newline="") as timing_stream:
                pred_writer = csv.writer(pred_stream)
                pred_writer.writerow(["sample_index", "sample_id", "target_mos", "prediction", "batch_index", "selected_frame_positions"])
                timing_writer = csv.writer(timing_stream)
                timing_writer.writerow(["batch_index", "videos", "in_cost_subset", "synchronized_wall_ms", "cuda_ms", "preprocessing_audit_ms", "h2d_audit_ms"])
                for batch_index, start in enumerate(range(0, len(rows), args.batch)):
                    selected = rows[start:start + args.batch]
                    if batch_index % 25 == 0:
                        audits.append({"start_index": start, "processes": process_audit()})
                    started = time.perf_counter()
                    cpu_inputs, positions = prepare(selected)
                    preprocessing = 1000 * (time.perf_counter() - started)
                    started = time.perf_counter()
                    inputs = transfer(cpu_inputs)
                    torch.cuda.synchronize()
                    h2d = 1000 * (time.perf_counter() - started)
                    in_cost = start + len(selected) <= 200
                    sampler.set_phase(f"active:{batch_index}" if in_cost else None)
                    try:
                        values, host_ms, cuda_ms = forward(inputs)
                    finally:
                        sampler.set_phase(None)
                    if len(values) != len(selected):
                        raise RuntimeError("Prediction / sample count mismatch")
                    timing = [batch_index, len(selected), in_cost, host_ms, cuda_ms, preprocessing, h2d]
                    timings.append(timing)
                    timing_writer.writerow(timing)
                    for offset, (row, value) in enumerate(zip(selected, values)):
                        predictions.append({"sample_id": row["sample_id"], "target_mos": float(row["temporal"]), "prediction": value})
                        pred_writer.writerow([start + offset, row["sample_id"], row["temporal"], value, batch_index, json.dumps(positions[offset])])
                    pred_stream.flush()
                    timing_stream.flush()
                    del inputs, cpu_inputs
                    if batch_index == 0 or len(predictions) % 50 < args.batch or len(predictions) == 558:
                        write_json(args.output / "status.json", {"status": "evaluating", "completed_test_videos": len(predictions), **identity})
                        print(f"[{args.baseline} B={args.batch}] {len(predictions)}/558 inference={host_ms:.3f}ms audit_prepare={preprocessing:.1f}ms", flush=True)
            power_samples = sampler.stop()
            sampler = None
            measurement.save_power_samples(args.output / "power_samples.csv", power_samples)
            cost_rows = [row for row in timings if row[2]]
            if sum(row[1] for row in cost_rows) != 200:
                raise RuntimeError("Cost subset is not the first 200 videos")
            power = measurement.power_report(power_samples, [row[3] for row in cost_rows], power_limit_w=rated_w)
            final_audit = process_audit()
            report = {
                "status": "complete", "schema_version": 1, **identity,
                "test_videos": len(predictions), "cost_test_videos": 200,
                "performance": core.metrics(np.asarray([row["target_mos"] for row in predictions]), np.asarray([row["prediction"] for row in predictions])),
                "explicit_warmup_forwards": 0, "first_test_batch_included": True,
                "execution_order": "CPU decode/processor then H2D then frozen inference for each distinct-video batch",
                "timing_boundary": "native first backbone block input to MOS ready on GPU; synchronized host wall time",
                "excluded_from_final_latency_and_energy": ["model load", "decode", "crop/resize", "processor", "H2D", "audit overhead"],
                "energy_formula": "(338.2 W host proxy + measured inference-active A100 mean W) * inference wall seconds",
                "energy_measurement_note": "mean tagged-window board power times mean boundary duration, not a calibrated per-call energy integral; telemetry time resolution is 50 ms",
                "cost": system_cost(statistics.mean(row[3] for row in cost_rows), power["active_mean_w"], args.batch),
                "inference_batch_wall_ms": measurement.summarize([row[3] for row in cost_rows]),
                "inference_batch_cuda_ms_secondary": measurement.summarize([row[4] for row in cost_rows]),
                "preprocessing_audit_ms_not_in_cost": measurement.summarize([row[5] for row in timings]),
                "h2d_audit_ms_not_in_cost": measurement.summarize([row[6] for row in timings]),
                "power": power, "process_audits": audits, "process_audit_at_end": final_audit,
                "checkpoint_selection": "previously TEST-selected quality rows (development metric); no training or selection in this evaluation",
            }
            write_json(args.output / "report.json", report)
            write_json(args.output / "status.json", {"status": "complete", "completed_test_videos": 558, **identity})
            print(json.dumps({"baseline": args.baseline, "batch": args.batch, "performance": report["performance"], "cost": report["cost"]}), flush=True)
    except BaseException as error:
        write_json(args.output / "status.json", {"status": "failed", "error": repr(error), **identity})
        raise
    finally:
        if sampler is not None:
            samples = sampler.stop()
            measurement.save_power_samples(args.output / "power_samples.csv", samples)
        if timer is not None:
            timer.close()
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(CHECKPOINT_SHA), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    for name in ("manifest", "checkpoint", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reproduction-receipt", type=Path,
                        help="Explicit SHA-bound receipt for a newly retrained original-protocol head")
    args = parser.parse_args()
    for name in ("manifest", "checkpoint", "model", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.reproduction_receipt is not None:
        args.reproduction_receipt = args.reproduction_receipt.expanduser().resolve()
    preflight_files(args)
    run(args)


if __name__ == "__main__":
    main()
