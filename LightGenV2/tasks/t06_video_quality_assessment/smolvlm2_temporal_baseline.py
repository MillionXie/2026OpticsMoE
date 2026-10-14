"""SmolVLM2: Qwen-matched five output rows only, then fresh A100 B1/4/8.

No LoRA, backbone updates, adapters, architecture or hyperparameter search.
Shared historical Qwen/measurement modules are reused without modifying them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from . import evaluate_temporal_baseline_batches as frozen
from . import batch_telemetry as telemetry

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
CONFIG = TASK / "configs/baselines/smolvlm2_temporal_4f_a100.json"
WORDS = ("Bad", "Poor", "Fair", "Good", "Excellent")
PROMPT = ("Please evaluate the temporal quality of this video and rate it using one of "
          "the following five levels: Excellent, Good, Fair, Poor, or Bad.")


def read_config():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    required = {"frames": 4, "hidden_width": 2048, "epochs": 50,
                "training_batch_size": 512, "feature_batch_size": 2,
                "learning_rate": .001, "weight_decay": 0., "seed": 42,
                "backbone_trainable_parameters": 0, "quality_output_trainable_parameters": 10240,
                "loss": "cross_entropy_only", "scheduler": "cosine",
                "train_samples": 2250, "test_samples": 558, "cost_samples": 200,
                "inference_batches": [1, 4, 8], "frame_fractions": [.1, .37, .63, .9],
                "center_crop_short_side_fraction": .65, "decode_resize": 448,
                "processor_native_frame_size": 384, "quality_token_prefix": " ",
                "selection": "highest_TEST_SRCC_development_no_VAL",
                "manifest_sha256": frozen.MANIFEST_SHA, "gpu_uuid": frozen.A100_UUID,
                "power_limit_w": 250, "host_proxy_w": frozen.HOST_W}
    for name, value in required.items():
        if config.get(name) != value:
            raise ValueError(f"Qwen-matched contract changed: {name}")


def new_output(path):
    owner = (TASK / "runs/simulation").resolve()
    if path.exists():
        raise FileExistsError(f"Refusing existing output {path}")
    if path.resolve() == owner or not path.resolve().is_relative_to(owner):
        raise ValueError("Output must be a new child of canonical T06 runs/simulation")


def sources():
    paths = [Path(__file__), CONFIG, TASK / "quality_token_common.py",
             TASK / "quality_token_train.py", Path(telemetry.__file__), Path(frozen.__file__),
             REPO / "LightGenV2/common/baseline_measurement.py"]
    return {str(p.relative_to(REPO)): frozen.sha256(p) for p in paths}


def guard():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Only the authorized physical A100 UUID may be visible")
    records = frozen.process_audit()
    limit = float(subprocess.check_output([
        "nvidia-smi", f"--id={frozen.A100_UUID}", "--query-gpu=power.limit",
        "--format=csv,noheader,nounits"], text=True).strip())
    if limit != 250:
        raise RuntimeError(f"A100 power limit changed: {limit}; will not change it")
    return records


def asset_valid(path, spec):
    if not path.is_file():
        return False
    if "sha256" in spec:
        return path.stat().st_size == spec["size"] and frozen.sha256(path) == spec["sha256"]
    content = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() == spec["git_blob_sha1"]


def model_files(model, config):
    result = {}
    for name, spec in config["official_assets"].items():
        path = model / name
        if not asset_valid(path, spec):
            raise ValueError(f"Official fixed-revision model asset mismatch: {name}")
        result[name] = frozen.sha256(path)
    return result


def prepare_assets(args, config):
    import requests
    from transformers import AutoProcessor
    args.model.mkdir(parents=True, exist_ok=True)
    # Metadata first: token contract is verified before downloading large weights.
    names = sorted(config["official_assets"], key=lambda s: s.endswith(".safetensors"))
    processor_checked = False
    for name in names:
        if name.endswith(".safetensors"):
            verify_tokenizer(args.model, config)
            if not processor_checked:
                from . import quality_token_common as core
                processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
                processor.tokenizer.padding_side = "left"
                prompt = core.render_prompt(processor, "temporal")
                rows = core.read_manifest(args.manifest)[:2]
                inputs, positions = prepare_batch(rows, processor, prompt)
                frozen.write_json(args.output / "cpu_processor_preflight.json", {
                    "status": "passed", "gpu_initialized": False,
                    "two_real_train_video_ids": [row["sample_id"] for row in rows],
                    "frame_positions": positions,
                    "native_tensor_shapes": {key: list(value.shape) for key, value in inputs.items()},
                    "quality_token_ids": verify_tokenizer(args.model, config)})
                processor_checked = True
        path = args.model / name
        spec = config["official_assets"][name]
        if path.exists():
            if not asset_valid(path, spec):
                raise ValueError(f"Existing model file differs; will not overwrite {path}")
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            raise FileExistsError(f"Partial download preserved; diagnose before retry: {partial}")
        url = f"{args.endpoint}/{config['model_id']}/resolve/{config['model_revision']}/{name}"
        print(f"[download] {name}", flush=True)
        with requests.get(url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            written = 0
            with partial.open("xb") as stream:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        written += len(chunk)
                        if written < 8 * 1024 * 1024 or written % (64 * 1024 * 1024) == 0:
                            frozen.write_json(args.output / "status.json", {
                                "status": "downloading", "file": name, "bytes": written,
                                "updated_at_unix": time.time(), "gpu_initialized": False})
        if not asset_valid(partial, spec):
            raise ValueError(f"Download hash mismatch, partial preserved: {name}")
        partial.replace(path)
    tokens = verify_tokenizer(args.model, config)
    frozen.write_json(args.output / "model_asset_manifest.json", {
        "model_id": config["model_id"], "revision": config["model_revision"],
        "download_endpoint": args.endpoint, "sha256": model_files(args.model, config),
        "quality_token_ids": tokens, "verified_official_git_blobs_and_lfs_sha": True})


def verify_tokenizer(model, config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    encoded = [tokenizer.encode(config["quality_token_prefix"] + word, add_special_tokens=False) for word in WORDS]
    if encoded != [[i] for i in config["quality_token_ids"]]:
        raise RuntimeError(f"Native single-token output-row contract changed: {encoded}")
    return encoded


def load_runtime(model_path, config):
    import torch
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    from .quality_token_train import FiveNativeTokenRows
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval().requires_grad_(False).cuda()
    if any(p.requires_grad for p in model.parameters()) or model.config.text_config.hidden_size != 2048:
        raise RuntimeError("All pretrained parameters must remain frozen, width 2048")
    initial = model.lm_head.weight.detach()[config["quality_token_ids"]].float().cpu().clone()
    head = FiveNativeTokenRows(initial)
    if head.weight.data_ptr() == initial.data_ptr() or sum(p.numel() for p in head.parameters()) != 10240:
        raise RuntimeError("Exactly five independent untied output rows required")
    prompt = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True)
    return processor, model, head, initial, prompt


def prepare_batch(rows, processor, prompt):
    import numpy as np
    from . import quality_token_common as core
    decoded = [core.decode_random_seek(Path(row["video_path"]), core.FRAME_FRACTIONS[4], 448) for row in rows]
    inputs = processor(text=[prompt] * len(rows),
                       videos=[np.stack([np.asarray(f) for f in item[0]]) for item in decoded],
                       video_metadata=[item[1] for item in decoded],
                       padding=True, return_tensors="pt", do_sample_frames=False)
    pixels = inputs["pixel_values"]
    if tuple(pixels.shape[:2]) != (len(rows), 4) or tuple(pixels.shape[-2:]) != (384, 384):
        raise RuntimeError(f"Expected four native 384-pixel frames, got {tuple(pixels.shape)}")
    counts = inputs["input_ids"].eq(processor.image_token_id).sum(1)
    if not bool(counts.eq(4 * 81).all()):
        raise RuntimeError(f"Video image token count changed: {counts.tolist()}")
    return dict(inputs), [item[2] for item in decoded]


def transfer(inputs):
    import torch
    return {key: value.to("cuda", dtype=torch.bfloat16 if key == "pixel_values" else value.dtype)
            for key, value in inputs.items()}


def pooled_features(model, inputs):
    import torch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
    mask = inputs["attention_mask"].bool()
    last = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask).masked_fill(~mask, -1).amax(1)
    pooled = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
    if pooled.shape != (len(hidden), 2048) or not bool(torch.isfinite(pooled).all()):
        raise RuntimeError("Invalid last-valid-token feature")
    return pooled


def train_head(payload, initial, output, config, identity):
    import numpy as np
    import torch
    from . import quality_token_train as training
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    features = payload["features"].float()
    targets = payload["targets"].float()
    train = torch.tensor([i for i, s in enumerate(payload["splits"]) if s == "train"])
    test = torch.tensor([i for i, s in enumerate(payload["splits"]) if s == "test"])
    if (len(train), len(test)) != (2250, 558):
        raise RuntimeError("Unexpected split")
    boundaries = torch.linspace(float(targets[train].min()), float(targets[train].max()), 6)[1:-1]
    levels = torch.linspace(float(targets[train].min()), float(targets[train].max()), 5)
    labels = torch.bucketize(targets, boundaries, right=False)
    if any(not bool(labels[train].eq(i).any()) for i in range(5)):
        raise RuntimeError("Empty training quality class")
    head = training.FiveNativeTokenRows(initial).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features[train], labels[train]),
        batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42), pin_memory=True)
    history, best_srcc, best_state, best_epoch = [], -float("inf"), None, None
    for epoch in range(1, 51):
        guard()
        head.train()
        loss_sum, count = 0., 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x.cuda(non_blocking=True)), y.cuda(non_blocking=True))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite train loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
        scheduler.step()
        head.eval()
        metrics, *_ = training.evaluate(head, features[test], targets[test], labels[test], levels, torch.device("cuda"))
        if not np.isfinite(metrics["srcc"]):
            raise RuntimeError("Non-finite TEST SRCC")
        history.append({"epoch": epoch, "train_cross_entropy": loss_sum/count,
                        "learning_rate": optimizer.param_groups[0]["lr"], "test": metrics})
        if metrics["srcc"] > best_srcc:
            best_srcc, best_epoch = metrics["srcc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        frozen.write_json(output / "train_history.json", history)
        frozen.write_json(output / "status.json", {"status": "training_head", "epoch": epoch,
            "train_loss": loss_sum/count, "best_test_srcc": best_srcc, "updated_at_unix": time.time()})
        print(f"[head] e{epoch}/50 loss={loss_sum/count:.6f} TEST_SRCC={metrics['srcc']:.9f}", flush=True)
    last = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    results = {}
    for name, state, epoch in (("best", best_state, best_epoch), ("last", last, 50)):
        head.load_state_dict(state)
        gpu_metrics, gpu_predictions, *_ = training.evaluate(head, features[test], targets[test], labels[test], levels, torch.device("cuda"))
        cpu_head = training.FiveNativeTokenRows(state["weight"]).eval().requires_grad_(False)
        cpu_metrics, cpu_predictions, *_ = training.evaluate(cpu_head, features[test], targets[test], labels[test], levels, torch.device("cpu"))
        payload_pt = {"architecture": "frozen_smolvlm2_plus_five_untied_native_output_rows",
            "state_dict": state, "initial_native_rows": initial, "selected_epoch": epoch,
            "boundaries": boundaries, "level_scores": levels, "metrics": gpu_metrics,
            "quality_token_ids": config["quality_token_ids"], "trainable_parameters": 10240,
            "all_backbone_parameters_frozen": True, "selection": config["selection"], "identity": identity}
        training.atomic_save(output / f"{name}_checkpoint.pt", payload_pt)
        with (output / f"cpu_{name}_predictions.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "target_mos", "prediction_mos"])
            writer.writerows((payload["sample_ids"][i], float(targets[i]), float(p)) for i, p in zip(test.tolist(), cpu_predictions))
        results[name] = {"epoch": epoch, "gpu_metrics": gpu_metrics, "strict_cpu_metrics": cpu_metrics,
                         "max_cpu_gpu_mos_difference": float((cpu_predictions-gpu_predictions).abs().max())}
    frozen.write_json(output / "cpu_verification.json", results)
    frozen.write_json(output / "training_report.json", {"status": "complete", "contract": config,
        "results": results, "backbone_updates": 0, "output_row_parameters": 10240,
        "native_rows_before_sha256": training.sha256_tensor(initial),
        "best_rows_sha256": training.sha256_tensor(best_state["weight"]), "identity": identity})


def run_training(args, config, identity):
    import torch
    from . import quality_token_common as core
    from .quality_token_train import atomic_save
    rows = core.read_manifest(args.manifest)
    if len(rows) != 2808 or len({r["sample_id"] for r in rows}) != 2808:
        raise RuntimeError("Expected 2808 unique videos")
    processor, model, head, initial, prompt = load_runtime(args.model, config)
    features = torch.empty((2808, 2048), dtype=torch.float16)
    parts = args.output / "feature_parts"
    parts.mkdir()
    shapes = None
    for begin in range(0, 2808, 64):
        guard()
        end = min(begin + 64, 2808)
        for start in range(begin, end, 2):
            selected = rows[start:min(start+2, end)]
            inputs, _ = prepare_batch(selected, processor, prompt)
            shapes = {k: list(v.shape) for k, v in inputs.items()}
            gpu_inputs = transfer(inputs)
            with torch.inference_mode():
                features[start:start+len(selected)] = pooled_features(model, gpu_inputs).cpu().half()
            del inputs, gpu_inputs
        atomic_save(parts / f"rows_{begin:05d}_{end:05d}.pt", {
            "sample_ids": [r["sample_id"] for r in rows[begin:end]], "features": features[begin:end].clone(), "identity": identity})
        frozen.write_json(args.output / "status.json", {"status": "extracting", "videos": end,
            "total": 2808, "input_shapes": shapes, "updated_at_unix": time.time()})
        print(f"[features] {end}/2808", flush=True)
    payload = {"features": features, "targets": torch.tensor([r["temporal"] for r in rows]),
        "splits": [r["split"] for r in rows], "sample_ids": [r["sample_id"] for r in rows],
        "initial_native_rows": initial, "identity": identity}
    atomic_save(args.output / "features.pt", payload)
    del processor, model, head
    torch.cuda.empty_cache()
    train_head(payload, initial, args.output, config, identity)
    if model_files(args.model, config) != identity["model_sha256"]:
        raise RuntimeError("Pretrained model files changed")
    artifacts = {str(p.relative_to(args.output)): frozen.sha256(p) for p in args.output.iterdir()
                 if p.is_file() and p.name != "status.json"}
    frozen.write_json(args.output / "training_receipt.json", {"status": "complete", "identity": identity, "artifacts": artifacts})


def run_evaluation(args, config, identity):
    import torch
    import numpy as np
    from . import quality_token_common as core
    from LightGenV2.common.baseline_measurement import FirstBlockTimer
    receipt = json.loads((args.checkpoint.parent / "training_receipt.json").read_text())
    for name, sha in receipt["artifacts"].items():
        if frozen.sha256(args.checkpoint.parent / name) != sha:
            raise ValueError(f"Training receipt mismatch: {name}")
    for key in ("model_sha256", "manifest_sha256", "source_sha256"):
        if receipt["identity"][key] != identity[key]:
            raise ValueError(f"Training/evaluation identity mismatch: {key}")
    rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "test"]
    if len(rows) != 558:
        raise ValueError("Expected complete 558 TEST")
    processor, model, head, _, prompt = load_runtime(args.model, config)
    pt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(pt["state_dict"], strict=True)
    head = head.eval().requires_grad_(False).cuda()
    scores = pt["level_scores"].cuda()
    timer = FirstBlockTimer(model.model.vision_model.encoder.layers[0])
    sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
    samples, predictions, timings = [], [], []

    def forward(inputs):
        with torch.inference_mode():
            torch.cuda.synchronize()
            timer.reset()
            pooled = pooled_features(model, inputs)
            # Match Qwen's retained full native-vocabulary projection at the final token.
            native_full_logits = model.lm_head(pooled.to(model.lm_head.weight.dtype))
            values = head(pooled).softmax(-1) @ scores
            result = timer.finish()
            if result["first_block_calls"] != 1:
                raise RuntimeError("Expected exactly one vision block-0 call per batch")
            return values.detach().cpu().tolist(), result
    try:
        sampler.start()
        sampler.set_phase("idle")
        time.sleep(2)
        sampler.set_phase(None)
        with (args.output / "predictions.csv").open("x", newline="") as pred_stream, \
             (args.output / "batch_timing.csv").open("x", newline="") as time_stream:
            pred_writer, time_writer = csv.writer(pred_stream), csv.writer(time_stream)
            pred_writer.writerow(["sample_index", "sample_id", "target_mos", "prediction", "batch_index", "frame_positions"])
            time_writer.writerow(["batch_index", "videos", "in_cost_subset", "synchronized_wall_ms", "cuda_ms", "preprocessing_ms", "h2d_ms"])
            for batch_index, (begin, end) in enumerate(telemetry.inference_plan(558, args.batch)):
                if batch_index % 25 == 0:
                    guard()
                selected = rows[begin:end]
                started = time.perf_counter()
                inputs, positions = prepare_batch(selected, processor, prompt)
                prep_ms = (time.perf_counter()-started)*1000
                started = time.perf_counter()
                gpu_inputs = transfer(inputs)
                torch.cuda.synchronize()
                h2d_ms = (time.perf_counter()-started)*1000
                sampler.set_phase(f"active:{batch_index}" if end <= 200 else None)
                try:
                    values, timed = forward(gpu_inputs)
                finally:
                    sampler.set_phase(None)
                timing = [batch_index, end-begin, end <= 200, timed["host_ms"], timed["cuda_ms"], prep_ms, h2d_ms]
                timings.append(timing)
                time_writer.writerow(timing)
                for offset, (row, value) in enumerate(zip(selected, values)):
                    predictions.append(value)
                    pred_writer.writerow([begin+offset, row["sample_id"], row["temporal"], value, batch_index, json.dumps(positions[offset])])
                pred_stream.flush()
                time_stream.flush()
                del inputs, gpu_inputs
                if batch_index == 0 or batch_index % 25 == 0 or end == 558:
                    frozen.write_json(args.output / "status.json", {"status": "evaluating", "batch": args.batch,
                        "test_videos": end, "updated_at_unix": time.time()})
                    print(f"[Smol B{args.batch}] {end}/558 wall={timed['host_ms']:.3f}ms", flush=True)
        samples = sampler.stop()
        telemetry.save_samples(args.output / "telemetry.csv", samples)
        active = telemetry.telemetry_report(samples)
        cost = telemetry.workload_cost([r for r in timings if r[2]], active["mean_board_power_w"])
        report = {"status": "complete", "batch_size_videos": args.batch, "test_videos": 558,
            "cost_test_videos": 200, "identity": identity, "checkpoint_sha256": frozen.sha256(args.checkpoint),
            "performance": core.metrics(np.array([r["temporal"] for r in rows]), np.array(predictions)),
            "cost": cost, "telemetry": active, "explicit_warmup_forwards": 0,
            "timing_boundary": "first native vision Transformer block input to scalar MOS ready on GPU, synchronized wall clock",
            "excluded_costs": ["model loading", "decode", "crop/resize", "processor", "H2D", "audit"],
            "native_full_vocabulary_projection": True, "precision": "BF16 pretrained backbone, FP32 five-row head",
            "selection": config["selection"], "process_audit": guard()}
        # Keep resident-input load telemetry separate, exactly as the Qwen measurement.
        cpu_inputs, _ = prepare_batch(rows[:args.batch], processor, prompt)
        resident = transfer(cpu_inputs)
        torch.cuda.synchronize()
        sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
        sampler.start()
        sampler.set_phase("settling")
        time.sleep(2)
        sampler.set_phase("sustained:")
        latencies, started = [], time.monotonic()
        while time.monotonic()-started < 15:
            _, timed = forward(resident)
            latencies.append(timed["host_ms"])
        busy = sampler.stop()
        telemetry.save_samples(args.output / "sustained_telemetry.csv", busy)
        with (args.output / "sustained_timing.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["call_index", "synchronized_wall_ms"])
            writer.writerows(enumerate(latencies))
        busy_report = telemetry.telemetry_report(busy, "sustained:")
        time16 = sum(latencies)/len(latencies)*16/args.batch
        report["sustained_resident_input"] = {"calls": len(latencies), "telemetry": busy_report,
            "latency_16_videos_ms": time16, "energy_16_videos_j": (338.2+busy_report["mean_board_power_w"])*time16/1000,
            "workload": "15-second repeated first B distinct TEST videos resident on GPU; per-call sync/CPU MOS copy, not a gap-free saturated GPU workload",
            "not_used_for_dataset_once_cost_or_srcc": True}
        report["process_audit_after_sustained"] = guard()
        frozen.write_json(args.output / "report.json", report)
    finally:
        remaining = sampler.stop()
        if not (args.output / "telemetry.csv").exists() and remaining:
            telemetry.save_samples(args.output / "telemetry.csv", remaining)
        timer.close()


def suite(args, config):
    import fcntl
    source = sources()
    state = {"status": "running", "phase": "prepare", "completed": [], "pid": os.getpid(),
             "source_sha256": source, "contract": config}
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for phase, batch in (("prepare", None), ("train", None), ("evaluate", 1), ("evaluate", 4), ("evaluate", 8)):
                if sources() != source:
                    raise RuntimeError("Source changed during suite")
                if phase != "prepare":
                    guard()
                name = phase if batch is None else f"batch{batch:02d}"
                state["phase"] = name
                frozen.write_json(args.output / "suite_status.json", state)
                command = [sys.executable, "-u", "-m", __package__ + ".smolvlm2_temporal_baseline",
                    "--phase", phase, "--model", str(args.model), "--manifest", str(args.manifest),
                    "--output", str(args.output / name), "--endpoint", args.endpoint]
                if batch is not None:
                    command += ["--batch", str(batch), "--checkpoint", str(args.output / "train/best_checkpoint.pt")]
                frozen.write_json(args.output / f"{name}.command.json", command)
                with (args.output / f"{name}.log").open("x") as log:
                    subprocess.run(command, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
                state["completed"].append(name)
                state["updated_at_unix"] = time.time()
                frozen.write_json(args.output / "suite_status.json", state)
            state["status"] = "complete"
            state["gpu_release_audit"] = guard()
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            frozen.write_json(args.output / "suite_status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate", "suite"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"), default="https://huggingface.co")
    args = parser.parse_args()
    for name in ("model", "manifest", "output", "checkpoint"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    new_output(args.output)
    config = read_config()
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Original data manifest SHA mismatch")
    if args.phase == "evaluate" and (args.checkpoint is None or args.batch is None):
        parser.error("Evaluation needs checkpoint and batch")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": sources(), "manifest_sha256": frozen.MANIFEST_SHA,
        "model_id": config["model_id"], "revision": config["model_revision"],
        "pid": os.getpid(), "python": platform.python_version(), "contract": config,
        "original_command": sys.argv, "created_at_unix": time.time()}
    frozen.write_json(args.output / "run_identity.json", identity)
    try:
        if args.phase == "suite":
            suite(args, config)
        elif args.phase == "prepare":
            prepare_assets(args, config)
        else:
            guard()
            import torch, transformers
            if "A100" not in torch.cuda.get_device_name(0):
                raise RuntimeError("A100 required")
            identity.update(model_sha256=model_files(args.model, config),
                            torch=torch.__version__, transformers=transformers.__version__,
                            gpu_uuid=frozen.A100_UUID, cuda=torch.version.cuda)
            frozen.write_json(args.output / "run_identity.json", identity)
            if args.phase == "train":
                run_training(args, config, identity)
            else:
                run_evaluation(args, config, identity)
        frozen.write_json(args.output / "status.json", {"status": "complete", "updated_at_unix": time.time(), "identity": identity})
    except BaseException as error:
        frozen.write_json(args.output / "status.json", {"status": "failed", "error": repr(error), "identity": identity})
        raise


if __name__ == "__main__":
    main()
