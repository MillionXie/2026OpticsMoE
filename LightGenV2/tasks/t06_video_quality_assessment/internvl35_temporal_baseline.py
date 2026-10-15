"""Pinned InternVL3.5-2B: original four-frame/five-output-row protocol, accuracy first.

No backbone updates, LoRA, adapters, extra frames, or hyperparameter search.
Concurrent A100 diagnostics are explicitly permitted for accuracy only, never costs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from . import smolvlm2_temporal_baseline as shared

TASK, REPO = shared.TASK, shared.REPO
CONFIG = TASK / "configs/baselines/internvl35_temporal_4f.json"
frozen = shared.frozen


def read_config():
    config = json.loads(CONFIG.read_text())
    required = dict(frames=4, hidden_width=2048, epochs=50, training_batch_size=512,
        feature_batch_size=2, learning_rate=.001, weight_decay=0., seed=42,
        backbone_trainable_parameters=0, quality_output_trainable_parameters=10240,
        train_samples=2250, test_samples=558, frame_fractions=[.1, .37, .63, .9],
        center_crop_short_side_fraction=.65, decode_resize=448, native_frame_size=448,
        native_image_tokens_per_frame=256, loss="cross_entropy_only", scheduler="cosine",
        quality_token_prefix="", selection="highest_TEST_SRCC_development_no_VAL",
        model_id="OpenGVLab/InternVL3_5-2B-HF", manifest_sha256=frozen.MANIFEST_SHA)
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError("Original budget/native interface contract changed: " + key)
    return config


def sources():
    return {**shared.sources(), str(Path(__file__).relative_to(REPO)): frozen.sha256(Path(__file__)),
            str(CONFIG.relative_to(REPO)): frozen.sha256(CONFIG)}


def gpu_guard(args, identity=None):
    if identity is not None and sources() != identity["source_sha256"]:
        raise RuntimeError("Run source changed; preserve partial evidence")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("Only explicitly authorized physical GPU UUID may be visible")
    if args.gpu_uuid == shared.TRAINING_3090_UUID:
        if args.phase not in ("train", "gpu-smoke"):
            raise ValueError("3090 is training/smoke only")
        return shared.training_guard(args.gpu_uuid)
    if args.gpu_uuid != frozen.A100_UUID or not args.accuracy_only_concurrent:
        raise ValueError("A100 use must explicitly acknowledge accuracy-only concurrency")
    raw = subprocess.check_output(["nvidia-smi", f"--id={args.gpu_uuid}",
        "--query-gpu=name,power.limit,memory.free", "--format=csv,noheader,nounits"], text=True)
    name, limit, free = next(csv.reader(raw.splitlines()))
    if "A100" not in name or float(limit) != 250:
        raise RuntimeError("Authorized A100 identity/power limit changed; do not modify power")
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    records = [dict(zip(("pid", "gpu_uuid", "memory_mib"), map(str.strip, row)))
               for row in csv.reader(raw.splitlines()) if len(row) == 3 and row[1].strip() == args.gpu_uuid]
    # Initial admission only; never reserve or displace another job. BF16 model + B8 margin.
    if not any(r["pid"] == str(os.getpid()) for r in records) and float(free) < 18000:
        raise RuntimeError(f"Insufficient free A100 memory for safe admission: {free} MiB")
    return records


def processor_runtime(model_path):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    encoded = [processor.tokenizer.encode(word, add_special_tokens=False) for word in shared.WORDS]
    if any(len(tokens) != 1 for tokens in encoded) or len({v[0] for v in encoded}) != 5:
        raise RuntimeError(f"Expected five distinct native single tokens: {encoded}")
    prompt = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": shared.PROMPT}]}],
        tokenize=False, add_generation_prompt=True)
    return processor, [tokens[0] for tokens in encoded], prompt


def prepare_batch(rows, processor, prompt):
    import numpy as np
    from . import quality_token_common as core
    decoded = [core.decode_random_seek(Path(row["video_path"]), core.FRAME_FRACTIONS[4], 448) for row in rows]
    inputs = processor(text=[prompt]*len(rows), videos=[np.stack([np.asarray(f) for f in d[0]]) for d in decoded],
        padding=True, return_tensors="pt", do_sample_frames=False,
        videos_kwargs={"size": {"height": 448, "width": 448},
                       "image_mean": processor.image_processor.image_mean,
                       "image_std": processor.image_processor.image_std})
    if tuple(inputs["pixel_values"].shape) != (len(rows)*4, 3, 448, 448):
        raise RuntimeError(f"Expected exactly four native448 frames: {inputs['pixel_values'].shape}")
    counts = inputs["input_ids"].eq(processor.image_token_id).sum(1)
    if not bool(counts.eq(4*256).all()):
        raise RuntimeError(f"Expected 1024 image tokens per video: {counts.tolist()}")
    return dict(inputs), [d[2] for d in decoded]


def load_runtime(model_path):
    import torch
    from transformers import InternVLForConditionalGeneration
    processor, ids, prompt = processor_runtime(model_path)
    model = InternVLForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).eval().requires_grad_(False).cuda()
    if any(p.requires_grad for p in model.parameters()) or model.config.text_config.hidden_size != 2048:
        raise RuntimeError("Frozen2048 backbone required")
    initial = model.lm_head.weight.detach()[ids].float().cpu().clone()
    return processor, model, initial, ids, prompt


def train_head(payload, initial, args, config, identity, ids):
    import numpy as np
    import torch
    from . import quality_token_train as training
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    features, targets = payload["features"].float(), payload["targets"].float()
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
        gpu_guard(args, identity)
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
        history.append(dict(epoch=epoch, train_cross_entropy=loss_sum/count,
            learning_rate=optimizer.param_groups[0]["lr"], test=metrics))
        if metrics["srcc"] > best_srcc:
            best_srcc, best_epoch = metrics["srcc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        frozen.write_json(args.output / "train_history.json", history)
        frozen.write_json(args.output / "status.json", dict(status="training_head", epoch=epoch,
            train_loss=loss_sum/count, best_test_srcc=best_srcc, updated_at_unix=time.time()))
        print(f"[head] e{epoch}/50 loss={loss_sum/count:.6f} TEST_SRCC={metrics['srcc']:.9f}", flush=True)
    last = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    results = {}
    for name, state, epoch in (("best", best_state, best_epoch), ("last", last, 50)):
        head.load_state_dict(state)
        gpu_metrics, gpu_predictions, *_ = training.evaluate(head, features[test], targets[test], labels[test], levels, torch.device("cuda"))
        cpu_head = training.FiveNativeTokenRows(state["weight"]).eval().requires_grad_(False)
        cpu_metrics, cpu_predictions, *_ = training.evaluate(cpu_head, features[test], targets[test], labels[test], levels, torch.device("cpu"))
        training.atomic_save(args.output / f"{name}_checkpoint.pt", dict(
            architecture="frozen_internvl35_plus_five_untied_native_output_rows", state_dict=state,
            initial_native_rows=initial, selected_epoch=epoch, boundaries=boundaries, level_scores=levels,
            metrics=gpu_metrics, quality_token_ids=ids, trainable_parameters=10240,
            all_backbone_parameters_frozen=True, selection=config["selection"], identity=identity))
        with (args.output / f"cpu_{name}_predictions.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "target_mos", "prediction_mos"])
            writer.writerows((payload["sample_ids"][i], float(targets[i]), float(p)) for i, p in zip(test.tolist(), cpu_predictions))
        results[name] = dict(epoch=epoch, gpu_metrics=gpu_metrics, strict_cpu_metrics=cpu_metrics,
            max_cpu_gpu_mos_difference=float((cpu_predictions-gpu_predictions).abs().max()))
    frozen.write_json(args.output / "cpu_verification.json", results)
    frozen.write_json(args.output / "training_report.json", dict(status="complete", contract=config,
        results=results, backbone_updates=0, output_row_parameters=10240,
        native_rows_before_sha256=training.sha256_tensor(initial),
        best_rows_sha256=training.sha256_tensor(best_state["weight"]), identity=identity))


def run_train(args, config, identity):
    import torch
    from . import quality_token_common as core
    from .quality_token_train import atomic_save
    rows = core.read_manifest(args.manifest)
    if len(rows) != 2808 or len({r["sample_id"] for r in rows}) != 2808:
        raise RuntimeError("Expected 2808 unique videos")
    processor, model, initial, ids, prompt = load_runtime(args.model)
    parts = args.output / "feature_parts"
    parts.mkdir()
    features = torch.empty((2808, 2048), dtype=torch.float16)
    for begin in range(0, 2808, 64):
        gpu_guard(args, identity)
        end = min(begin+64, 2808)
        for start in range(begin, end, 2):
            inputs, _ = prepare_batch(rows[start:min(start+2, end)], processor, prompt)
            gpu_inputs = shared.transfer(inputs)
            features[start:start+len(inputs["input_ids"])] = shared.pooled_features(model, gpu_inputs).cpu().half()
            del inputs, gpu_inputs
        gpu_guard(args, identity)
        atomic_save(parts / f"rows_{begin:05d}_{end:05d}.pt", dict(
            sample_ids=[r["sample_id"] for r in rows[begin:end]], features=features[begin:end].clone(), identity=identity))
        frozen.write_json(args.output / "status.json", dict(status="extracting", videos=end,
            total=2808, updated_at_unix=time.time()))
        print(f"[features] {end}/2808", flush=True)
    if not torch.equal(initial, model.lm_head.weight.detach()[ids].float().cpu()) or any(
            p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen backbone/native output rows audit failed")
    payload = dict(features=features, targets=torch.tensor([r["temporal"] for r in rows]),
        splits=[r["split"] for r in rows], sample_ids=[r["sample_id"] for r in rows],
        initial_native_rows=initial, quality_token_ids=ids, identity=identity)
    atomic_save(args.output / "features.pt", payload)
    del processor, model
    torch.cuda.empty_cache()
    train_head(payload, initial, args, config, identity, ids)
    if shared.model_files(args.model, config) != identity["model_sha256"]:
        raise RuntimeError("Official model files changed during training")
    artifacts = {str(p.relative_to(args.output)): frozen.sha256(p) for p in args.output.rglob("*")
                 if p.is_file() and p.name != "status.json"}
    frozen.write_json(args.output / "training_receipt.json", dict(status="complete", identity=identity, artifacts=artifacts))


def run_smoke(args, config, identity):
    import torch
    from . import quality_token_common as core
    from .quality_token_train import FiveNativeTokenRows, atomic_save
    processor, model, initial, ids, prompt = load_runtime(args.model)
    rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "train"][:2]
    inputs, positions = prepare_batch(rows, processor, prompt)
    features = shared.pooled_features(model, shared.transfer(inputs)).detach().cpu().half().clone()
    head = FiveNativeTokenRows(initial).cuda()
    labels = torch.tensor([0, 4], device="cuda")  # Isolated update probe, not reported as task performance.
    loss = torch.nn.functional.cross_entropy(head(features.float().cuda()), labels)
    loss.backward()
    torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.).step()
    if not bool(torch.isfinite(loss)) or torch.equal(head.weight.detach().cpu(), initial) or any(
            p.requires_grad or p.grad is not None for p in model.parameters()) or not torch.equal(
            initial, model.lm_head.weight.detach()[ids].float().cpu()):
        raise RuntimeError("Frozen backbone/isolated10240-row smoke audit failed")
    atomic_save(args.output / "smoke_evidence.pt", dict(features=features, initial=initial,
        updated=head.weight.detach().cpu(), sample_ids=[r["sample_id"] for r in rows]))
    frozen.write_json(args.output / "gpu_smoke_report.json", dict(status="passed", real_train_videos=2,
        output_trainable_parameters=10240, backbone_frozen=True, original_rows_unchanged=True,
        loss_probe_only=float(loss.detach()), shapes={k: list(v.shape) for k, v in inputs.items()},
        frame_positions=positions, token_ids=ids, not_a_performance_result=True, identity=identity))


def run_eval(args, config, identity):
    import torch
    import numpy as np
    from . import quality_token_common as core
    from .quality_token_train import FiveNativeTokenRows
    receipt = json.loads((args.checkpoint.parent / "training_receipt.json").read_text())
    for name, sha in receipt["artifacts"].items():
        if frozen.sha256(args.checkpoint.parent / name) != sha:
            raise ValueError("Training receipt artifact differs: " + name)
    for key in ("model_sha256", "manifest_sha256", "source_sha256"):
        if receipt["identity"][key] != identity[key]:
            raise ValueError("Training/evaluation identity differs: " + key)
    rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "test"]
    if len(rows) != 558:
        raise RuntimeError("Expected complete558TEST")
    processor, model, _, ids, prompt = load_runtime(args.model)
    pt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if pt["quality_token_ids"] != ids:
        raise ValueError("Quality token rows changed")
    head = FiveNativeTokenRows(pt["state_dict"]["weight"]).eval().requires_grad_(False).cuda()
    levels = pt["level_scores"].cuda()
    predictions, audits = [], []
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "sample_id", "target_mos", "prediction", "batch_index", "frame_positions"])
        for bi, begin in enumerate(range(0, 558, args.batch)):
            if bi % 25 == 0:
                audits.append(dict(at=begin, processes=gpu_guard(args, identity)))
            selected = rows[begin:begin+args.batch]
            inputs, positions = prepare_batch(selected, processor, prompt)
            gpu_inputs = shared.transfer(inputs)
            with torch.inference_mode():
                values = (head(shared.pooled_features(model, gpu_inputs)).softmax(-1) @ levels).cpu().tolist()
            predictions.extend(values)
            writer.writerows((begin+i, r["sample_id"], r["temporal"], v, bi, json.dumps(positions[i]))
                for i, (r, v) in enumerate(zip(selected, values)))
            stream.flush()
            del inputs, gpu_inputs
            if bi % 25 == 0 or len(predictions) == 558:
                frozen.write_json(args.output / "status.json", dict(status="evaluating", batch=args.batch,
                    test_videos=len(predictions), updated_at_unix=time.time()))
                print(f"[InternVL B{args.batch}] {len(predictions)}/558", flush=True)
    frozen.write_json(args.output / "report.json", dict(status="complete", test_videos=558,
        batch_size_videos=args.batch, checkpoint_sha256=frozen.sha256(args.checkpoint),
        performance=core.metrics(np.array([r["temporal"] for r in rows]), np.array(predictions)),
        accuracy_only=True, formal_latency_energy_measured=False, concurrent_diagnostics_permitted=True,
        process_audits=audits, precision="BF16 frozen backbone, FP32 five-row head",
        selection=config["selection"], identity=identity))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu-preflight", "gpu-smoke", "train", "evaluate"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch", choices=(1, 4, 8), type=int)
    parser.add_argument("--gpu-uuid", choices=(frozen.A100_UUID, shared.TRAINING_3090_UUID), default=frozen.A100_UUID)
    parser.add_argument("--accuracy-only-concurrent", action="store_true")
    args = parser.parse_args()
    for name in ("model", "manifest", "output", "checkpoint"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    config = read_config()
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Original manifest differs")
    if args.phase == "evaluate" and (args.checkpoint is None or args.batch is None):
        parser.error("Evaluation requires fixed checkpoint and batch")
    shared.new_output(args.output)
    args.output.mkdir(parents=True)
    identity = dict(source_sha256=sources(), manifest_sha256=frozen.MANIFEST_SHA, contract=config,
        pid=os.getpid(), created_at_unix=time.time(), gpu_uuid=args.gpu_uuid,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        command=sys.argv, accuracy_only=True, concurrent_a100_authorized=args.accuracy_only_concurrent)
    frozen.write_json(args.output / "run_identity.json", identity)
    try:
        if args.phase == "cpu-preflight":
            from . import quality_token_common as core
            processor, ids, prompt = processor_runtime(args.model)
            rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "train"][:2]
            results = []
            for b in (1, 2):
                inputs, positions = prepare_batch(rows[:b], processor, prompt)
                results.append(dict(batch=b, shapes={k: list(v.shape) for k, v in inputs.items()}, positions=positions))
            frozen.write_json(args.output / "cpu_preflight_report.json", dict(status="passed", results=results,
                native_quality_token_ids=ids, gpu_initialized=False, contract=config))
        else:
            identity["process_audit_before"] = gpu_guard(args)
            identity["model_sha256"] = shared.model_files(args.model, config)
            import torch, transformers
            identity.update(torch=torch.__version__, transformers=transformers.__version__,
                gpu_name=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
            frozen.write_json(args.output / "run_identity.json", identity)
            if args.phase == "train":
                run_train(args, config, identity)
            elif args.phase == "gpu-smoke":
                run_smoke(args, config, identity)
            else:
                run_eval(args, config, identity)
            gpu_guard(args, identity)
        frozen.write_json(args.output / "status.json", dict(status="complete", updated_at_unix=time.time()))
    except BaseException as error:
        frozen.write_json(args.output / "status.json", dict(status="failed", error=repr(error), updated_at_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
