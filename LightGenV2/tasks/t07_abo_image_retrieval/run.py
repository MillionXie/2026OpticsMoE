"""ABO similarity10: frozen full-Qwen baseline and six-capture optical retrieval.

One image + a fixed instruction in BOTH modalities. No category text or title
is supplied at inference. All 120 train-product centroids are ranked directly.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from .retrieval_contract import (
    INSTRUCTION, EXPECTED_ARCHIVE_SHA256, _load_contract, _gallery_centroids,
    _category_prototypes, _evaluate, _inputs, sha256_file,
)
from .refinement import WeakAugmentationDataset, CrossProductBatchSampler, lr_multiplier, enhance_graph, TrainProductBank
from LightGenV2.tasks.t01_object_retrieval.modeling import (
    load_backbone, build_student, initialize_student,
)
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings, save_resolved_config
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs, teacher_embeddings, student_embeddings, preprocess_images, validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    seed_everything, write_json, write_csv,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import _read_config, _nested
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GrocerySample, GroceryRetrievalDataset, collate_grocery,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    PKBatchSampler, _build_optimizer, encode_student_samples, save_checkpoint,
    load_checkpoint, initialize_parameter_ema, update_parameter_ema, use_parameter_ema,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import phase_dc_loss

TASK = Path(__file__).resolve().parent


def initialize_pinned_student(settings, replacement, readout):
    # The original checkpoint stores the HF repo ID, whereas offline loading
    # uses its exact pinned snapshot path. Allow ONLY this audited alias; all
    # checkpoint SHA, architecture, shapes and selection checks still run.
    pinned = Path(settings.model_id)
    init_settings = copy.copy(settings)
    if (pinned.name == "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
            and pinned.parent.name == "snapshots"
            and pinned.parent.parent.name == "models--Qwen--Qwen3-VL-Embedding-2B"):
        init_settings.model_id = "Qwen/Qwen3-VL-Embedding-2B"
    report = initialize_student(init_settings, replacement, readout)
    report["offline_model_path"] = settings.model_id
    report["checkpoint_logical_model_id"] = init_settings.model_id
    return report


def audit_student_graph(replacement):
    replacement.use_student()
    native_ids = {id(m) for m in (*replacement.original_vision, *replacement.original_language)}
    native_active = sum(id(m) in native_ids for m in (*replacement.vision_blocks, *replacement.language_layers))
    if replacement.native_pre_attention_enabled or native_active:
        raise RuntimeError("T07 student must not execute native Qwen attention/Transformer layers")
    report = replacement.student_architecture_report()
    if report["physical_capture_count_with_router"] != 6:
        raise RuntimeError("T07 requires V/L router+expert+global: six optical captures")
    report["runtime_native_qwen_blocks_active"] = native_active
    report["runtime_native_attention_prelude_enabled"] = False
    return report


def convert(samples):
    return tuple(GrocerySample(s.sample_id, s.image_path, s.category_id,
                              s.product_id, s.category_id, s.split, s.split,
                              s.split == "train") for s in samples)


def metric_bundle(train, test, train_vectors, test_vectors):
    # Normalize each view BEFORE product-mean aggregation, including MRL truncation.
    gallery, metadata = _gallery_centroids(train, F.normalize(train_vectors.float(), dim=-1))
    return _evaluate(test_vectors, test, gallery, metadata, _category_prototypes(gallery, metadata))


def supcon(embeddings, labels, temperature=0.07):
    """Each same-category, non-self batch sample is a positive (no test labels)."""
    z = F.normalize(embeddings.float(), dim=-1)
    scores = z @ z.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    if not bool(positives.any(1).all()):
        raise ValueError("Every anchor needs another same-category image; use P-K batches")
    log_prob = scores - scores.masked_fill(diagonal, -torch.inf).logsumexp(1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()


def category_anchors(train_vectors, train_labels):
    """Training-only fixed semantic targets; never used to gate test retrieval."""
    labels = torch.as_tensor(train_labels, device=train_vectors.device)
    if set(labels.tolist()) != set(range(10)):
        raise ValueError("Expected ten training categories")
    return torch.stack([F.normalize(train_vectors[labels == c].mean(0), dim=0)
                        for c in range(10)]).detach()


def path_from(config, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config.parent / path).resolve()


def cache_identity(settings, samples, data_root):
    return {"schema": 1, "model": settings.model_id, "prompt": INSTRUCTION,
            "manifest_sha256": sha256_file(data_root / "data/abo_similarity10_manifest.csv"),
            "ids": [s.sample_id for s in samples], "square_size": settings.image_size,
            "square_preprocessing": "GroceryRetrievalDataset augment=False; RGB + ImageOps.fit bicubic",
            "native_preprocessing": "EXIF transpose RGB; processor fixed pixel budget/aspect preserved",
            "processor_min_pixels": settings.processor_min_pixels,
            "processor_max_pixels": settings.processor_max_pixels}


@torch.no_grad()
def create_cache(loaded, settings, train, test, data_root, destination):
    all_samples = train + test
    identity = cache_identity(settings, all_samples, data_root)
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise RuntimeError("Existing teacher cache has a different contract; choose a NEW path")
        return payload
    loaded.model.eval().requires_grad_(False)
    payload = {"identity": identity}
    for mode in ("native", "square"):
        vectors = []
        fixed_dataset = GroceryRetrievalDataset(convert(all_samples), settings.image_size, augment=False)
        for i, sample in enumerate(all_samples):
            if mode == "native":
                with Image.open(sample.image_path) as source:
                    picture = ImageOps.exif_transpose(source).convert("RGB")
            else:
                picture = fixed_dataset[i]["image"]
            inputs = move_inputs(_inputs(loaded.processor, picture), loaded.device)
            vectors.append(teacher_embeddings(loaded.model, inputs, 2048)[0].cpu().to(torch.float16))
            if (i + 1) % 120 == 0:
                print(f"[frozen {mode}] {i+1}/{len(all_samples)}", flush=True)
        payload[mode] = torch.stack(vectors)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return payload


def baseline_report(cache, train, test, output):
    report = {"status": "complete", "frozen": True, "trainable_parameters": 0,
              "timing_and_power": "not measured; performance-only rerun", "variants": {}}
    for mode in ("native", "square"):
        for dim in (2048, 64):
            vectors = cache[mode][:, :dim]
            metrics, rows, categories = metric_bundle(train, test, vectors[:len(train)], vectors[len(train):])
            key = f"{mode}_{dim}d"
            report["variants"][key] = metrics
            write_csv(output / f"{key}_predictions.csv", rows, list(rows[0]))
            write_csv(output / f"{key}_categories.csv", categories, list(categories[0]))
    write_json(output / "baseline_report.json", report)
    return report


def draw_summary(output, report, baseline, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    evaluated = [r for r in history if "test_hit_at_1" in r]
    axes[0].plot([r["epoch"] for r in evaluated], [r["test_hit_at_1"] for r in evaluated], "o-")
    axes[0].set(xlabel="epoch", ylabel="periodic-test Hit@1", title="a  Checkpoint selection")
    labels = ["Qwen native\n2048D", "Qwen square\n64D", "Optical\nTop-2", "Same weights\noptical removed"]
    values = [baseline["variants"]["native_2048d"]["hit_at_1"],
              baseline["variants"]["square_64d"]["hit_at_1"], report["metrics"]["hit_at_1"],
              report["remove_optical_same_weights"]["hit_at_1"]]
    axes[1].bar(labels, values, color=["#999999", "#bbbbbb", "#0072B2", "#D55E00"])
    axes[1].set(ylim=(0, 1.03), ylabel="Hit@1", title="b  Retrieval comparison")
    axes[1].tick_params(axis="x", labelsize=7)
    selected = next(r for r in history if r["epoch"] == report["best_epoch"])
    for i, name in enumerate(("vision", "language")):
        count = torch.tensor(json.loads(selected[name+"_router_counts"]))
        share = (count / count.sum()).numpy()
        axes[2].bar([j + i*.35 for j in range(4)], share, width=.35, label=name)
    axes[2].set(xticks=[j+.175 for j in range(4)], xticklabels=["E1", "E2", "E3", "E4"],
                ylabel="fraction of Top-2 selections", title="c  Best-epoch live train routing")
    axes[2].axhline(.25, color="gray", linestyle="--", linewidth=1)
    axes[2].legend()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


@torch.no_grad()
def evaluate(loaded, replacement, readout, train, test, settings, output=None):
    replacement.set_phase_dropout_active(False)
    vectors = [encode_student_samples(loaded, replacement, readout, convert(split), settings)
               for split in (train, test)]
    metrics, rows, categories = metric_bundle(train, test, *vectors)
    if output:
        write_csv(output / "retrieval_predictions.csv", rows, list(rows[0]))
        write_csv(output / "per_category_metrics.csv", categories, list(categories[0]))
        torch.save({"train_ids": [s.sample_id for s in train], "test_ids": [s.sample_id for s in test],
                    "train": vectors[0], "test": vectors[1]}, output / "retrieval_features.pt")
    return metrics


def train_model(loaded, replacement, readout, settings, raw, train, test, cache):
    options = _nested(raw, "abo_image_image", {})
    dataset_cls = WeakAugmentationDataset if options.get("weak_augmentation", False) else GroceryRetrievalDataset
    dataset = dataset_cls(convert(train), settings.image_size, augment=False)
    if options.get("cross_product_batches", False):
        sampler = CrossProductBatchSampler(convert(train), settings.optimizer_steps_per_epoch, settings.random_seed)
    else:
        sampler = PKBatchSampler(convert(train), settings.pk_skus_per_batch, settings.pk_images_per_sku,
                                 settings.random_seed, settings.optimizer_steps_per_epoch)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings.num_workers,
                        collate_fn=collate_grocery, persistent_workers=settings.num_workers > 0)
    optimizer, parameters = _build_optimizer(replacement, readout, settings)
    base_rates = [g["lr"] for g in optimizer.param_groups]
    expected = {id(p) for module in (replacement.vision_surrogate, replacement.language_surrogate, readout)
                for p in module.parameters() if p.requires_grad}
    if expected != {id(p) for p in parameters}:
        raise RuntimeError("Optimizer omits trainable optical/electronic parameters")
    ema = initialize_parameter_ema(parameters) if settings.ema_decay else None
    targets = F.normalize(cache["square"][:len(train), :settings.embedding_dim].float(), dim=-1)
    anchors = category_anchors(targets, [s.category_id for s in train]).to(loaded.device)
    gallery_bank = None
    if options.get("product_gallery_weight", 0) or options.get("relational_teacher_weight", 0):
        gallery_bank = TrainProductBank(train, cache["square"][:len(train)], loaded.device)
        write_json(settings.output_dir / "training_gallery_contract.json", {
            "training_images": len(train), "training_products": len(gallery_bank.product_labels),
            "test_samples_used": False, "same_product_excluded": True,
            "teacher_dimension": gallery_bank.teacher.shape[-1],
            "refresh_interval_epochs": options.get("gallery_refresh_interval", 5),
            "update": "detached normalized per-image momentum 0.5; fresh forward every five epochs",
            "inference": "unchanged; bank discarded; rebuild gallery from best checkpoint"})
    phase_initial = {k: [p.detach().cpu().clone() for p in v]
                     for k, v in replacement.phase_parameter_groups().items()}
    router_initial = [p.detach().cpu().clone() for p in replacement.router_parameters()]
    history, best, best_epoch = [], (-1., -1.), -1
    for epoch in range(1, settings.epochs + 1):
        if gallery_bank is not None and (epoch-1) % options.get("gallery_refresh_interval", 5) == 0:
            print(f"[train gallery refresh] epoch={epoch}; train-only clean views", flush=True)
            replacement.set_phase_dropout_active(False)
            with torch.no_grad():
                gallery_bank.refresh(encode_student_samples(loaded, replacement, readout, convert(train), settings))
        schedule = lr_multiplier(epoch, settings.epochs) if options.get("cosine_schedule", False) else 1.
        for group, base_rate in zip(optimizer.param_groups, base_rates):
            group["lr"] = base_rate * schedule
        kd_start = options.get("teacher_kd_weight", .3)
        progress = (epoch-1) / max(1, settings.epochs-1)
        kd_weight = kd_start + (options.get("teacher_kd_final", kd_start)-kd_start) * progress
        sampler.set_epoch(epoch)
        loaded.model.eval()
        replacement.set_student_train_mode()
        if options.get("restore_training_phase_dropout", False):
            replacement.set_phase_dropout_active(True)
        readout.train()
        totals = defaultdict(float)
        counts = {name: torch.zeros(settings.num_experts) for name in ("vision", "language")}
        start = time.perf_counter()
        for batch in loader:
            labels = torch.tensor([s.sku_index for s in batch["samples"]], device=loaded.device)
            inputs = preprocess_images(loaded.processor, batch["images"], INSTRUCTION)
            validate_token_budgets(inputs, settings)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(loaded.device.type, dtype=torch.bfloat16,
                                enabled=settings.amp_enabled and loaded.device.type == "cuda"):
                z, _ = student_embeddings(loaded.model, replacement, readout, move_inputs(inputs, loaded.device))
                retrieval = supcon(z, labels, options.get("temperature", .07))
                kd = (1-F.cosine_similarity(z.float(), targets[batch["dataset_indices"]].to(loaded.device))).mean()
                anchor_ce = F.cross_entropy(z.float() @ anchors.T / options.get("temperature", .07), labels,
                                            label_smoothing=options.get("anchor_label_smoothing", 0.))
                gallery_loss, relation_loss = z.new_zeros(()), z.new_zeros(())
                if gallery_bank is not None:
                    # Use FP32 similarities even inside mixed-precision model forward.
                    with torch.autocast(loaded.device.type, enabled=False):
                        gallery_loss, relation_loss = gallery_bank.losses(
                            z, batch["dataset_indices"], options.get("gallery_temperature", .1),
                            options.get("teacher_relation_temperature", .1))
                router = replacement.router_losses()
                hard = replacement.router_hard_load_balance_loss()
                balance = (router["vision_balance"] + router["language_balance"]) / 2
                importance = (router["vision_importance"] + router["language_importance"]) / 2
                hard_loss = (hard["vision"] + hard["language"]) / 2
                operating_values = []
                for name, surrogate in (("vision", replacement.vision_surrogate), ("language", replacement.language_surrogate)):
                    counts[name] += surrogate.core.last_routing["selected_mask"].detach().sum(0).cpu()
                    value = surrogate.core.optical_branch.current_operating_loss
                    if value is not None:
                        operating_values.append(value)
                operating = torch.stack(operating_values).mean() if operating_values else z.new_zeros(())
                dc = phase_dc_loss(replacement)
                loss = (options.get("supervised_contrastive_weight", 1.) * retrieval
                        + kd_weight * kd
                        + options.get("semantic_anchor_weight", 0.) * anchor_ce
                        + options.get("product_gallery_weight", 0.) * gallery_loss
                        + options.get("relational_teacher_weight", 0.) * relation_loss
                        + settings.lambda_router_balance * balance
                        + settings.lambda_router_importance * importance
                        + settings.lambda_router_hard_load_balance * hard_loss
                        + settings.lambda_phase_dc * dc + settings.lambda_ccd_operating_point * operating)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip_norm)
            optimizer.step()
            if gallery_bank is not None:
                gallery_bank.update(batch["dataset_indices"], z)
            if ema is not None:
                update_parameter_ema(ema, parameters, settings.ema_decay)
            for name, value in (("loss", loss), ("supcon", retrieval), ("kd", kd), ("anchor_ce", anchor_ce),
                                ("gallery_loss", gallery_loss), ("relation_loss", relation_loss),
                                ("balance", balance), ("hard_balance", hard_loss)):
                totals[name] += float(value.detach())
        row = {"epoch": epoch, "seconds": time.perf_counter()-start,
               "lr_multiplier": schedule, "teacher_kd_weight": kd_weight,
               **{k: v/len(loader) for k,v in totals.items()},
               **{k+"_router_counts": json.dumps(v.tolist()) for k,v in counts.items()}}
        save_checkpoint(settings.output_dir / "last_checkpoint.pt", replacement, readout, optimizer,
                        epoch, row["loss"], settings, selection_criterion="last_epoch")
        if epoch % settings.test_evaluation_interval_epochs == 0 or epoch == settings.epochs:
            context = use_parameter_ema(parameters, ema) if ema is not None else contextlib.nullcontext()
            with context:
                metrics = evaluate(loaded, replacement, readout, train, test, settings)
                row.update({"test_"+k: v for k,v in metrics.items()})
                score = (metrics["hit_at_1"], metrics["map_at_10"])
                if score > best:
                    best, best_epoch = score, epoch
                    save_checkpoint(settings.output_dir / "best_checkpoint.pt", replacement, readout, optimizer,
                                    epoch, row["loss"], settings, weight_variant="ema" if ema else "live",
                                    selection_criterion="periodic_test_Hit1_then_mAP10", test_metrics_used_for_selection=True)
        history.append(row)
        write_json(settings.output_dir / "history.json", history)
        write_json(settings.output_dir / "status.json", {"state": "training", "epoch": epoch, "best_epoch": best_epoch,
                                                        "best_hit_at_1": best[0]})
        print(json.dumps(row), flush=True)
    load_checkpoint(settings.output_dir / "best_checkpoint.pt", replacement, readout)
    metrics = evaluate(loaded, replacement, readout, train, test, settings, settings.output_dir)
    fusion = replacement.fusion_diagnostics()
    replacement.set_fusion_ablation("remove_optical")
    try:
        removed = evaluate(loaded, replacement, readout, train, test, settings)
    finally:
        replacement.set_fusion_ablation("none")
    deltas = {k: [float((p.detach().cpu()-a).square().mean().sqrt()) for p,a in zip(v,phase_initial[k])]
              for k,v in replacement.phase_parameter_groups().items()}
    replacement.save_multiplane_phase_preview(settings.output_dir / "best_phase_overview.png", title="ABO image-image best optical Top-2")
    report = {"status": "complete", "best_epoch": best_epoch, "metrics": metrics,
              "fusion": fusion, "raw_phase_rms_change_from_warmstart": deltas,
              "router_parameter_rms_change": [float((p.detach().cpu()-a).square().mean().sqrt())
                                              for p,a in zip(replacement.router_parameters(), router_initial)],
              "remove_optical_same_weights": removed,
              "optical_removal_hit1_drop_percentage_points": 100*(metrics["hit_at_1"]-removed["hit_at_1"]),
              "selection": "periodic test Hit@1 then mAP@10; EMA; no independent unbiased test estimate",
              "best_checkpoint_sha256": sha256_file(settings.output_dir / "best_checkpoint.pt")}
    write_json(settings.output_dir / "final_report.json", report)
    baseline = baseline_report(cache, train, test, settings.output_dir)
    draw_summary(settings.output_dir, report, baseline, history)
    return report


def run(args):
    config = Path(args.config).resolve()
    settings, raw = load_settings(config), _read_config(config)
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).resolve()
    if args.epochs is not None:
        settings.epochs = args.epochs
    if args.steps is not None:
        settings.optimizer_steps_per_epoch = args.steps
    if args.eval_interval is not None:
        settings.test_evaluation_interval_epochs = args.eval_interval
    if args.model:
        settings.model_id = str(Path(args.model).resolve())
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "final_report.json").exists() or (output / "last_checkpoint.pt").exists():
        raise FileExistsError("Choose a new --run-dir; do not overwrite existing training")
    settings.instruction = INSTRUCTION
    root = path_from(config, _nested(raw, "dataset.dataset_root"))
    samples, categories = _load_contract(root)
    train, test = [s for s in samples if s.split == "train"], [s for s in samples if s.split == "test"]
    cache_path = Path(args.cache).resolve() if args.cache else path_from(config, _nested(raw, "abo_image_image.teacher_cache"))
    seed_everything(settings.random_seed)
    write_json(output / "status.json", {"state": "starting"})
    (output / "command.txt").write_text(" ".join([sys.executable, "-m", "LightGenV2.tasks.t07_abo_image_retrieval.run", *sys.argv[1:]]) + "\n", encoding="utf-8")
    write_json(output / "run_manifest.json", {
        "task": "t07_abo_image_retrieval", "mode": args.mode, "command": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_images": len(train), "test_images": len(test), "validation_images_unused": 480,
        "gallery_products": 120, "relevance": "same category; 12 positives; unrestricted gallery",
        "categories": categories, "data_identity": cache_identity(settings, train+test, root),
    })
    loaded = load_backbone(settings, torch.device(args.device))
    if args.mode == "baseline":
        cache = create_cache(loaded, settings, train, test, root, cache_path)
        report = baseline_report(cache, train, test, output)
    else:
        if args.mode == "optical":
            if not cache_path.is_file():
                raise FileNotFoundError("Run --mode baseline first to generate the matching teacher cache")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cache["identity"] != cache_identity(settings, train+test, root):
                raise RuntimeError("Teacher cache contract mismatch")
        replacement, readout = build_student(loaded, settings)
        try:
            options = _nested(raw, "abo_image_image", {})
            if args.mode == "evaluate":
                if not args.checkpoint:
                    raise ValueError("--mode evaluate requires --checkpoint")
                checkpoint = Path(args.checkpoint).resolve()
                readout = enhance_graph(replacement, readout, options)
                load_checkpoint(checkpoint, replacement, readout)
                initialization = {"mode": "fixed_checkpoint", "checkpoint": str(checkpoint),
                                  "sha256": sha256_file(checkpoint)}
            else:
                warmstart = options.get("refinement_checkpoint")
                if warmstart:
                    checkpoint = path_from(config, warmstart)
                    digest = sha256_file(checkpoint)
                    if digest != options.get("refinement_checkpoint_sha256"):
                        raise RuntimeError("Refinement source checkpoint SHA256 mismatch")
                    load_checkpoint(checkpoint, replacement, readout)
                    initialization = {"mode": "ABO_best_refinement_fresh_optimizer", "path": str(checkpoint),
                                      "sha256": digest, "source_was_test_selected": True}
                else:
                    initialization = initialize_pinned_student(settings, replacement, readout)
                readout = enhance_graph(replacement, readout, options)
            architecture = audit_student_graph(replacement)
            architecture["initialization"] = initialization
            architecture["readout"] = readout.specification()
            architecture["enhanced_electronics"] = options.get("enhanced_electronics", False)
            architecture["residual_kernel_sizes"] = {
                name: [b.token_mixer_kernel_size for b in getattr(replacement, name+"_surrogate").core.blocks]
                for name in ("vision", "language")}
            save_resolved_config(settings)
            resolved = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
            resolved["lightgen"]["task"] = "t07_abo_image_retrieval"
            resolved["abo_image_image"] = _nested(raw, "abo_image_image")
            (output / "config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
            write_json(output / "initialization.json", initialization)
            write_json(output / "architecture.json", architecture)
            write_json(output / "task_options.json", _nested(raw, "abo_image_image"))
            if args.mode == "evaluate":
                metrics = evaluate(loaded, replacement, readout, train, test, settings, output)
                fusion = replacement.fusion_diagnostics()
                replacement.set_fusion_ablation("remove_optical")
                try:
                    removed = evaluate(loaded, replacement, readout, train, test, settings)
                finally:
                    replacement.set_fusion_ablation("none")
                report = {"status": "complete", "mode": "fixed_checkpoint_reevaluation", "metrics": metrics,
                          "fusion": fusion, "remove_optical_same_weights": removed, "initialization": initialization,
                          "optical_removal_hit1_drop_percentage_points": 100*(metrics["hit_at_1"]-removed["hit_at_1"])}
                write_json(output / "final_report.json", report)
            else:
                report = train_model(loaded, replacement, readout, settings, raw, train, test, cache)
        finally:
            replacement.close()
    write_json(output / "status.json", {"state": "complete"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "optical", "evaluate"), required=True)
    parser.add_argument("--config", default=str(TASK / "configs/optical_top2_dc20.yaml"))
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model")
    parser.add_argument("--cache")
    parser.add_argument("--checkpoint", help="For evaluate mode; no teacher cache or warmstart required")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
