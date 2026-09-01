from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import load_imagenet
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.settings import (
    load_settings as load_imagenet_settings,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.teacher_cache import (
    ClipFeatureStore,
    DistillationViewDataset,
)

from .deployment_robustness import (
    DeploymentCondition,
    build_differentiable_deployment_state,
)
from .formal_settings import load_formal_settings
from .general_backbone_pretraining import (
    CompactOpticalImageNetStudent,
    SubsetEpochViewSampler,
    load_p06_settings,
    make_loader,
    sha256_file,
    stratified_base_indices,
    write_json,
)


def load_settings(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "pretraining_config",
        "checkpoint",
        "checkpoint_sha256",
        "output_dir",
        "deployment_seed",
        "validation_samples_per_class",
        "batch_size",
        "conditions",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing robustness settings: {missing}")
    conditions = [DeploymentCondition(**item) for item in payload["conditions"]]
    for condition in conditions:
        condition.validate()
    names = [condition.name for condition in conditions]
    if not names or names[0] != "ideal" or len(names) != len(set(names)):
        raise ValueError("Conditions must be unique and begin with ideal")
    payload["conditions"] = conditions
    return payload


def build_model(pretraining_config: Path, checkpoint: Path, checksum: str, device: torch.device):
    settings = load_p06_settings(pretraining_config)
    architecture = load_formal_settings(settings.architecture_config).base
    model = CompactOpticalImageNetStudent(
        architecture.optical,
        selected_stage_indices=settings.model.selected_stage_indices,
        pool_size=settings.model.pool_size,
        projection_dim=settings.model.projection_dim,
        num_classes=settings.model.num_classes,
        source_num_classes=architecture.num_classes,
        classifier_mode=settings.model.classifier_mode,
        classifier_hidden_dim=settings.model.classifier_hidden_dim,
        classifier_dropout=settings.model.classifier_dropout,
    )
    model.load_source(
        settings.source_checkpoint,
        settings.source_checkpoint_sha256,
        load_mode=settings.source_checkpoint_load_mode,
    )
    report = model.load_pretraining_checkpoint(checkpoint, checksum, load_mode="strict")
    return model.to(device).eval(), settings, report


@torch.inference_mode()
def evaluate(model, loader, condition, deployment_seed: int, device: torch.device, max_batches):
    deployment, metadata = build_differentiable_deployment_state(
        model.encoder,
        condition,
        deployment_seed=deployment_seed,
        device=device,
    )
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        teacher = F.normalize(
            batch["teacher_embedding"].to(device, non_blocking=True).float(), dim=-1
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, embedding, _ = model(images, deployment=deployment)
        top5 = logits.float().topk(5, dim=-1).indices
        count = labels.numel()
        totals[0] += count
        totals[1] += F.cross_entropy(logits.float(), labels, reduction="sum").double()
        totals[2] += logits.argmax(-1).eq(labels).sum()
        totals[3] += top5.eq(labels[:, None]).any(-1).sum()
        totals[4] += (embedding.float() * teacher).sum()
        totals[5] += 1
        if batch_index % 250 == 0:
            print(
                f"[robustness] condition={condition.name} batch={batch_index}/{len(loader)}",
                flush=True,
            )
    count = max(float(totals[0]), 1.0)
    return {
        "condition": asdict(condition),
        "deployment": metadata,
        "samples": int(totals[0].item()),
        "loss_ce": float(totals[1] / count),
        "top1_accuracy": float(totals[2] / count),
        "top5_accuracy": float(totals[3] / count),
        "clip_cosine": float(totals[4] / count),
        "batches": int(totals[5].item()),
        "seconds": time.perf_counter() - started,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_settings(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(config["checkpoint"])
    expected = str(config["checkpoint_sha256"])
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise RuntimeError(f"Checkpoint checksum mismatch: expected {expected}, got {actual}")
    model, settings, load_report = build_model(
        Path(config["pretraining_config"]), checkpoint, expected, device
    )
    imagenet_settings = load_imagenet_settings(settings.imagenet_config)
    bundle = load_imagenet(imagenet_settings)
    store = ClipFeatureStore("validation", bundle.validation, bundle, imagenet_settings)
    dataset = DistillationViewDataset(bundle.validation, store)
    indices = stratified_base_indices(
        bundle.validation.targets,
        int(config["validation_samples_per_class"]),
        settings.training.seed + 1,
    )
    sampler = SubsetEpochViewSampler(
        bundle.validation,
        indices,
        shuffle=False,
        seed=settings.training.seed,
    )
    loader = make_loader(
        dataset,
        sampler,
        batch_size=int(config["batch_size"]),
        settings=settings,
    )
    rows = []
    for condition in config["conditions"]:
        row = evaluate(
            model,
            loader,
            condition,
            int(config["deployment_seed"]),
            device,
            config.get("max_validation_batches"),
        )
        rows.append(row)
        print(
            f"[condition] {condition.name} top1={row['top1_accuracy']:.4f} "
            f"top5={row['top5_accuracy']:.4f} cosine={row['clip_cosine']:.4f}",
            flush=True,
        )
    ideal = rows[0]["top1_accuracy"]
    for row in rows:
        row["top1_absolute_drop"] = ideal - row["top1_accuracy"]
        row["top1_relative_drop"] = (
            0.0 if ideal <= 0 else 1.0 - row["top1_accuracy"] / ideal
        )
    result = {
        "status": "complete",
        "config": str(config_path),
        "pretraining_config": str(config["pretraining_config"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual,
        "checkpoint_load": load_report,
        "deployment_seed": int(config["deployment_seed"]),
        "validation_samples": len(indices),
        "conditions": rows,
    }
    output_dir = Path(config["output_dir"])
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate P06 backbone under fixed optical shifts")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
