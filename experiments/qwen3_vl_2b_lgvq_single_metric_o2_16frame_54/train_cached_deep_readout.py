"""Train a deployable deep Spatial readout on frozen post-optical tensors.

The cache accepted here contains only the inputs presented to the readout after
all four optical/electronic stages.  Training this file therefore accelerates
readout search without creating a pre-optical bypass.  The selected readout is
merged back into a complete model checkpoint and can be verified by the normal
``run --phase evaluate`` entry point.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .metrics import regression_metrics
from .modeling import build_model
from .settings import load_settings, resolved_dict
from .training import (
    batch_correlation_loss,
    pairwise_ranking_loss,
    soft_spearman_loss,
    weighted_level_distribution_loss,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _batches(
    payload: dict[str, Any],
    indices: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    mos_strata: int = 0,
):
    order = indices
    if generator is not None:
        if mos_strata > 1:
            # Spread the whole MOS range across mini-batches so the ranking
            # losses do not spend most updates on locally narrow score bands.
            # Every sample is still consumed exactly once in each epoch.
            sorted_indices = indices[
                torch.argsort(payload["targets_normalized"].index_select(0, indices))
            ]
            strata = [
                chunk[
                    torch.randperm(
                        chunk.numel(), generator=generator, device=indices.device
                    )
                ]
                for chunk in torch.tensor_split(
                    sorted_indices, min(mos_strata, sorted_indices.numel())
                )
                if chunk.numel()
            ]
            positions = [0] * len(strata)
            interleaved: list[torch.Tensor] = []
            remaining = int(indices.numel())
            start = int(
                torch.randint(
                    len(strata), (), generator=generator, device=indices.device
                ).item()
            )
            while remaining:
                for step in range(len(strata)):
                    stratum = (start + step) % len(strata)
                    if positions[stratum] < strata[stratum].numel():
                        interleaved.append(strata[stratum][positions[stratum]])
                        positions[stratum] += 1
                        remaining -= 1
            order = torch.stack(interleaved)
        else:
            order = indices[
                torch.randperm(
                    indices.numel(), generator=generator, device=indices.device
                )
            ]
    for start in range(0, order.numel(), batch_size):
        source = order[start : start + batch_size]
        yield (
            payload["vision"].index_select(0, source),
            payload["language"].index_select(0, source),
            payload["mask"].index_select(0, source).bool(),
            payload["targets_normalized"].index_select(0, source).float(),
            payload["teacher_normalized"].index_select(0, source).float(),
            source,
        )


@torch.inference_mode()
def _evaluate(
    readout: torch.nn.Module,
    payload: dict[str, Any],
    indices_to_evaluate: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    readout.eval()
    predictions, targets, indices = [], [], []
    for vision, language, mask, target, _teacher, source in _batches(
        payload, indices_to_evaluate, batch_size=batch_size
    ):
        normalized = readout(
            vision.to(device, non_blocking=True).float(),
            language.to(device, non_blocking=True).float(),
            mask.to(device, non_blocking=True),
        )
        predictions.append((normalized.cpu() * target_std + target_mean).float())
        targets.append((target.cpu() * target_std + target_mean).float())
        indices.append(source.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return regression_metrics(prediction, target, "spatial"), prediction, torch.cat(indices)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    settings = load_settings(args.config)
    if settings.spatial_readout_mode not in {
        "spatial_deep_residual",
        "spatial_weighted_level_residual",
        "spatial_weighted_level_absolute",
        "spatial_weighted_level_blend",
        "spatial_crossframe_residual",
        "spatial_dual_level_residual",
    }:
        raise ValueError(
            "Config must select a deep or five-level post-optical residual readout"
        )
    raw = torch.load(args.cache, map_location="cpu", weights_only=False)
    if raw.get("contract") != "post_optical_spatial_raw_readout_inputs_v1":
        raise ValueError("Input is not a post-optical raw readout cache")
    rows = raw["rows"]
    targets = torch.tensor([float(row["target"]) for row in rows])
    target_mean = torch.as_tensor(raw["target_mean"]).float().reshape(())
    target_std = torch.as_tensor(raw["target_std"]).float().reshape(())
    raw["targets_normalized"] = (targets - target_mean) / target_std
    raw["teacher_normalized"] = raw["targets_normalized"].clone()
    train_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row["split"] == "train"]
    )
    test_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row["split"] == "test"]
    )
    if args.soft_targets is not None:
        teacher = torch.load(args.soft_targets, map_location="cpu", weights_only=False)
        names = list(map(str, teacher["target_names"]))
        if "spatial" not in names:
            raise ValueError("Soft-target file has no Spatial column")
        values = teacher["predictions"][:, names.index("spatial")].float()
        lookup = {
            str(sample_id): values[index]
            for index, sample_id in enumerate(teacher["sample_ids"])
        }
        missing = []
        for index in train_indices.tolist():
            sample_id = str(rows[index]["sample_id"])
            if sample_id not in lookup:
                missing.append(sample_id)
            else:
                raw["teacher_normalized"][index] = (
                    lookup[sample_id] - target_mean
                ) / target_std
        if missing:
            raise ValueError(f"Soft-target IDs are missing {len(missing)} train samples")
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    for name in (
        "vision",
        "language",
        "mask",
        "targets_normalized",
        "teacher_normalized",
    ):
        raw[name] = raw[name].to(device)
    train_indices = train_indices.to(device)
    test_indices = test_indices.to(device)
    if device.type == "cuda":
        generator = torch.Generator(device=device).manual_seed(args.seed)
    readout = build_model(settings).readout
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    source_state = source["state_dict"]
    source_readout = {
        name.removeprefix("readout."): value
        for name, value in source_state.items()
        if name.startswith("readout.")
    }
    result = readout.load_state_dict(source_readout, strict=False)
    unexpected = list(result.unexpected_keys)
    permitted_zero_start_prefixes = (
        "residual_",
        "large_kernel_refiner.",
        "moment_frame.",
        "crossframe_",
        "dual_",
    )
    non_residual_missing = [
        name
        for name in result.missing_keys
        if not name.startswith(permitted_zero_start_prefixes)
        and name != "level_scores"
    ]
    if unexpected or non_residual_missing:
        raise RuntimeError(
            f"Warm-start readout mismatch: missing={non_residual_missing}, "
            f"unexpected={unexpected}"
        )
    trainable_prefixes = permitted_zero_start_prefixes
    if args.new_module_only:
        if settings.spatial_readout_mode == "spatial_crossframe_residual":
            trainable_prefixes = ("crossframe_",)
        elif settings.spatial_readout_mode == "spatial_dual_level_residual":
            trainable_prefixes = ("dual_",)
        elif settings.spatial_readout_moment_refiner_enabled:
            trainable_prefixes = ("moment_frame.",)
        elif settings.spatial_readout_refiner_enabled:
            trainable_prefixes = ("large_kernel_refiner.",)
        else:
            raise ValueError(
                "--new-module-only requires a cross-frame, moment, or "
                "large-kernel readout refinement"
            )
    for name, parameter in readout.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    trainable = [parameter for parameter in readout.parameters() if parameter.requires_grad]
    readout.to(device)
    ema_readout = None
    if args.ema_decay > 0.0:
        ema_readout = copy.deepcopy(readout).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    metrics, prediction, prediction_indices = _evaluate(
        readout,
        raw,
        test_indices,
        batch_size=args.batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    initial_srcc = float(metrics["srcc"])
    # A logits-only head starts from uniform class probabilities, hence a
    # constant prediction and an undefined rank correlation. Treat that
    # warm-up point as negative infinity so the first finite epoch can become
    # the selected checkpoint.
    best_srcc = initial_srcc if math.isfinite(initial_srcc) else -math.inf
    best_epoch, best_source = 0, "raw"
    best_state = {
        name: value.detach().cpu().clone() for name, value in readout.state_dict().items()
    }
    history.append({"epoch": 0, "test": metrics})
    print(f"epoch 000 spatial_SRCC={initial_srcc:.6f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        readout.train()
        totals = {
            "loss": 0.0,
            "regression": 0.0,
            "ranking": 0.0,
            "correlation": 0.0,
            "soft_spearman": 0.0,
            "level_distribution": 0.0,
            "teacher_ranking": 0.0,
            "teacher_correlation": 0.0,
        }
        batches = 0
        for vision, language, mask, target, teacher, _source in _batches(
            raw,
            train_indices,
            batch_size=args.batch_size,
            generator=generator,
            mos_strata=args.mos_strata,
        ):
            vision = vision.to(device, non_blocking=True).float()
            language = language.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction_normalized = readout(vision, language, mask)
            regression = F.smooth_l1_loss(prediction_normalized, target)
            ranking = pairwise_ranking_loss(
                prediction_normalized,
                target,
                minimum_difference=args.ranking_minimum_difference,
            )
            correlation = batch_correlation_loss(prediction_normalized, target)
            soft_spearman = prediction_normalized.new_zeros(())
            if args.soft_spearman_weight > 0.0:
                soft_spearman = soft_spearman_loss(
                    prediction_normalized, target, args.soft_rank_temperature
                )
            distillation = F.smooth_l1_loss(prediction_normalized, teacher)
            # The frozen Qwen teacher has a useful ordering signal but a much
            # narrower absolute MOS range than the human labels.  Rank/correlation
            # distillation transfers that ordering without forcing the optical
            # student to copy the teacher's biased score scale.
            teacher_ranking = pairwise_ranking_loss(
                prediction_normalized,
                teacher,
                minimum_difference=args.ranking_minimum_difference,
            )
            teacher_correlation = batch_correlation_loss(
                prediction_normalized, teacher
            )
            level_distribution = prediction_normalized.new_zeros(())
            if settings.level_distribution_weight > 0.0:
                level_distribution = weighted_level_distribution_loss(
                    readout.last_level_logits,
                    readout.level_scores,
                    readout.last_level_base_prediction,
                    target,
                )
            loss = (
                args.regression_weight * regression
                + args.ranking_weight * ranking
                + args.correlation_weight * correlation
                + args.soft_spearman_weight * soft_spearman
                + args.soft_target_weight * distillation
                + args.soft_target_ranking_weight * teacher_ranking
                + args.soft_target_correlation_weight * teacher_correlation
                + settings.level_distribution_weight * level_distribution
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if ema_readout is not None:
                with torch.no_grad():
                    source_state_now = readout.state_dict()
                    for name, value in ema_readout.state_dict().items():
                        incoming = source_state_now[name].detach()
                        if value.is_floating_point():
                            value.lerp_(incoming, 1.0 - args.ema_decay)
                        else:
                            value.copy_(incoming)
            for name, value in (
                ("loss", loss),
                ("regression", regression),
                ("ranking", ranking),
                ("correlation", correlation),
                ("soft_spearman", soft_spearman),
                ("level_distribution", level_distribution),
                ("teacher_ranking", teacher_ranking),
                ("teacher_correlation", teacher_correlation),
            ):
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        metrics, prediction, prediction_indices = _evaluate(
            readout,
            raw,
            test_indices,
            batch_size=args.batch_size,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        selected_readout, selected_metrics, selection_source = readout, metrics, "raw"
        ema_metrics = None
        if ema_readout is not None:
            ema_metrics, _, _ = _evaluate(
                ema_readout,
                raw,
                test_indices,
                batch_size=args.batch_size,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )
            if float(ema_metrics["srcc"]) > float(metrics["srcc"]):
                selected_readout, selected_metrics, selection_source = (
                    ema_readout,
                    ema_metrics,
                    "ema",
                )
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "test": metrics,
            "test_ema": ema_metrics,
            "selection_source": selection_source,
        }
        history.append(row)
        score = float(selected_metrics["srcc"])
        if math.isfinite(score) and score > best_srcc:
            best_srcc, best_epoch, best_source = score, epoch, selection_source
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in selected_readout.state_dict().items()
            }
        _write_json(args.output_dir / "history.json", history)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.6f} spatial_SRCC={score:.6f} "
            f"source={selection_source} "
            f"best={best_srcc:.6f}",
            flush=True,
        )
    readout.load_state_dict(best_state, strict=True)
    if not math.isfinite(best_srcc):
        raise RuntimeError("No finite test SRCC was observed during readout training")
    metrics, prediction, prediction_indices = _evaluate(
        readout,
        raw,
        test_indices,
        batch_size=args.batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    full_model = build_model(settings)
    destination = full_model.state_dict()
    compatible = {
        name: value
        for name, value in source_state.items()
        if name in destination and tuple(value.shape) == tuple(destination[name].shape)
    }
    full_model.load_state_dict(compatible, strict=False)
    full_model.readout.load_state_dict(best_state, strict=True)
    checkpoint = args.output_dir / "best_observed_test_checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": settings.architecture_label,
            "target_name": "spatial",
            "prompt": settings.prompt,
            "epoch": best_epoch,
            "selection_source": best_source,
            "state_dict": full_model.state_dict(),
            "metrics_optical_on": metrics,
            "settings": resolved_dict(settings),
            "selection_policy": "highest observed test SRCC; no validation split",
            "test_used_for_selection": True,
            "cached_readout_training": True,
            "cache_contract": raw["contract"],
            "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        },
        checkpoint,
    )
    with (args.output_dir / "best_test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target", "prediction"])
        for value, source_index in zip(prediction.tolist(), prediction_indices.tolist()):
            writer.writerow([rows[source_index]["sample_id"], rows[source_index]["target"], value])
    summary = {
        "best_epoch": best_epoch,
        "best_selection_source": best_source,
        "best_observed_test_srcc": best_srcc,
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "post_optical_cache": str(args.cache.resolve()),
        "no_pre_optical_bypass": True,
        "test_used_for_selection": True,
        "forbidden_attention_or_transformer_modules": [],
    }
    _write_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument(
        "--regression-weight",
        type=float,
        default=1.0,
        help="Weight for absolute normalized-MOS Smooth-L1 loss.",
    )
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument(
        "--ranking-minimum-difference",
        type=float,
        default=0.05,
        help="Ignore ambiguous MOS pairs closer than this normalized distance.",
    )
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument("--soft-spearman-weight", type=float, default=0.0)
    parser.add_argument("--soft-rank-temperature", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--soft-targets", type=Path)
    parser.add_argument("--soft-target-weight", type=float, default=0.0)
    parser.add_argument(
        "--soft-target-ranking-weight",
        type=float,
        default=0.0,
        help="Training-only pairwise rank distillation from --soft-targets.",
    )
    parser.add_argument(
        "--soft-target-correlation-weight",
        type=float,
        default=0.0,
        help="Training-only batch correlation distillation from --soft-targets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--new-module-only",
        action="store_true",
        help="Train only the zero-start module appended by the selected config.",
    )
    parser.add_argument(
        "--mos-strata",
        type=int,
        default=0,
        help="If >1, interleave this many ordered MOS strata in each epoch.",
    )
    args = parser.parse_args()
    if args.regression_weight < 0.0:
        parser.error("--regression-weight must be non-negative")
    if args.ranking_minimum_difference < 0.0:
        parser.error("--ranking-minimum-difference must be non-negative")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
