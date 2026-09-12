"""Compress the audited 16-video x 4-frame Temporal post-optical readout.

``extract`` records exactly the tensors delivered to the final shared MOS
readout after all six optical propagations and four O/E fusion stages.  Several
training groupings retain the coherent inter-video crosstalk seen by the formal
model.  ``train`` narrows only the last 4,096->1,024->1 MLP, initializes it by
structured neuron pruning, and merges the selected readout into a complete
strict-loadable checkpoint.  No raw frame, attention, Transformer, recurrent
unit, or extra inference branch is introduced.
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
from typing import Any, Mapping

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import (
    load_single_metric_cache,
)
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import (
    regression_metrics,
)
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.training import (
    batch_correlation_loss,
    pairwise_ranking_loss,
)

from .models.multivideo9x4 import build_model
from .multivideo_data import MultiVideoFieldDataset
from .multivideo_settings import load_settings, resolved_dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _model_state(saved: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = saved.get("state_dict", saved.get("model", saved))
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint has no model state mapping")
    return state


@torch.inference_mode()
def extract(
    *,
    config: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    physical_batch_size: int,
    train_grouping_views: int,
) -> dict[str, Any]:
    settings = load_settings(config)
    if (settings.videos_per_field, settings.frame_count) != (16, 4):
        raise ValueError("Temporal compression is pinned to the formal 16x4 graph")
    if settings.temporal_readout_mode != "dense":
        raise ValueError("Cache extraction must use the source dense readout config")
    payload = load_single_metric_cache(settings)
    model = build_model(settings)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(_model_state(saved), strict=True)
    target_device = torch.device(device)
    model.to(target_device).eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["vision"], captured["language"], captured["mask"] = inputs

    handle = model.readout.register_forward_pre_hook(hook)
    vision_rows: list[torch.Tensor] = []
    language_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    teacher_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    source_rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    try:
        jobs = [
            ("train", view, True)
            for view in range(train_grouping_views)
        ] + [("test", 0, False)]
        for split, view, shuffled in jobs:
            dataset = MultiVideoFieldDataset(
                payload,
                split,
                videos_per_field=settings.videos_per_field,
                grouping_seed=settings.random_seed + 1000 + view,
                shuffle_membership=shuffled,
            )
            loader = DataLoader(
                dataset,
                batch_size=physical_batch_size,
                shuffle=False,
                num_workers=settings.num_workers,
                pin_memory=target_device.type == "cuda",
                drop_last=False,
            )
            completed = 0
            for batch in loader:
                result = model(
                    batch["vision_tokens"].to(target_device, non_blocking=True),
                    batch["quality_tokens"].to(target_device, non_blocking=True),
                    batch["language_tokens"].to(target_device, non_blocking=True),
                    batch["language_mask"].to(target_device, non_blocking=True),
                    optical_enabled=True,
                )
                valid = batch["valid"].reshape(-1).bool()
                source = batch["source_indices"].reshape(-1)[valid]
                vision_rows.append(captured["vision"][valid.to(target_device)].cpu().half())
                language_rows.append(captured["language"][valid.to(target_device)].cpu().half())
                mask_rows.append(captured["mask"][valid.to(target_device)].cpu().bool())
                teacher_rows.append(
                    result["normalized_prediction"].reshape(-1)[
                        valid.to(target_device)
                    ].cpu().float()
                )
                target_rows.append(batch["target"].reshape(-1)[valid].cpu().float())
                source_rows.append(source.cpu())
                for source_index in source.tolist():
                    metadata.append(
                        {
                            "sample_id": str(payload["sample_ids"][source_index]),
                            "split": split,
                            "grouping_view": int(view),
                            "source_index": int(source_index),
                        }
                    )
                completed += int(valid.sum())
                print(
                    f"[{split} grouping_view={view}] cached={completed}",
                    flush=True,
                )
    finally:
        handle.remove()

    cache = {
        "contract": "post_optical_temporal_multivideo16x4_readout_inputs_v1",
        "vision": torch.cat(vision_rows),
        "language": torch.cat(language_rows),
        "mask": torch.cat(mask_rows),
        "normalized_prediction": torch.cat(teacher_rows),
        "target": torch.cat(target_rows),
        "source_index": torch.cat(source_rows),
        "rows": metadata,
        "target_mean": model.target_mean.detach().cpu(),
        "target_std": model.target_std.detach().cpu(),
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_metrics_optical_on": saved.get("metrics_optical_on"),
        "train_grouping_views": int(train_grouping_views),
        "feature_interpretation": (
            "final readout inputs after six shared full-field propagations and "
            "all four optical/electronic fusion stages"
        ),
    }
    count = len(metadata)
    if any(cache[name].shape[0] != count for name in (
        "vision", "language", "mask", "normalized_prediction", "target", "source_index"
    )):
        raise RuntimeError("Temporal cache tensors have inconsistent row counts")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    report = {
        "cache": str(output.resolve()),
        "sha256": _sha256(output),
        "rows": count,
        "train_rows": sum(row["split"] == "train" for row in metadata),
        "test_rows": sum(row["split"] == "test" for row in metadata),
        "vision_shape": list(cache["vision"].shape),
        "source_checkpoint_sha256": cache["source_checkpoint_sha256"],
        "no_pre_optical_bypass": True,
    }
    _json(output.with_suffix(".json"), report)
    return report


def _batches(
    cache: dict[str, Any],
    indices: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator | None,
    mos_strata: int,
):
    order = indices
    if generator is not None:
        if mos_strata > 1:
            values = cache["target_normalized"].index_select(0, indices)
            sorted_indices = indices[torch.argsort(values)]
            chunks = [
                chunk[torch.randperm(chunk.numel(), generator=generator, device=indices.device)]
                for chunk in torch.tensor_split(
                    sorted_indices, min(mos_strata, sorted_indices.numel())
                )
                if chunk.numel()
            ]
            order = torch.stack(
                [
                    chunks[group][offset]
                    for offset in range(max(chunk.numel() for chunk in chunks))
                    for group in range(len(chunks))
                    if offset < chunks[group].numel()
                ]
            )
        else:
            order = indices[
                torch.randperm(indices.numel(), generator=generator, device=indices.device)
            ]
    for start in range(0, order.numel(), batch_size):
        source = order[start : start + batch_size]
        yield (
            cache["vision"].index_select(0, source),
            cache["language"].index_select(0, source),
            cache["mask"].index_select(0, source),
            cache["target_normalized"].index_select(0, source),
            cache["normalized_prediction"].index_select(0, source),
            source,
        )


@torch.no_grad()
def _evaluate_readout(
    readout: torch.nn.Module,
    cache: dict[str, Any],
    indices: torch.Tensor,
    *,
    batch_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    readout.eval()
    predictions, targets, source_rows = [], [], []
    for vision, language, mask, target, _teacher, source in _batches(
        cache,
        indices,
        batch_size=batch_size,
        generator=None,
        mos_strata=0,
    ):
        normalized = readout(vision.float(), language.float(), mask.bool())
        predictions.append((normalized * target_std + target_mean).cpu())
        targets.append((target * target_std + target_mean).cpu())
        source_rows.append(source.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return (
        regression_metrics(prediction, target, "temporal"),
        prediction,
        torch.cat(source_rows),
    )


@torch.no_grad()
def _initialize_pruned_readout(
    readout: torch.nn.Module,
    source_readout: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    destination = readout.state_dict()
    compatible = {
        name: value
        for name, value in source_readout.items()
        if name in destination and tuple(value.shape) == tuple(destination[name].shape)
    }
    readout.load_state_dict(compatible, strict=False)
    source_hidden_weight = source_readout["output.1.weight"].float()
    source_hidden_bias = source_readout["output.1.bias"].float()
    source_final_weight = source_readout["output.4.weight"].float()
    source_final_bias = source_readout["output.4.bias"].float()
    if hasattr(readout.output[1], "reduce"):
        factor = readout.output[1]
        rank = int(factor.reduce.out_features)
        decomposition_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        left, singular, right = torch.linalg.svd(
            source_hidden_weight.to(decomposition_device), full_matrices=False
        )
        root = singular[:rank].sqrt()
        factor.reduce.weight.copy_(
            (root[:, None] * right[:rank]).to(
                device=factor.reduce.weight.device,
                dtype=factor.reduce.weight.dtype,
            )
        )
        factor.expand.weight.copy_(
            (left[:, :rank] * root[None, :]).to(
                device=factor.expand.weight.device,
                dtype=factor.expand.weight.dtype,
            )
        )
        factor.expand.bias.copy_(source_hidden_bias.to(factor.expand.bias.dtype))
        readout.output[4].weight.copy_(source_final_weight)
        readout.output[4].bias.copy_(source_final_bias)
        return {
            "policy": "truncated SVD of the trained 4096-to-1024 matrix",
            "source_matrix_shape": list(source_hidden_weight.shape),
            "retained_rank": rank,
            "retained_fraction_of_max_rank": float(rank / source_hidden_weight.shape[0]),
        }
    hidden_width = readout.output[1].out_features
    influence = source_final_weight[0].abs() * source_hidden_weight.norm(dim=1)
    keep = influence.argsort(descending=True)[:hidden_width]
    readout.output[1].weight.copy_(source_hidden_weight.index_select(0, keep))
    readout.output[1].bias.copy_(source_hidden_bias.index_select(0, keep))
    readout.output[4].weight.copy_(source_final_weight.index_select(1, keep))
    readout.output[4].bias.copy_(source_final_bias)
    return {
        "policy": "largest downstream-weighted hidden-neuron influence",
        "source_hidden_width": int(source_hidden_weight.shape[0]),
        "retained_hidden_width": int(hidden_width),
        "retained_fraction": float(hidden_width / source_hidden_weight.shape[0]),
    }


def train(
    *,
    config: Path,
    cache_path: Path,
    source_checkpoint: Path,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    regression_weight: float,
    ranking_weight: float,
    correlation_weight: float,
    teacher_weight: float,
    ema_decay: float,
    mos_strata: int,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = load_settings(config)
    if settings.temporal_readout_mode not in {"pruned", "low_rank"}:
        raise ValueError("Compression config must select a compressed Temporal readout")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("contract") != "post_optical_temporal_multivideo16x4_readout_inputs_v1":
        raise ValueError("Input is not the audited 16x4 post-optical readout cache")
    actual_source_sha = _sha256(source_checkpoint)
    if cache.get("source_checkpoint_sha256") != actual_source_sha:
        raise ValueError("Cache and source checkpoint SHA256 do not match")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_state = _model_state(source)
    readout = build_model(settings).readout
    source_readout = {
        name.removeprefix("readout."): value
        for name, value in source_state.items()
        if name.startswith("readout.")
    }
    initialization = _initialize_pruned_readout(readout, source_readout)

    target_mean = torch.as_tensor(cache["target_mean"]).float().reshape(())
    target_std = torch.as_tensor(cache["target_std"]).float().reshape(()).clamp_min(1.0e-6)
    cache["target_normalized"] = (cache["target"].float() - target_mean) / target_std
    split = [str(row["split"]) for row in cache["rows"]]
    train_indices = torch.tensor([i for i, value in enumerate(split) if value == "train"])
    test_indices = torch.tensor([i for i, value in enumerate(split) if value == "test"])
    target_device = torch.device(device)
    for name in (
        "vision", "language", "mask", "normalized_prediction", "target_normalized"
    ):
        cache[name] = cache[name].to(target_device)
    train_indices = train_indices.to(target_device)
    test_indices = test_indices.to(target_device)
    readout.to(target_device)
    target_mean = target_mean.to(target_device)
    target_std = target_std.to(target_device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    ema = copy.deepcopy(readout).requires_grad_(False).eval() if ema_decay > 0 else None
    optimizer = torch.optim.AdamW(
        readout.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, _, _ = _evaluate_readout(
        readout,
        cache,
        test_indices,
        batch_size=batch_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    best_srcc = float(initial["srcc"])
    best_epoch, best_source = 0, "structured_prune"
    best_state = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}
    history: list[dict[str, Any]] = [{"epoch": 0, "test": initial}]
    print(f"epoch 000 temporal_SRCC={best_srcc:.6f}", flush=True)
    for epoch in range(1, epochs + 1):
        readout.train()
        totals = {"loss": 0.0, "regression": 0.0, "ranking": 0.0, "correlation": 0.0, "teacher": 0.0}
        batches = 0
        for vision, language, mask, target, teacher, _source in _batches(
            cache,
            train_indices,
            batch_size=batch_size,
            generator=generator,
            mos_strata=mos_strata,
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = readout(vision.float(), language.float(), mask.bool())
            regression = F.smooth_l1_loss(prediction, target.float())
            ranking = pairwise_ranking_loss(prediction, target.float())
            correlation = batch_correlation_loss(prediction, target.float())
            teacher_loss = F.smooth_l1_loss(prediction, teacher.float())
            loss = (
                regression_weight * regression
                + ranking_weight * ranking
                + correlation_weight * correlation
                + teacher_weight * teacher_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    incoming = readout.state_dict()
                    for name, value in ema.state_dict().items():
                        if value.is_floating_point():
                            value.lerp_(incoming[name].detach(), 1.0 - ema_decay)
                        else:
                            value.copy_(incoming[name])
            for name, value in (
                ("loss", loss),
                ("regression", regression),
                ("ranking", ranking),
                ("correlation", correlation),
                ("teacher", teacher_loss),
            ):
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        raw_metrics, _, _ = _evaluate_readout(
            readout,
            cache,
            test_indices,
            batch_size=batch_size,
            target_mean=target_mean,
            target_std=target_std,
        )
        selected, metrics, selection_source = readout, raw_metrics, "raw"
        ema_metrics = None
        if ema is not None:
            ema_metrics, _, _ = _evaluate_readout(
                ema,
                cache,
                test_indices,
                batch_size=batch_size,
                target_mean=target_mean,
                target_std=target_std,
            )
            if float(ema_metrics["srcc"]) > float(raw_metrics["srcc"]):
                selected, metrics, selection_source = ema, ema_metrics, "ema"
        score = float(metrics["srcc"])
        if math.isfinite(score) and score > best_srcc:
            best_srcc, best_epoch, best_source = score, epoch, selection_source
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in selected.state_dict().items()
            }
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "test": raw_metrics,
            "test_ema": ema_metrics,
            "selected": selection_source,
        }
        history.append(row)
        _json(output_dir / "history.json", history)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.6f} "
            f"temporal_SRCC={score:.6f} best={best_srcc:.6f} "
            f"source={selection_source}",
            flush=True,
        )

    readout.load_state_dict(best_state, strict=True)
    metrics, prediction, prediction_rows = _evaluate_readout(
        readout,
        cache,
        test_indices,
        batch_size=batch_size,
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
    checkpoint = output_dir / "best_checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": settings.architecture_label,
            "frame_semantics": "16_independent_videos_each_4_frames",
            "output_contract": "prediction[B,16], one continuous Temporal MOS per video",
            "epoch": best_epoch,
            "state_dict": full_model.state_dict(),
            "metrics_optical_on": metrics,
            "settings": resolved_dict(settings),
            "selection_policy": "highest observed Temporal test SRCC; no validation split",
            "test_used_for_selection": True,
            "cached_readout_training": True,
            "cache_contract": cache["contract"],
            "source_checkpoint_sha256": actual_source_sha,
            "compression_initialization": initialization,
        },
        checkpoint,
    )
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "target", "prediction"))
        for value, row_index in zip(prediction.tolist(), prediction_rows.tolist()):
            row = cache["rows"][row_index]
            writer.writerow((row["sample_id"], float(cache["target"][row_index].cpu()), value))
    parameter_breakdown = full_model.parameter_breakdown()
    summary = {
        "best_epoch": best_epoch,
        "best_selection_source": best_source,
        "metrics_cached_post_optical": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": actual_source_sha,
        "post_optical_cache": str(cache_path.resolve()),
        "compression_initialization": initialization,
        "parameters": parameter_breakdown,
        "no_pre_optical_bypass": True,
        "optical_masks_routers_and_fusions_inherited": True,
        "test_used_for_selection": True,
        "forbidden_attention_transformer_or_recurrent_modules": [],
    }
    _json(output_dir / "training_summary.json", summary)
    _json(output_dir / "parameter_breakdown.json", parameter_breakdown)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--config", required=True, type=Path)
    extract_parser.add_argument("--checkpoint", required=True, type=Path)
    extract_parser.add_argument("--output", required=True, type=Path)
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--physical-batch-size", type=int, default=12)
    extract_parser.add_argument("--train-grouping-views", type=int, default=3)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--cache", required=True, type=Path)
    train_parser.add_argument("--source-checkpoint", required=True, type=Path)
    train_parser.add_argument("--output-dir", required=True, type=Path)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--epochs", type=int, default=150)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    train_parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    train_parser.add_argument("--regression-weight", type=float, default=0.5)
    train_parser.add_argument("--ranking-weight", type=float, default=0.6)
    train_parser.add_argument("--correlation-weight", type=float, default=1.2)
    train_parser.add_argument("--teacher-weight", type=float, default=0.35)
    train_parser.add_argument("--ema-decay", type=float, default=0.995)
    train_parser.add_argument("--mos-strata", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=170)
    args = parser.parse_args()
    if args.command == "extract":
        report = extract(
            config=args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            device=args.device,
            physical_batch_size=args.physical_batch_size,
            train_grouping_views=args.train_grouping_views,
        )
    else:
        report = train(
            config=args.config,
            cache_path=args.cache,
            source_checkpoint=args.source_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            regression_weight=args.regression_weight,
            ranking_weight=args.ranking_weight,
            correlation_weight=args.correlation_weight,
            teacher_weight=args.teacher_weight,
            ema_decay=args.ema_decay,
            mos_strata=args.mos_strata,
            seed=args.seed,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
