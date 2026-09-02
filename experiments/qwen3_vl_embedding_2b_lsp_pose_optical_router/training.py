from __future__ import annotations

import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    LSPPoseDataset,
    pose_collate,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.training import (
    _print_report,
    _seed_worker,
    _train_epoch,
    _write_model_report,
    evaluate_model,
)

from .modeling import (
    architecture_label,
    build_router_student,
    load_common_initialization,
    sha256_file,
    student_architecture_report,
    trainable_parameter_report,
)
from .protocol import PoseProtocolBundle


class ModelEMA:
    """Collision-free EMA over core and head state dictionaries."""

    def __init__(self, core: nn.Module, head: nn.Module, decay: float) -> None:
        self.modules = {"core": core, "head": head}
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for prefix, module in self.modules.items():
            for name, value in module.state_dict().items():
                if value.is_floating_point():
                    self.shadow[f"{prefix}.{name}"] = value.detach().clone()

    def update(self) -> None:
        with torch.no_grad():
            for prefix, module in self.modules.items():
                for name, value in module.state_dict().items():
                    key = f"{prefix}.{name}"
                    if key in self.shadow:
                        self.shadow[key].mul_(self.decay).add_(
                            value.detach(), alpha=1.0 - self.decay
                        )

    @contextmanager
    def applied(self) -> Iterator[None]:
        saved: dict[str, dict[str, torch.Tensor]] = {}
        try:
            for prefix, module in self.modules.items():
                saved[prefix] = {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }
                module.load_state_dict(
                    {
                        key: self.shadow[f"{prefix}.{key}"]
                        for key in module.state_dict()
                        if f"{prefix}.{key}" in self.shadow
                    },
                    strict=False,
                )
            yield
        finally:
            for prefix, module in self.modules.items():
                module.load_state_dict(saved[prefix], strict=True)


def _loader(
    records: list[Any],
    settings: Any,
    *,
    training: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(settings.random_seed)
    return DataLoader(
        LSPPoseDataset(records, settings, training=training),
        batch_size=(
            settings.student_batch_size
            if training
            else settings.inference_batch_size
        ),
        shuffle=training,
        generator=generator,
        num_workers=settings.num_workers,
        collate_fn=pose_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        worker_init_fn=_seed_worker,
    )


def _optimizer(model: nn.Module, settings: Any) -> torch.optim.Optimizer:
    phase = [
        *model.core.expert_layers.parameters(),
        *model.core.global_phase.parameters(),
    ]
    router = list(model.core.router.parameters())
    head = list(model.head.parameters())
    phase_ids = {id(value) for value in phase}
    router_ids = {id(value) for value in router}
    head_ids = {id(value) for value in head}
    readout = [
        value
        for name, value in model.core.named_parameters()
        if value.requires_grad
        and ("readout" in name or "output_adapter" in name)
        and id(value) not in phase_ids
        and id(value) not in router_ids
    ]
    readout_ids = {id(value) for value in readout}
    base = [
        value
        for value in model.parameters()
        if value.requires_grad
        and id(value) not in phase_ids
        and id(value) not in router_ids
        and id(value) not in head_ids
        and id(value) not in readout_ids
    ]
    groups = [
        {"params": base, "lr": settings.student_learning_rate, "name": "electronic"},
        {"params": router, "lr": settings.router_learning_rate, "name": "router"},
        {"params": phase, "lr": settings.phase_learning_rate, "name": "feature_phase"},
        {
            "params": readout,
            "lr": settings.dense_readout_learning_rate,
            "name": "ccd_readout",
        },
        {"params": head, "lr": settings.dense_head_learning_rate, "name": "pose_head"},
    ]
    flat = [parameter for group in groups for parameter in group["params"]]
    if len(flat) != len({id(parameter) for parameter in flat}):
        raise RuntimeError("Optimizer parameter groups overlap")
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if {id(parameter) for parameter in flat} != expected:
        raise RuntimeError("Optimizer groups do not cover exactly all trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    pck = float(metrics["pck_at_0.2_torso"])
    nme_value = metrics.get("normalized_mean_error_torso")
    nme = float(nme_value) if nme_value is not None else math.inf
    return (-pck, nme, float(metrics["loss"]), int(epoch))


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    settings: Any,
    *,
    epoch: int,
    train_metrics: dict[str, Any],
    development_metrics: dict[str, Any],
    initialization_report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "epoch": int(epoch),
            "checkpoint_architecture": architecture_label(settings),
            "router_contract_sha256": settings.router_contract_sha256,
            "router_backend": settings.router_backend,
            "top_k": int(settings.top_k),
            "weight_variant": "ema",
            "selection": {
                "split": "development_first_LSP1000_fixed_200",
                "primary": "max_pck_at_0.2_torso",
                "ties": [
                    "min_normalized_mean_error_torso",
                    "min_development_loss",
                    "earliest_epoch",
                ],
                "sealed_test_used": False,
            },
            "train_metrics": train_metrics,
            "development_metrics": development_metrics,
            "common_initialization": initialization_report,
            "core": model.core.state_dict(),
            "head": model.head.state_dict(),
        },
        path,
    )


def train(
    loaded: Any,
    bundle: PoseProtocolBundle,
    settings: Any,
) -> dict[str, Any]:
    train_loader = _loader(bundle.train, settings, training=True)
    development_loader = _loader(bundle.development, settings, training=False)
    model = build_router_student(loaded, settings)
    initialization = load_common_initialization(model, settings)
    report = trainable_parameter_report(model, "student")
    _write_model_report(
        settings,
        "student",
        report,
        model.head.specification(),
        model.core.parameter_breakdown(),
    )
    _print_report(report)
    (settings.output_dir / "student_architecture.json").write_text(
        json.dumps(
            student_architecture_report(model, settings),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (settings.output_dir / "initialization_report.json").write_text(
        json.dumps(initialization, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    optimizer = _optimizer(model, settings)
    ema = ModelEMA(model.core, model.head, settings.ema_decay)
    # PhaseLayer keeps an explicit safety switch in addition to .training.
    # Enable it for training; model.eval() still disables dropout on dev/test.
    model.core.set_phase_dropout_active(True)
    best_key: tuple[float, float, float, int] | None = None
    best_epoch = -1
    history: list[dict[str, Any]] = []
    checkpoint = settings.output_dir / "checkpoints" / "ema_best_development_pck.pt"
    try:
        for epoch in range(1, settings.student_epochs + 1):
            if settings.router_noise_std > 0.0:
                fraction = (epoch - 1) / max(settings.student_epochs, 1)
                model.core.router.set_noise_std(
                    settings.router_noise_std * (1.0 - fraction)
                )
            started = time.perf_counter()
            train_metrics = _train_epoch(
                model,
                "student",
                train_loader,
                loaded.processor,
                loaded.device,
                optimizer,
                settings,
                epoch,
                teacher=None,
                teacher_cache=None,
                ema=ema,
            )
            with ema.applied():
                development_metrics, _ = evaluate_model(
                    model,
                    "student",
                    development_loader,
                    loaded.processor,
                    loaded.device,
                    settings,
                    phase="development_selection",
                    epoch=epoch,
                    save_outputs=False,
                    tta=False,
                )
                key = _selection_key(development_metrics, epoch)
                if best_key is None or key < best_key:
                    best_key = key
                    best_epoch = epoch
                    _save_checkpoint(
                        checkpoint,
                        model,
                        settings,
                        epoch=epoch,
                        train_metrics=train_metrics,
                        development_metrics=development_metrics,
                        initialization_report=initialization,
                    )
            row: dict[str, Any] = {
                "epoch": epoch,
                "epoch_time_sec": time.perf_counter() - started,
                "new_best_development_at_epoch": epoch == best_epoch,
                "final_selected_checkpoint": False,
            }
            row.update(
                {
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                    if not isinstance(value, dict)
                }
            )
            row.update(
                {
                    f"development_{key}": value
                    for key, value in development_metrics.items()
                    if not isinstance(value, dict)
                }
            )
            history.append(row)
            _write_history(settings.output_dir / "metrics" / "training_history.csv", history)
            print(
                f"epoch {epoch:03d} train_loss={train_metrics['loss']:.5f} "
                f"dev_PCK={development_metrics['pck_at_0.2_torso']:.4f} "
                f"dev_PCKh={development_metrics['pckh_at_0.5_head']:.4f} "
                f"dev_NME={development_metrics['normalized_mean_error_torso']:.4f} "
                f"best_epoch={best_epoch} test=SEALED",
                flush=True,
            )
    finally:
        model.core.set_phase_dropout_active(False)
        model.restore_native()
    # During training several epochs can be a new running best.  Only one row
    # may describe the final selected checkpoint in the completed history.
    for row in history:
        row["final_selected_checkpoint"] = int(row["epoch"]) == best_epoch
    _write_history(settings.output_dir / "metrics" / "training_history.csv", history)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": best_epoch,
        "development_selection_key": list(best_key) if best_key else None,
        "sealed_test_evaluations": 0,
    }
    (settings.output_dir / "training_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def evaluate_sealed_test(
    loaded: Any,
    bundle: PoseProtocolBundle,
    settings: Any,
    checkpoint: Path,
) -> dict[str, Any]:
    report_path = settings.output_dir / "sealed_test_evaluation.json"
    if report_path.exists():
        raise FileExistsError(
            f"Sealed test was already evaluated for this run: {report_path}"
        )
    checkpoint = checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location=loaded.device, weights_only=False)
    if payload.get("checkpoint_architecture") != architecture_label(settings):
        raise RuntimeError("Checkpoint architecture does not match this Router variant")
    if payload.get("router_contract_sha256") != settings.router_contract_sha256:
        raise RuntimeError("Checkpoint Router contract does not match the config")
    model = build_router_student(loaded, settings)
    model.core.load_state_dict(payload["core"], strict=True)
    model.head.load_state_dict(payload["head"], strict=True)
    test_loader = _loader(bundle.test, settings, training=False)
    try:
        metrics, _ = evaluate_model(
            model,
            "student",
            test_loader,
            loaded.processor,
            loaded.device,
            settings,
            phase="sealed_test",
            epoch=int(payload["epoch"]),
            save_outputs=True,
            tta=settings.tta_enabled,
        )
    finally:
        model.restore_native()
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_epoch": int(payload["epoch"]),
        "selection_split": "development",
        "test_used_for_selection": False,
        "sealed_test_samples": len(bundle.test),
        "metrics": metrics,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["ModelEMA", "evaluate_sealed_test", "train"]
