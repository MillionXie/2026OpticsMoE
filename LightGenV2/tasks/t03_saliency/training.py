"""Periodic-test SALICON trainer retaining only selected-best and last."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import (
    training as legacy,
)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.visualization import (
    save_examples,
)

from .modeling import build_student, initialize_student, optimizer
from .visualize import render
from .training_support import ModelEMA, TrainTeacherMaps, AlignedFlipLoader, AlignedWeakLoader, distillation_weight, supervision_for_epoch
from .plateau import PlateauController
from .training_support import use_spawn_workers


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selection_report(counts: torch.Tensor) -> dict[str, Any]:
    total = float(counts.sum())
    share = counts.float() / max(total, 1.0)
    return {
        "selected_slots": [int(value) for value in counts.tolist()],
        "selection_share": [float(value) for value in share.tolist()],
        "effective_experts_inverse_simpson": float(
            1.0 / share.square().sum().clamp_min(1.0e-12)
        ),
        "unused_experts": [
            index for index, value in enumerate(counts.tolist()) if value == 0
        ],
    }


def _checkpoint(
    path: Path,
    model: Any,
    epoch: int,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    weight_kind: str = "live",
    ema_state: Any = None,
    test_weight_kind: str | None = None,
    training_only_hint: Any = None,
    training_only_semantic: Any = None,
    training_only_mgd: Any = None,
    training_only_first_stage: Any = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "architecture": model.checkpoint_architecture,
            "epoch": int(epoch),
            "weight_kind": weight_kind,
            "test_metrics_weight_kind": test_weight_kind or weight_kind,
            "ema_state": ema_state,
            "training_only_hint": training_only_hint,
            "training_only_semantic": training_only_semantic,
            **({"training_only_mgd": training_only_mgd} if training_only_mgd is not None else {}),
            **({"training_only_first_stage": training_only_first_stage} if training_only_first_stage is not None else {}),
            "core": model.core.state_dict(),
            "saliency_head": model.head.state_dict(),
            "train_metrics": train_metrics,
            "fusion_contract": {"minimum": model.core.hybrid.fusion_alpha_min,
                                "maximum": model.core.hybrid.fusion_alpha_max},
            "test_metrics": test_metrics,
            "selection_biased": True,
        },
        path,
    )


def staged_epoch(optim: Any, settings: Any, epoch: int) -> dict[str, Any]:
    """Closed-form LR schedule, without recursively compounding multipliers."""
    warmup = settings.staged_warmup_epochs
    if epoch <= warmup:
        stage, factor = "optics_readout_adaptation", 1.0
    elif epoch < settings.staged_polish_start:
        stage = "joint"
        progress = (epoch - warmup - 1) / max(1, settings.staged_polish_start - warmup - 1)
        factor = 0.2 + 0.8 * (1 + math.cos(math.pi * progress)) / 2
    else:
        stage = "polish"
        progress = (epoch - settings.staged_polish_start) / max(1, settings.student_epochs - settings.staged_polish_start)
        factor = 0.02 + 0.08 * (1 + math.cos(math.pi * progress)) / 2
    for group in optim.param_groups:
        group.setdefault("schedule_base_lr", group["lr"])
        group["lr"] = group["schedule_base_lr"] * (0.0 if stage == "optics_readout_adaptation" and group["name"] == "electronic" else factor)
        if getattr(settings, "staged_freeze_electronic_gradients", False) and group["name"] == "electronic":
            for parameter in group["params"]:
                parameter.requires_grad_(stage != "optics_readout_adaptation")
    progress = max(0., min(1., (epoch - warmup) / max(1, settings.staged_polish_start - warmup)))
    hard = settings.router_hard_load_balance_weight * (1-progress) + settings.staged_final_hard_balance * progress
    return {"stage": stage, "hard_balance_weight": hard,
            **{f"lr_{g['name']}": g["lr"] for g in optim.param_groups}}


def train(loaded: Any, bundle: Any, settings: Any) -> dict[str, Any]:
    model = build_student(loaded, settings)
    initialization = initialize_student(model, settings)
    feature_targets = None
    relation_targets = None
    if getattr(settings, 'relational_distillation', {}):
        from .relational_distillation import RelationalTargets
        relation_targets = RelationalTargets(settings, bundle.train_records)
        _write_json(settings.output_dir/'relational_distillation_provenance.json', relation_targets.provenance)
    if getattr(settings, 'feature_pretraining', {}).get('enabled', False):
        from .feature_pretraining import FixedFeatureTargets, initialize_teacher_decoder
        feature_targets = FixedFeatureTargets(settings, bundle.train_records)
        initialization.update(initialize_teacher_decoder(model, settings))
        _write_json(settings.output_dir/'feature_pretraining_provenance.json', {
            **feature_targets.provenance, **settings.feature_pretraining,
            'initial_evaluation_uses_replaced_teacher_decoder': True,
            'ground_truth_retained_in_both_stages': True,
        })
    _write_json(settings.output_dir / "initialization_report.json", initialization)
    loader_settings = copy(settings)
    aligned_flip = settings.augmentation_enabled and settings.augmentation_mode == "aligned_flip"
    aligned_weak = settings.augmentation_enabled and settings.augmentation_mode == "aligned_weak"
    fixed_crop = bool(getattr(settings, 'fixed_crop_distillation', {}))
    if aligned_flip or aligned_weak:
        loader_settings.augmentation_enabled = False
    train_loader, test_loader = legacy.build_loaders(bundle, loader_settings, training=True)
    use_spawn_workers(train_loader)
    use_spawn_workers(test_loader)
    optim = optimizer(model, settings)
    masked_targets = None
    first_stage = None
    if getattr(settings, 'first_stage_supervision', {}):
        from .first_stage_supervision import FirstStageSupervisor
        first_stage = FirstStageSupervisor(model).to(loaded.device)
        optim.add_param_group({'params': list(first_stage.parameters()), 'name': 'training_first_stage',
                              'lr': settings.first_stage_supervision['learning_rate'], 'weight_decay': 0.0})
        _write_json(settings.output_dir/'first_stage_supervision_provenance.json',
                    dict(first_stage.provenance, **settings.first_stage_supervision))
    if getattr(settings, 'masked_distillation', {}):
        from .masked_distillation import MaskedTeacherRecovery
        masked_targets = MaskedTeacherRecovery(settings, bundle.train_records).to(loaded.device)
        optim.add_param_group({'params': list(masked_targets.parameters()), 'name': 'training_mgd',
                              'lr': settings.masked_distillation['learning_rate'], 'weight_decay': 0.0})
        _write_json(settings.output_dir/'masked_distillation_provenance.json', masked_targets.provenance)
    if feature_targets is not None:
        from .feature_pretraining import configure_router_path_optimizer
        _write_json(settings.output_dir/'feature_router_path_provenance.json',
                    configure_router_path_optimizer(model, optim, settings))
    ema = ModelEMA(model, settings.ema_decay) if settings.ema_decay else None
    ema_hook = optim.register_step_post_hook(ema.update) if ema else None
    teacher = TrainTeacherMaps(settings, bundle.train_records) if settings.distillation_initial_weight else None
    hints = None
    if settings.feature_hint_initial_weight > 0:
        from .feature_hints import TrainFeatureHints
        from .modeling import sha256_file
        hints = TrainFeatureHints(settings,bundle.train_records).to(loaded.device)
        optim.add_param_group({'params':list(hints.parameters()),'name':'training_hint',
                              'lr':settings.feature_hint_learning_rate,'weight_decay':0.0})
        _write_json(settings.output_dir/'feature_hint_provenance.json',{
            **hints.manifest,'cache_sha256':sha256_file(settings.feature_hint_cache),
            'training_projection_parameters':sum(p.numel() for p in hints.parameters()),
            'inference_parameters_added':0,'student_architecture_unchanged':True,
            'loss':'mean(1-cosine(project(student fused latent),teacher decoder input)), channels normalized per pixel',
            'loss_mode':settings.feature_hint_loss_mode,
            'training_loss_spatial_mean_removed':settings.feature_hint_loss_mode=='spatial_centered_cosine',
            'projection_checkpoint':'last_checkpoint.pt:training_only_hint, separate from inference core/head'})
    if aligned_flip:
        train_loader = AlignedFlipLoader(train_loader, settings.horizontal_flip_probability,
                                        settings.random_seed + 703, teacher)
    if aligned_weak:
        train_loader = AlignedWeakLoader(train_loader, settings, teacher)
    if fixed_crop:
        from .fixed_crop_training import FixedCropTargets, FixedCropLoader
        crop_targets = FixedCropTargets(settings, bundle.train_records)
        train_loader = FixedCropLoader(train_loader, settings, teacher, crop_targets)
        _write_json(settings.output_dir/'fixed_crop_provenance.json', crop_targets.provenance)
    if teacher is not None:
        from .modeling import sha256_file
        _write_json(settings.output_dir / "teacher_cache_provenance.json", {
            **teacher.manifest, "cache_sha256": sha256_file(settings.distillation_cache),
            "distillation_loss": getattr(settings, "distillation_loss", "kl"),
            "teacher_executed_during_student_inference": False,
            "student_train_augmentation": "fixed_crop" if fixed_crop else settings.augmentation_mode if settings.augmentation_enabled else "none",
            "teacher_map_transform": (crop_targets.provenance['teacher_target_used'] if fixed_crop else "crop/resize/flip teacher probability density, renormalize, log; approximate view consistency, not online teacher inference"
                                      if aligned_weak else "same horizontal flip as image/density/fixation" if aligned_flip else "none")})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, int(settings.student_epochs))
    )
    history: list[dict[str, Any]] = []
    controller = (PlateauController(**settings.adaptive_plateau_options)
                  if settings.adaptive_plateau_enabled else None)
    adaptive_events = []
    best_cc = -math.inf
    best_epoch = -1
    started = time.perf_counter()
    extra_stream = None
    semantic = None
    try:
        if getattr(settings, 'unlabeled_weight', 0) > 0:
            from .semisupervised import build_unlabeled_stream
            extra_stream = build_unlabeled_stream(settings)
            _write_json(settings.output_dir/'unlabeled_teacher_provenance.json', {
                **extra_stream.teacher.manifest,
                'cache_sha256': settings.unlabeled_cache_sha256,
                'unlabeled_weight': settings.unlabeled_weight,
                'ground_truth_supervision_retained': True,
                'inference_parameters_added': 0,
            })
            if getattr(settings,'semantic_weight',0):
                from .semantic_auxiliary import SemanticAuxiliary
                semantic = SemanticAuxiliary(settings,extra_stream.loader.dataset).to(loaded.device)
                optim.add_param_group({'params':list(semantic.parameters()),'name':'training_semantic',
                                      'lr':settings.semantic_learning_rate,'weight_decay':.01})
                _write_json(settings.output_dir/'semantic_auxiliary_provenance.json',semantic.provenance)
        if getattr(settings, "initialization_checkpoint", None) is not None:
            model.core.set_phase_dropout_active(False)
            initial_metrics, _ = legacy.evaluate_model(model, test_loader, loaded, settings)
            best_cc, best_epoch = float(initial_metrics["cc"]), 0
            if controller is not None:
                controller.observe(0, best_cc)
            _checkpoint(settings.output_dir / "best_checkpoint.pt", model, 0, {}, initial_metrics,
                        training_only_semantic=semantic.state_dict() if semantic is not None else None)
            _write_json(settings.output_dir / "warmstart_evaluation.json", initial_metrics)
            print(f"[T03] warmstart CC={best_cc:.6f}", flush=True)
        for epoch in range(1, int(settings.student_epochs) + 1):
            settings.mixup_active = bool(getattr(settings,'mixup',{})) and epoch <= settings.mixup['end_epoch']
            stage_report = {}
            if aligned_weak:
                train_loader.enabled = settings.augmentation_end_epoch == 0 or epoch <= settings.augmentation_end_epoch
                stage_report["augmentation_active"] = train_loader.enabled
            if fixed_crop:
                train_loader.enabled = epoch <= settings.fixed_crop_distillation['end_epoch']
                stage_report['augmentation_active'] = train_loader.enabled
                stage_report['fixed_crop_teacher_mode'] = settings.fixed_crop_distillation['mode']
            if settings.staged_training:
                stage_report.update(staged_epoch(optim, settings, epoch))
                model._router_hard_weight = stage_report["hard_balance_weight"]
                if controller is not None:
                    controller.scale_epoch_rates(optim)
                    stage_report.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
                    stage_report["adaptive_lr_multiplier"] = controller.multiplier
            model.core.set_phase_dropout_active(True)
            settings.map_kd_weight = distillation_weight(
                settings.distillation_initial_weight, settings.distillation_end_epoch, epoch,
                settings.distillation_final_weight)
            stage_report["kd_weight"] = settings.map_kd_weight
            epoch_settings, supervision_report = supervision_for_epoch(settings, epoch)
            stage_report.update(supervision_report)
            if feature_targets is not None:
                from .feature_pretraining import prepare_epoch, train_epoch as train_feature_epoch
                epoch_settings, feature_report = prepare_epoch(model, optim, epoch_settings, epoch)
                stage_report.update(feature_report)
                train_metrics = train_feature_epoch(model, train_loader, loaded, epoch_settings, optim,
                    teacher, feature_targets, feature_report['feature_weight'])
            elif extra_stream is not None:
                from .semisupervised import train_semisupervised_epoch
                train_metrics = train_semisupervised_epoch(model,train_loader,loaded,epoch_settings,optim,
                                                            teacher,extra_stream,semantic=semantic)
                stage_report['unlabeled_weight'] = settings.unlabeled_weight
                if semantic is not None:
                    stage_report['semantic_weight'] = settings.semantic_weight
            elif getattr(settings, "sam_rho", 0) > 0:
                from .sam_training import train_sam_epoch
                relation_kwargs = {}
                if relation_targets is not None:
                    options = settings.relational_distillation
                    epoch_settings.relational_current_weight = distillation_weight(
                        options['initial_weight'], options['end_epoch'], epoch, options['final_weight'])
                    stage_report['relational_weight'] = epoch_settings.relational_current_weight
                    relation_kwargs['relation_targets'] = relation_targets
                if masked_targets is not None:
                    options = settings.masked_distillation
                    epoch_settings.masked_current_weight = distillation_weight(
                        options['initial_weight'], options['end_epoch'], epoch, options['final_weight'])
                    epoch_settings.masked_generator_warmup = epoch <= options['generator_warmup_epochs']
                    stage_report.update(masked_weight=epoch_settings.masked_current_weight,
                                        masked_generator_warmup=epoch_settings.masked_generator_warmup)
                    relation_kwargs['masked_targets'] = masked_targets
                if first_stage is not None:
                    options = settings.first_stage_supervision
                    epoch_settings.first_stage_current_weight = distillation_weight(
                        options['initial_weight'], options['end_epoch'], epoch, options['final_weight'])
                    epoch_settings.first_stage_head_warmup = epoch <= options['head_warmup_epochs']
                    stage_report.update(first_stage_weight=epoch_settings.first_stage_current_weight,
                                        first_stage_head_warmup=epoch_settings.first_stage_head_warmup)
                    relation_kwargs['first_stage'] = first_stage
                train_metrics = train_sam_epoch(model, train_loader, loaded, epoch_settings, optim,
                                                teacher if settings.map_kd_weight > 0 else None, **relation_kwargs)
            elif hints is None:
                train_metrics = legacy._train_epoch(
                    "student", model, train_loader, loaded, settings, optim,
                    teacher_cache=teacher if settings.map_kd_weight > 0 else None,
                )
            else:
                from .feature_hints import train_hint_epoch
                settings.feature_hint_current_weight = distillation_weight(settings.feature_hint_initial_weight,
                    settings.feature_hint_end_epoch,epoch,settings.feature_hint_final_weight)
                stage_report['feature_hint_weight'] = settings.feature_hint_current_weight
                train_metrics = train_hint_epoch(model,train_loader,loaded,settings,optim,
                    teacher if settings.map_kd_weight > 0 else None,hints)
            if aligned_weak or fixed_crop:
                stage_report["augmentation_images"] = train_loader.epoch_augmented_images
                stage_report["augmentation_total_images"] = train_loader.epoch_images
                if fixed_crop:
                    stage_report['augmentation_empty_fixation_fallbacks'] = train_loader.epoch_fallback_images
            model.core.set_phase_dropout_active(False)
            scheduled_test = (
                epoch == 1
                or epoch % int(settings.test_interval_epochs) == 0
                or epoch == int(settings.student_epochs)
            )
            test_metrics = None
            if scheduled_test:
                with ema.applied() if ema else nullcontext():
                    test_metrics, _ = legacy.evaluate_model(model, test_loader, loaded, settings)
                    if float(test_metrics["cc"]) > best_cc:
                        best_cc, best_epoch = float(test_metrics["cc"]), epoch
                        _checkpoint(settings.output_dir / "best_checkpoint.pt", model,
                                    epoch, train_metrics, test_metrics,
                                    weight_kind="ema" if ema else "live",
                                    training_only_semantic=semantic.state_dict() if semantic is not None else None)
                if controller is not None:
                    action = controller.observe(epoch, float(test_metrics["cc"]))
                    event = {"epoch": epoch, "test_cc": float(test_metrics["cc"]),
                             "action": action, "next_lr_multiplier": controller.multiplier,
                             "bad_tests": controller.bad_tests, "reductions": controller.reductions}
                    adaptive_events.append(event)
                    _write_json(settings.output_dir / "metrics" / "adaptive_events.json", adaptive_events)
                    stage_report["adaptive_action"] = action
                    print(f"[T03 adaptive] {json.dumps(event)}", flush=True)
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **(
                    {f"test_{key}": value for key, value in test_metrics.items()}
                    if test_metrics is not None
                    else {}
                ),
                "learning_rate": optim.param_groups[0]["lr"],
                **stage_report,
                "alpha1": float(model.core.hybrid.block1_optical_fusion.detach()),
                "alpha2": float(model.core.hybrid.block2_optical_fusion.detach()),
                "test_weight_kind": "ema" if ema else "live",
            }
            history.append(row)
            _write_csv(settings.output_dir / "metrics" / "training_history.csv", history)
            _checkpoint(
                settings.output_dir / "last_checkpoint.pt",
                model,
                epoch,
                train_metrics,
                test_metrics,
                ema_state=ema.shadow if ema else None,
                test_weight_kind="ema" if ema else "live",
                training_only_hint=hints.state_dict() if hints is not None else None,
                training_only_semantic=semantic.state_dict() if semantic is not None else None,
                **({'training_only_mgd': masked_targets.state_dict()} if masked_targets is not None else {}),
                **({'training_only_first_stage': first_stage.state_dict()} if first_stage is not None else {}),
            )
            if not settings.staged_training:
                scheduler.step()
            suffix = "" if test_metrics is None else f" test_CC={test_metrics['cc']:.4f}"
            print(
                f"[T03] epoch={epoch:03d}/{settings.student_epochs:03d} "
                f"train_loss={train_metrics['loss']:.5f}{suffix} best_CC={best_cc:.4f}",
                flush=True,
            )
            if controller is not None and controller.stopped:
                break
    finally:
        if extra_stream is not None:
            extra_stream.close()
        if ema_hook is not None:
            ema_hook.remove()
        model.core.set_phase_dropout_active(False)
        model.restore_native()
    render(settings.output_dir / "best_checkpoint.pt", settings.output_dir / "best_visualization")
    report = {
        "elapsed_seconds": time.perf_counter() - started,
        "selected_epoch": best_epoch,
        "selected_periodic_test_cc": best_cc,
        "selection": f"maximum public-test CC at warmstart, epoch 1, every {settings.test_interval_epochs} epochs, and final",
        "checkpoint_retention": ["best_checkpoint.pt", "last_checkpoint.pt"],
        "completed_epochs": len(history),
        "requested_epochs": int(settings.student_epochs),
        "stop_reason": "public_test_plateau" if controller is not None and controller.stopped else "epoch_budget",
        "adaptive_public_test_control": controller is not None,
        "selection_biased": True,
    }
    _write_json(settings.output_dir / "training_report.json", report)
    return report


@torch.inference_mode()
def evaluate_selected_checkpoint(
    loaded: Any, bundle: Any, settings: Any, checkpoint: Path
) -> dict[str, Any]:
    model = build_student(loaded, settings)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != model.checkpoint_architecture:
        raise RuntimeError("T03 checkpoint architecture mismatch")
    model.core.load_state_dict(payload["core"], strict=True)
    model.head.load_state_dict(payload["saliency_head"], strict=True)
    _, loader = legacy.build_loaders(bundle, settings, training=False)
    use_spawn_workers(loader)
    router_counts = torch.zeros(4, dtype=torch.long)
    router_energy_sum = torch.zeros(4, dtype=torch.float64)
    router_probability_sum = torch.zeros(4, dtype=torch.float64)
    router_samples = 0
    handle = None
    if settings.lightgen_model_variant != "d2nn_active_expert_matched":
        router = model.core.optical_branch.core.router

        def collect_selection(
            _module: Any, _inputs: Any, output: dict[str, Any]
        ) -> None:
            nonlocal router_samples
            router_counts.add_(
                output["selected_mask"].detach().sum(dim=0).cpu()
            )
            router_energy_sum.add_(
                output["detector_energy_fraction"]
                .detach()
                .double()
                .sum(dim=0)
                .cpu()
            )
            router_probability_sum.add_(
                output["probabilities"].detach().double().sum(dim=0).cpu()
            )
            router_samples += int(output["selected_mask"].shape[0])

        handle = router.register_forward_hook(collect_selection)
    try:
        model.core.set_phase_dropout_active(False)
        metrics, examples = legacy.evaluate_model(
            model,
            loader,
            loaded,
            settings,
            collect_examples=settings.visualization_sample_count,
        )
        result = {
            "system": settings.lightgen_model_variant,
            "split": "SALICON official val2014 used as public test/selection",
            "test_samples": len(bundle.validation_records),
            "selected_epoch": int(payload["epoch"]),
            "weight_kind": payload.get("weight_kind", "live"),
            "checkpoint": str(checkpoint),
            "selection_biased": True,
            "metrics": metrics,
            "alpha": [float(model.core.hybrid.block1_optical_fusion.detach()),
                      float(model.core.hybrid.block2_optical_fusion.detach())],
            "ccd_normalization": settings.ccd_normalization,
            "phase_change_from_initialization": phase_change_report(payload, settings),
            "router_audit": None if handle is None else {
                **_selection_report(router_counts),
                "samples": router_samples,
                "mean_detector_energy_fraction": [
                    float(value) for value in (router_energy_sum / router_samples).tolist()
                ],
                "mean_router_probability": [
                    float(value)
                    for value in (router_probability_sum / router_samples).tolist()
                ],
            },
        }
        _write_json(settings.output_dir / "selected_checkpoint_test_evaluation.json", result)
        save_examples(
            settings.output_dir / "best_visualization" / "saliency_examples",
            examples,
            kind="student",
        )
        return result
    finally:
        if handle is not None:
            handle.remove()
        model.restore_native()


__all__ = ["evaluate_selected_checkpoint", "train"]


def phase_change_report(payload: dict, settings: Any) -> dict:
    """Compare physical phases even when their parameter coordinates changed."""
    from .router_phase import checkpoint_phase, RADIANS_SUFFIX
    source = getattr(settings, "initialization_checkpoint", None)
    if source is None:
        return {}
    old_payload = torch.load(source, map_location="cpu", weights_only=False)
    previous = old_payload['core']
    result = {}
    for name, value in payload["core"].items():
        if not any(key in name for key in ("raw_phase", "raw_router_phase")) or name not in previous:
            continue
        before, after = previous[name].float(), value.float()
        radians = (checkpoint_phase(name,after,payload.get('architecture'))
                   - checkpoint_phase(name,before,old_payload.get('architecture')))
        comparable = ('raw_router_phase' not in name or
            str(payload.get('architecture')).endswith(RADIANS_SUFFIX)==str(old_payload.get('architecture')).endswith(RADIANS_SUFFIX))
        circular = torch.atan2(radians.sin(), radians.cos())
        result[name] = {"elements": value.numel(),
                        "raw_rms_change": float((after-before).square().mean().sqrt()) if comparable else None,
                        "raw_coordinates_comparable": comparable,
                        "circular_phase_rms_rad": float(circular.square().mean().sqrt()),
                        "fraction_above_001_rad": float((circular.abs() > .01).float().mean())}
    return result
