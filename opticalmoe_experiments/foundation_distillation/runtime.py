import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from common.config.layout_config import layout_from_config
from common.optics.distilled_moe import (
    DetectorFeatureASGlobalRouterMoEClassifier,
    FeatureDistilledASGlobalRouterMoEClassifier,
)
from common.training.phase_dropout import phase_dropout_settings

from foundation_distillation.scripts.distillation_losses import feature_distillation_loss


def resolve_cache_dir(value: str, experiments_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "opticalmoe_experiments":
        return experiments_root.parent / path
    return experiments_root / path


def _layout_and_optical_kwargs(config: Dict):
    resolved_config = dict(config)
    resolved_model = dict(config.get("model", {}) or {})
    resolved_model.setdefault("num_experts", int(config.get("student", {}).get("num_experts", 9)))
    resolved_config["model"] = resolved_model
    layout = layout_from_config(resolved_config)
    optics = config.get("optics", {})
    prompt = config.get("prompt", {})
    dropout = phase_dropout_settings(config)
    kwargs = dict(
        wavelength_m=float(optics.get("wavelength_m", 5.32e-7)),
        pixel_size_m=float(optics.get("pixel_size_m", 8.0e-6)),
        num_layers=int(optics.get("num_layers", 5)),
        distances_m=optics.get("distances_m"),
        focal_length_m=float(optics.get("focal_length_m", 0.10)),
        aperture_mode=str(optics.get("aperture_mode", "hard")),
        phase_param=str(optics.get("phase_param", "unconstrained")),
        expert_phase_init=str(optics.get("expert_phase_init", "identity")),
        expert_init_std=float(optics.get("expert_init_std", 0.02)),
        global_fc_phase_init=str(optics.get("global_fc_phase_init", "identity")),
        global_fc_init_std=float(optics.get("global_fc_init_std", 0.02)),
        global_fc_phase_mode=str(optics.get("global_fc_phase_mode", "center_window")),
        global_fc_phase_size=int(optics.get("global_fc_phase_size", layout.active_window_size)),
        global_fc_padding_mode=str(optics.get("global_fc_padding_mode", "transparent")),
        prompt_mode=str(prompt.get("mode", "complex_order_router")),
        prompt_amplitude_init_logits=float(prompt.get("amplitude_init_logits", 2.0)),
        train_prompt_amplitudes=bool(prompt.get("train_amplitudes", True)),
        train_prompt_phase_biases=bool(prompt.get("train_phase_biases", True)),
        grating_scale=float(prompt.get("grating_scale", 1.0)),
        grating_sign_x=float(prompt.get("grating_sign_x", 1.0)),
        grating_sign_y=float(prompt.get("grating_sign_y", 1.0)),
        prompt_normalize=str(prompt.get("normalize", "sum_amplitude")),
        expert_phase_dropout_mode=dropout["expert_mode"],
        expert_phase_dropout_p=dropout["expert_p"],
        global_fc_phase_dropout_mode=dropout["global_fc_mode"],
        global_fc_phase_dropout_p=dropout["global_fc_p"],
        phase_dropout_block_size=dropout["block_size"],
        phase_dropout_batch_shared=dropout["batch_shared"],
        evanescent_mode=str(optics.get("evanescent_mode", "zero")),
    )
    return layout, kwargs


def build_student(config: Dict, num_classes: int, teacher_feature_dim: int):
    layout, optical_kwargs = _layout_and_optical_kwargs(config)
    model = FeatureDistilledASGlobalRouterMoEClassifier(
        num_classes=num_classes,
        teacher_feature_dim=teacher_feature_dim,
        layout=layout,
        feature_detector_config=config.get("feature_detector", {}),
        feature_preprocess_config=config.get("feature_preprocess", {}),
        classifier_config=config.get("classifier", {}),
        projector_config=config.get("projector", {}),
        **optical_kwargs,
    )
    model.set_phase_dropout_active(False)
    return model


def build_end_to_end_student(config: Dict, num_classes: int):
    layout, optical_kwargs = _layout_and_optical_kwargs(config)
    model = DetectorFeatureASGlobalRouterMoEClassifier(
        num_classes=num_classes,
        layout=layout,
        feature_detector_config=config.get("feature_detector", {}),
        feature_preprocess_config=config.get("feature_preprocess", {}),
        classifier_config=config.get("classifier", {}),
        **optical_kwargs,
    )
    model.set_phase_dropout_active(False)
    return model


def build_optimizer(model, config: Dict):
    cfg = config.get("optimizer", {})
    if str(cfg.get("type", "adamw")).lower() != "adamw":
        raise ValueError("foundation_distillation currently supports optimizer.type=adamw.")
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
    )


def run_distillation_epoch(
    model,
    loader,
    device,
    loss_cfg: Dict,
    optimizer=None,
    print_freq: int = 0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "total_loss": 0.0,
        "ce_loss": 0.0,
        "feature_loss": 0.0,
        "feature_cosine": 0.0,
        "leak_loss": 0.0,
        "outside_camera_energy_ratio": 0.0,
        "correct": 0,
        "samples": 0,
    }
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels, teacher_features, _indices) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            teacher_features = teacher_features.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _camera_raw, _camera_processed, _semantic, semantic_normalized, auxiliary = model.forward_with_aux(images)
            losses = feature_distillation_loss(
                logits,
                labels,
                semantic_normalized,
                teacher_features,
                outside_camera_energy_ratio=auxiliary["outside_camera_energy_ratio"],
                ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
                feature_distill_weight=float(loss_cfg.get("feature_distill_weight", 0.5)),
                leak_loss_weight=float(loss_cfg.get("leak_loss_weight", 0.0)),
            )
            if training:
                losses["total_loss"].backward()
                optimizer.step()
            batch_size = int(labels.numel())
            for key in (
                "total_loss",
                "ce_loss",
                "feature_loss",
                "feature_cosine",
                "leak_loss",
                "outside_camera_energy_ratio",
            ):
                totals[key] += float(losses[key].detach().item()) * batch_size
            totals["correct"] += int((logits.argmax(dim=1) == labels).sum().item())
            totals["samples"] += batch_size
            if training and print_freq > 0 and batch_index % int(print_freq) == 0:
                print(
                    f"  update {batch_index:03d}/{len(loader):03d} | total={losses['total_loss'].item():.4f} "
                    f"ce={losses['ce_loss'].item():.4f} feat={losses['feature_loss'].item():.4f} "
                    f"leak={losses['leak_loss'].item():.4f} "
                    f"acc={(logits.argmax(1) == labels).float().mean().item():.4f} cos={losses['feature_cosine'].item():.4f}"
                )
    samples = max(int(totals["samples"]), 1)
    return {
        "total_loss": totals["total_loss"] / samples,
        "ce_loss": totals["ce_loss"] / samples,
        "feature_loss": totals["feature_loss"] / samples,
        "feature_cosine": totals["feature_cosine"] / samples,
        "leak_loss": totals["leak_loss"] / samples,
        "outside_camera_energy_ratio": totals["outside_camera_energy_ratio"] / samples,
        "acc": totals["correct"] / samples,
        "samples": int(totals["samples"]),
    }


def run_supervised_epoch(
    model,
    loader,
    device,
    optimizer=None,
    print_freq: int = 0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_total, correct, samples = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            model_outputs = model(images)
            logits = model_outputs[0] if isinstance(model_outputs, (tuple, list)) else model_outputs
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.numel())
            loss_total += float(loss.detach().item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            samples += batch_size
            if training and print_freq > 0 and batch_index % int(print_freq) == 0:
                print(
                    f"  update {batch_index:03d}/{len(loader):03d} | ce={loss.item():.4f} "
                    f"acc={(logits.argmax(1) == labels).float().mean().item():.4f}"
                )
    denominator = max(samples, 1)
    return {"loss": loss_total / denominator, "acc": correct / denominator, "samples": samples}


@torch.no_grad()
def predict_distillation(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    predictions, targets, similarities = [], [], []
    for batch_index, (images, labels, teacher_features, _indices) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        logits, _camera_raw, _camera_processed, _semantic, semantic_normalized = model(
            images.to(device, non_blocking=True)
        )
        teacher = torch.nn.functional.normalize(teacher_features.to(device).float(), dim=-1)
        similarities.append(torch.nn.functional.cosine_similarity(semantic_normalized, teacher, dim=-1).cpu())
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    return torch.cat(predictions), torch.cat(targets), torch.cat(similarities)


@torch.no_grad()
def predict_supervised(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    predictions, targets = [], []
    for batch_index, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        model_outputs = model(images.to(device, non_blocking=True))
        logits = model_outputs[0] if isinstance(model_outputs, (tuple, list)) else model_outputs
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    return torch.cat(predictions), torch.cat(targets)


def architecture_payload(model, config: Dict, dataset_name: str, teacher_cfg: Dict) -> Dict:
    layout = model.layout
    global_fc_shape = list(model.global_fc.phase_size)
    active = layout.active_window_aperture
    return {
        "model": "FeatureDistilledASGlobalRouterMoEClassifier",
        "experiment_variant": "feature_distillation",
        "dataset_name": dataset_name,
        "teacher_type": teacher_cfg.get("type"),
        "teacher_backend": teacher_cfg.get("resolved_backend", teacher_cfg.get("backend", "auto")),
        "teacher_model_name": teacher_cfg.get("model_name"),
        "teacher_input_mode": teacher_cfg.get("input_mode"),
        "feature_type": teacher_cfg.get(
            "resolved_feature_type",
            teacher_cfg.get("feature_type", "image_embedding" if teacher_cfg.get("type") == "clip_image_encoder" else "cls"),
        ),
        "teacher_text_encoder_used": False,
        "student_model_type": "feature_distilled_optical_moe",
        "student_backbone_type": "optical_moe",
        "student_feature_dim": model.camera_feature_dim,
        "lenet_parameter_count": 0,
        "geometry_profile": layout.geometry_profile,
        "canvas_size": layout.canvas_size,
        "propagation_canvas_size": layout.canvas_size,
        "physical_camera_size": layout.active_window_size,
        "camera_region": list(model.camera_region),
        "padding_used_for_feature": False,
        "input_size": layout.input_size,
        "num_experts": layout.num_experts,
        "expert_size": layout.expert_size,
        "expert_pitch": layout.expert_pitch,
        "gap_px": layout.gap_px,
        "expert_union_bounds": layout.expert_union_bounds,
        "expert_union_size": layout.expert_union_size,
        "active_window_size": layout.active_window_size,
        "active_window_region": [active.y0, active.y1, active.x0, active.x1],
        "prompt_aperture_size": layout.prompt_aperture_size,
        "prompt_trainable_type": "channel_amplitude_and_phase_bias",
        "prompt_trainable_pixelwise": False,
        "global_fc_phase_size": global_fc_shape[0] if global_fc_shape[0] == global_fc_shape[1] else global_fc_shape,
        "global_fc_phase_shape": global_fc_shape,
        "global_fc_phase_region": model.global_fc.phase_region(),
        "global_fc_padding_is_trainable": False,
        "global_fc_parameter_count": model.global_fc.trainable_parameter_count(),
        "expert_phase_parameter_count": model.optical_backbone.expert_phase_parameter_count(),
        "feature_detector": model.feature_detector_config,
        "feature_source": "camera_intensity",
        "camera_feature_dim": model.camera_feature_dim,
        "feature_preprocess": model.feature_preprocess_config,
        "teacher_feature_dim": model.teacher_feature_dim,
        "classifier": model.classifier_config,
        "projector": model.projector_config,
        "projector_input_dim": model.projector_config["input_dim"],
        "projector_output_dim": model.projector_config["output_dim"],
        "projector_type": model.projector_config["type"],
        "classifier_input": model.classifier_config["input"],
        "leak_loss_weight": float(config.get("loss", {}).get("leak_loss_weight", 0.0)),
        "optical_parameter_count": model.optical_parameter_count(),
        "prompt_parameter_count": model.prompt_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "projector_parameter_count": model.projector_parameter_count(),
        "classifier_parameter_count": model.classifier_parameter_count(),
        "feature_preprocess_parameter_count": model.feature_preprocess_parameter_count(),
        "inference_parameter_count": model.total_parameter_count(),
        "training_parameter_count": model.total_parameter_count(),
        "total_parameter_count": model.total_parameter_count(),
        "phase_dropout_config": config.get("regularization", {}).get("phase_dropout", {}),
    }


def end_to_end_architecture_payload(model, config: Dict, dataset_name: str) -> Dict:
    layout = model.layout
    global_fc_shape = list(model.global_fc.phase_size)
    active = layout.active_window_aperture
    return {
        "model": "DetectorFeatureASGlobalRouterMoEClassifier",
        "experiment_variant": "end_to_end_ce_baseline",
        "dataset_name": dataset_name,
        "teacher_type": "none",
        "teacher_used": False,
        "feature_distillation_used": False,
        "student_model_type": "end_to_end_optical_moe",
        "student_backbone_type": "optical_moe",
        "student_feature_dim": model.camera_feature_dim,
        "lenet_parameter_count": 0,
        "geometry_profile": layout.geometry_profile,
        "canvas_size": layout.canvas_size,
        "propagation_canvas_size": layout.canvas_size,
        "physical_camera_size": layout.active_window_size,
        "camera_region": list(model.camera_region),
        "padding_used_for_feature": False,
        "input_size": layout.input_size,
        "num_experts": layout.num_experts,
        "expert_size": layout.expert_size,
        "expert_pitch": layout.expert_pitch,
        "gap_px": layout.gap_px,
        "expert_union_bounds": layout.expert_union_bounds,
        "expert_union_size": layout.expert_union_size,
        "active_window_size": layout.active_window_size,
        "active_window_region": [active.y0, active.y1, active.x0, active.x1],
        "prompt_aperture_size": layout.prompt_aperture_size,
        "prompt_trainable_type": "channel_amplitude_and_phase_bias",
        "prompt_trainable_pixelwise": False,
        "global_fc_phase_size": global_fc_shape[0] if global_fc_shape[0] == global_fc_shape[1] else global_fc_shape,
        "global_fc_phase_shape": global_fc_shape,
        "global_fc_phase_region": model.global_fc.phase_region(),
        "global_fc_padding_is_trainable": False,
        "global_fc_parameter_count": model.global_fc.trainable_parameter_count(),
        "expert_phase_parameter_count": model.optical_backbone.expert_phase_parameter_count(),
        "feature_detector": model.feature_detector_config,
        "feature_source": "camera_intensity",
        "camera_feature_dim": model.camera_feature_dim,
        "feature_preprocess": model.feature_preprocess_config,
        "classifier": config.get("classifier", {}),
        "projector": None,
        "optical_parameter_count": model.optical_parameter_count(),
        "prompt_parameter_count": model.prompt_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "classifier_parameter_count": model.classifier_parameter_count(),
        "feature_preprocess_parameter_count": model.feature_preprocess_parameter_count(),
        "projector_parameter_count": 0,
        "inference_parameter_count": model.total_parameter_count(),
        "training_parameter_count": model.total_parameter_count(),
        "total_parameter_count": model.total_parameter_count(),
        "phase_dropout_config": config.get("regularization", {}).get("phase_dropout", {}),
    }


def load_checkpoint_state(model, checkpoint_path: Path, device) -> Dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("model_state", payload.get("model", payload)))
    model.load_state_dict(state)
    return payload
