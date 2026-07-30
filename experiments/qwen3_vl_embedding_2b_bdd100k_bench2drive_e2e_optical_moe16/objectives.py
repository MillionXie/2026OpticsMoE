from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def normalized_feature_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    cosine_weight: float,
    smooth_l1_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if student.shape != teacher.shape or student.ndim != 2:
        raise RuntimeError(
            f"Student/teacher packed features differ: {tuple(student.shape)} vs "
            f"{tuple(teacher.shape)}"
        )
    student_ln = F.layer_norm(student.float(), (student.shape[-1],))
    teacher_ln = F.layer_norm(teacher.float(), (teacher.shape[-1],))
    cosine = (
        1.0 - F.cosine_similarity(student_ln, teacher_ln, dim=-1, eps=1e-8)
    ).mean()
    smooth_l1 = F.smooth_l1_loss(student_ln, teacher_ln, beta=0.1)
    total = cosine_weight * cosine + smooth_l1_weight * smooth_l1
    return total, {"feature_cosine": cosine, "feature_smooth_l1": smooth_l1}


def auxiliary_structure_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape != targets.shape or logits.shape[1] != 3:
        raise RuntimeError(
            f"Auxiliary logits/targets must be [B,3,H,W], got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    channel_losses = []
    metrics: dict[str, torch.Tensor] = {}
    names = ("drivable", "lane", "road_participant")
    for index, (name, weight) in enumerate(zip(names, weights)):
        prediction = logits[:, index]
        target = targets[:, index].float()
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        probability = prediction.sigmoid()
        intersection = (probability * target).sum(dim=(-2, -1))
        denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        value = bce + dice
        channel_losses.append(float(weight) * value)
        metrics[f"{name}_bce"] = bce
        metrics[f"{name}_dice_loss"] = dice
    return sum(channel_losses), metrics


def behavior_cloning_loss(
    predicted_normalized: torch.Tensor,
    controls: torch.Tensor,
    *,
    steer_weight: float,
    throttle_weight: float,
    brake_weight: float,
    exclusion_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from .modeling import decode_normalized_action, encode_control_target

    target = encode_control_target(controls)
    element = F.smooth_l1_loss(
        predicted_normalized.float(), target.float(), beta=0.1, reduction="none"
    )
    steer, throttle, brake = element.unbind(dim=-1)
    decoded = decode_normalized_action(predicted_normalized)
    exclusion = (decoded[:, 1] * decoded[:, 2]).mean()
    total = (
        steer_weight * steer.mean()
        + throttle_weight * throttle.mean()
        + brake_weight * brake.mean()
        + exclusion_weight * exclusion
    )
    return total, {
        "steer_loss": steer.mean(),
        "throttle_loss": throttle.mean(),
        "brake_loss": brake.mean(),
        "throttle_brake_exclusion": exclusion,
    }


def control_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = (predicted.float() - target.float()).abs()
    return {
        "steer_mae": float(error[:, 0].mean()),
        "throttle_mae": float(error[:, 1].mean()),
        "brake_mae": float(error[:, 2].mean()),
        "control_mae": float(error.mean()),
    }


def shaped_reward(
    info: dict[str, Any],
    action: torch.Tensor | list[float],
    previous_action: torch.Tensor | list[float] | None,
    settings: Any,
) -> tuple[float, dict[str, float]]:
    required = (
        "route_progress",
        "speed",
        "target_speed",
        "lane_offset",
        "collision",
        "offroad",
        "red_light",
    )
    missing = [key for key in required if key not in info]
    if missing:
        raise RuntimeError(
            f"Closed-loop environment info is missing reward signals {missing}. "
            "Do not silently replace privileged simulator supervision."
        )
    action_tensor = torch.as_tensor(action, dtype=torch.float32)
    previous = (
        action_tensor
        if previous_action is None
        else torch.as_tensor(previous_action, dtype=torch.float32)
    )
    progress = float(info["route_progress"])
    speed_score = float(
        torch.exp(
            torch.tensor(
                -abs(float(info["speed"]) - float(info["target_speed"]))
                / settings.reward_speed_scale
            )
        )
    )
    lane_score = max(
        0.0,
        1.0 - abs(float(info["lane_offset"])) / settings.reward_lane_scale,
    )
    smoothness = float((action_tensor - previous).square().mean())
    components = {
        "route_progress": settings.reward_weights["route_progress"] * progress,
        "target_speed": settings.reward_weights["target_speed"] * speed_score,
        "lane_keep": settings.reward_weights["lane_keep"] * lane_score,
        "collision": -settings.reward_weights["collision"]
        * float(bool(info["collision"])),
        "offroad": -settings.reward_weights["offroad"] * float(bool(info["offroad"])),
        "red_light": -settings.reward_weights["red_light"]
        * float(bool(info["red_light"])),
        "control_smoothness": -settings.reward_weights["control_smoothness"]
        * smoothness,
    }
    return float(sum(components.values())), components
