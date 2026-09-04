from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .optical_cores import D2NNFivePlaneCore, MultiplaneMoECore


def _phase(layer: Any) -> torch.Tensor:
    raw = layer.raw_phase.detach().cpu().float()
    if layer.parameterization == "sigmoid":
        return 2.0 * math.pi * torch.sigmoid(raw)
    return torch.remainder(raw, 2.0 * math.pi)


def _moe_mosaic(core: MultiplaneMoECore, layer: Any) -> torch.Tensor:
    value = torch.zeros(core.geometry.active_size, core.geometry.active_size)
    active = core.geometry.active_aperture
    for expert, aperture in zip(layer.experts, core.geometry.expert_apertures):
        y0 = aperture.y0 - active.y0
        x0 = aperture.x0 - active.x0
        value[y0 : y0 + core.geometry.expert_size, x0 : x0 + core.geometry.expert_size] = _phase(expert)
    return value


def stack_phase_payload(core: Any) -> dict[str, Any]:
    if isinstance(core, D2NNFivePlaneCore):
        planes = torch.stack([_phase(layer) for layer in core.expert_layers])
        raw = torch.stack(
            [layer.raw_phase.detach().cpu().float() for layer in core.expert_layers]
        )
        return {
            "family": "d2nn",
            "plane_labels": [f"d2nn_{index + 1}" for index in range(len(planes))],
            "physical_phase_rad": planes,
            "raw_phase": raw,
        }
    if isinstance(core, MultiplaneMoECore):
        expert_raw = torch.stack(
            [
                torch.stack(
                    [expert.raw_phase.detach().cpu().float() for expert in layer.experts]
                )
                for layer in core.expert_layers
            ]
        )
        expert_phase = torch.stack(
            [torch.stack([_phase(expert) for expert in layer.experts]) for layer in core.expert_layers]
        )
        mosaics = torch.stack([_moe_mosaic(core, layer) for layer in core.expert_layers])
        global_raw = core.global_phase.phase.raw_phase.detach().cpu().float()
        global_phase = _phase(core.global_phase.phase)
        return {
            "family": "moe4",
            "plane_labels": [
                *[f"expert_{index + 1}" for index in range(len(mosaics))],
                "global",
            ],
            "raw_expert_phase": expert_raw,
            "physical_expert_phase_rad": expert_phase,
            "physical_expert_mosaic_rad": mosaics,
            "raw_global_phase": global_raw,
            "physical_global_phase_rad": global_phase,
        }
    raise TypeError(f"Unsupported optical core {type(core).__name__}")


def save_snapshot(
    replacement: Any,
    output_dir: Path,
    *,
    epoch: int,
    train_loss: float,
    weight_variant: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "architecture": replacement.checkpoint_architecture,
        "variant": replacement.variant,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "weight_variant": str(weight_variant),
        "vision": stack_phase_payload(replacement.vision_surrogate.core),
        "language": stack_phase_payload(replacement.language_surrogate.core),
    }
    destination = output_dir / "phase_parameters.pt"
    torch.save(payload, destination)
    report = {
        "schema_version": 1,
        "architecture": replacement.checkpoint_architecture,
        "variant": replacement.variant,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "weight_variant": str(weight_variant),
        "phase_parameters": str(destination),
        "stacks": {},
    }
    for stack in ("vision", "language"):
        report["stacks"][stack] = {}
        for key, value in payload[stack].items():
            if torch.is_tensor(value):
                report["stacks"][stack][key] = {
                    "shape": list(value.shape),
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.mean()),
                    "std": float(value.std(unbiased=False)),
                }
            else:
                report["stacks"][stack][key] = value
    (output_dir / "phase_parameters.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _display_planes(core: Any) -> tuple[list[str], list[torch.Tensor]]:
    payload = stack_phase_payload(core)
    if payload["family"] == "d2nn":
        return payload["plane_labels"], list(payload["physical_phase_rad"])
    return payload["plane_labels"], [
        *list(payload["physical_expert_mosaic_rad"]),
        payload["physical_global_phase_rad"],
    ]


def save_preview(replacement: Any, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    stacks = [
        ("Vision", replacement.vision_surrogate.core),
        ("Language", replacement.language_surrogate.core),
    ]
    labels, _ = _display_planes(stacks[0][1])
    figure, axes = plt.subplots(2, len(labels), figsize=(3.0 * len(labels), 6.0))
    for row, (stack_name, core) in enumerate(stacks):
        current_labels, planes = _display_planes(core)
        for column, (label, plane) in enumerate(zip(current_labels, planes)):
            shown = axes[row, column].imshow(
                plane.numpy(), cmap="twilight", vmin=0.0, vmax=2.0 * math.pi
            )
            axes[row, column].set_title(
                f"{stack_name} {label}\nstd={float(plane.std(unbiased=False)):.3f} rad"
            )
            axes[row, column].axis("off")
    figure.suptitle(title)
    figure.colorbar(shown, ax=axes.ravel().tolist(), shrink=0.72, label="phase (rad)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def stage_diagnostic_payload(core: Any) -> dict[str, Any]:
    stages = []
    for index, values in enumerate(core.stage_diagnostics):
        stages.append(
            {
                "stage": index + 1,
                **{
                    key: {
                        "mean": float(value.float().mean().cpu()),
                        "std": float(value.float().std(unbiased=False).cpu()),
                        "min": float(value.float().min().cpu()),
                        "max": float(value.float().max().cpu()),
                    }
                    for key, value in values.items()
                },
            }
        )
    routers = []
    for index, routing in enumerate(core.stage_routings):
        routers.append(
            {
                "router_stage": index + 1,
                "importance": routing["importance"].detach().cpu().float().tolist(),
                "load": routing.get("load", routing["importance"]).detach().cpu().float().tolist(),
                "entropy": float(routing["normalized_entropy"].detach().cpu()),
                "mean_weight": routing["weights"].detach().cpu().float().mean(0).tolist(),
                "selection_count": routing["selected_mask"].detach().cpu().sum(0).tolist(),
            }
        )
    return {
        "core_type": type(core).__name__,
        "phase_stages": stages,
        "routers": routers,
        "final_ccd": (
            None
            if core.last_raw_detector_intensity is None
            else {
                "shape": list(core.last_raw_detector_intensity.shape),
                "mean": float(core.last_raw_detector_intensity.float().mean().cpu()),
                "std": float(core.last_raw_detector_intensity.float().std(unbiased=False).cpu()),
                "max": float(core.last_raw_detector_intensity.float().max().cpu()),
            }
        ),
    }


__all__ = [
    "save_preview",
    "save_snapshot",
    "stack_phase_payload",
    "stage_diagnostic_payload",
]
