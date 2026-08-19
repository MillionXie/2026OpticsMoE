from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .datasets import load_datasets, make_loader
from .fixed_feedback_training import METHODS, sha256_file
from .formal_settings import load_formal_settings
from .model import OpticalClassifier
from .optics import OpticalDeploymentState
from .training import build_model, set_seed


@dataclass(frozen=True)
class DeploymentCondition:
    name: str
    phase_noise_std_rad: float = 0.0
    phase_shift_pixels: float = 0.0
    detector_noise_relative_rms: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Deployment condition name cannot be empty")
        if self.phase_noise_std_rad < 0.0:
            raise ValueError(f"Negative phase noise in {self.name}")
        if self.phase_shift_pixels < 0:
            raise ValueError(f"Negative phase shift in {self.name}")
        if self.detector_noise_relative_rms < 0.0:
            raise ValueError(f"Negative detector noise in {self.name}")


@dataclass(frozen=True)
class RobustnessSettings:
    config_path: Path
    formal_config: Path
    output_dir: Path
    split: str
    methods: tuple[str, ...]
    training_seeds: tuple[int, ...]
    deployment_seeds: tuple[int, ...]
    conditions: tuple[DeploymentCondition, ...]


def load_robustness_settings(path: Path) -> RobustnessSettings:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    conditions = tuple(DeploymentCondition(**item) for item in payload["conditions"])
    for condition in conditions:
        condition.validate()
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Deployment condition names must be unique")
    if "ideal" not in names:
        raise ValueError("Deployment conditions must include an ideal reference")
    methods = tuple(str(value) for value in payload["methods"])
    if set(methods) != set(METHODS):
        raise ValueError(f"Robustness study must use exactly the four formal methods: {METHODS}")
    split = str(payload["split"])
    if split not in {"validation", "test"}:
        raise ValueError("Robustness split must be validation or test")
    return RobustnessSettings(
        config_path=path,
        formal_config=Path(payload["formal_config"]),
        output_dir=Path(payload["output_dir"]),
        split=split,
        methods=methods,
        training_seeds=tuple(int(value) for value in payload["training_seeds"]),
        deployment_seeds=tuple(int(value) for value in payload["deployment_seeds"]),
        conditions=conditions,
    )


def _translate_phase(phase: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Translate a finite phase mask without wrapping; uncovered pixels use zero phase."""

    if dy == 0 and dx == 0:
        return phase.clone()
    height, width = phase.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        return torch.zeros_like(phase)
    if float(dy).is_integer() and float(dx).is_integer():
        integer_y = int(dy)
        integer_x = int(dx)
        output = torch.zeros_like(phase)
        source_y = slice(max(-integer_y, 0), min(height - integer_y, height))
        target_y = slice(max(integer_y, 0), min(height + integer_y, height))
        source_x = slice(max(-integer_x, 0), min(width - integer_x, width))
        target_x = slice(max(integer_x, 0), min(width + integer_x, width))
        output[..., target_y, target_x] = phase[..., source_y, source_x]
        return output

    # Interpolate the complex phasor rather than the wrapped phase angle.
    # This avoids an artificial discontinuity between values close to 0 and 2pi.
    phasor = torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)
    theta = torch.eye(2, 3, device=phase.device, dtype=phase.dtype)
    theta = theta.unsqueeze(0).repeat(phase.shape[0], 1, 1)
    theta[:, 0, 2] = -2.0 * float(dx) / max(width - 1, 1)
    theta[:, 1, 2] = -2.0 * float(dy) / max(height - 1, 1)
    grid = F.affine_grid(theta, phasor.shape, align_corners=True)
    shifted = F.grid_sample(
        phasor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return torch.remainder(torch.atan2(shifted[:, 1], shifted[:, 0]), 2.0 * math.pi)


def build_deployment_state(
    model: OpticalClassifier,
    condition: DeploymentCondition,
    *,
    deployment_seed: int,
    device: torch.device,
) -> tuple[OpticalDeploymentState, dict[str, object]]:
    condition.validate()
    phase_generator = torch.Generator().manual_seed(int(deployment_seed) + 11)
    shift_generator = torch.Generator().manual_seed(int(deployment_seed) + 23)
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    overrides: list[torch.Tensor] = []
    shifts: list[list[float]] = []
    phase_error_squares: list[torch.Tensor] = []
    has_static_error = condition.phase_noise_std_rad > 0.0 or condition.phase_shift_pixels > 0
    if has_static_error:
        for stage in model.stages:
            phase = stage.phase().detach()
            dy = dx = 0.0
            if condition.phase_shift_pixels > 0:
                direction_index = int(torch.randint(len(directions), (), generator=shift_generator))
                direction = directions[direction_index]
                dy = direction[0] * condition.phase_shift_pixels
                dx = direction[1] * condition.phase_shift_pixels
                phase = _translate_phase(phase, dy, dx)
            if condition.phase_noise_std_rad > 0.0:
                unit_error = torch.randn(stage.phase().shape, generator=phase_generator)
                error = unit_error.to(device=phase.device, dtype=phase.dtype)
                error = error * condition.phase_noise_std_rad
                phase = torch.remainder(phase + error, 2.0 * math.pi)
                phase_error_squares.append(error.detach().float().square().mean().cpu())
            overrides.append(phase)
            shifts.append([dy, dx])

    detector_generators: list[torch.Generator] = []
    if condition.detector_noise_relative_rms > 0.0:
        for stage_index in range(len(model.stages)):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(deployment_seed) * 1000 + stage_index)
            detector_generators.append(generator)

    actual_phase_rms = (
        float(torch.stack(phase_error_squares).mean().sqrt()) if phase_error_squares else 0.0
    )
    state = OpticalDeploymentState(
        phase_overrides=tuple(overrides),
        detector_noise_relative_rms=condition.detector_noise_relative_rms,
        detector_generators=tuple(detector_generators),
    )
    metadata = {
        "phase_error_actual_rms_rad": actual_phase_rms,
        "phase_shifts_dy_dx": shifts,
    }
    return state, metadata


@torch.inference_mode()
def evaluate_deployment(
    model: OpticalClassifier,
    loader,
    device: torch.device,
    *,
    deployment: OpticalDeploymentState,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, deployment=deployment)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_samples += int(targets.numel())
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": float(total_samples),
        "seconds": time.perf_counter() - started,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_one(
    settings: RobustnessSettings,
    *,
    method: str,
    training_seed: int,
    deployment_seed: int,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    output = (
        settings.output_dir
        / method
        / f"seed_{training_seed}"
        / f"deployment_seed_{deployment_seed}"
        / "result.json"
    )
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    formal = load_formal_settings(settings.formal_config)
    datasets = load_datasets(formal.base, download=False)
    evaluation_dataset = datasets.validation if settings.split == "validation" else datasets.test
    loader = make_loader(evaluation_dataset, formal.base, train=False, seed=training_seed)
    model = build_model(formal.base, device)
    checkpoint = (
        formal.base.output_dir / method / f"seed_{training_seed}" / "best.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    formal_result_path = checkpoint.with_name("result.json")
    formal_result = json.loads(formal_result_path.read_text(encoding="utf-8"))
    expected_ideal = (
        float(formal_result["best_validation_accuracy"])
        if settings.split == "validation"
        else float(formal_result["test"]["ablations"]["normal"]["accuracy"])
    )
    condition_results: list[dict[str, object]] = []
    ideal_accuracy: float | None = None
    for condition in settings.conditions:
        deployment, metadata = build_deployment_state(
            model,
            condition,
            deployment_seed=deployment_seed,
            device=device,
        )
        metrics = evaluate_deployment(
            model,
            loader,
            device,
            deployment=deployment,
            max_batches=formal.base.training.max_evaluation_batches,
        )
        if condition.name == "ideal":
            ideal_accuracy = metrics["accuracy"]
            if abs(ideal_accuracy - expected_ideal) > 1.0e-12:
                raise RuntimeError(
                    f"Ideal deployment mismatch for {method}/seed_{training_seed}: "
                    f"expected {expected_ideal}, got {ideal_accuracy}"
                )
        condition_results.append(
            {
                **asdict(condition),
                **metadata,
                **metrics,
            }
        )
        print(
            f"[deployment] method={method} train_seed={training_seed} "
            f"deploy_seed={deployment_seed} condition={condition.name} "
            f"accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )
    if ideal_accuracy is None:
        raise RuntimeError("No ideal deployment condition was evaluated")
    for row in condition_results:
        row["absolute_accuracy_drop"] = ideal_accuracy - float(row["accuracy"])
        row["accuracy_retention"] = float(row["accuracy"]) / max(ideal_accuracy, 1.0e-12)
    result = {
        "config": str(settings.config_path),
        "config_sha256": sha256_file(settings.config_path),
        "formal_config": str(settings.formal_config),
        "formal_checkpoint": str(checkpoint),
        "formal_checkpoint_sha256": sha256_file(checkpoint),
        "split": settings.split,
        "method": method,
        "training_seed": training_seed,
        "deployment_seed": deployment_seed,
        "expected_ideal_accuracy": expected_ideal,
        "conditions": condition_results,
    }
    _write_json(output, result)
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def compare(settings: RobustnessSettings) -> dict[str, object]:
    results: list[dict[str, object]] = []
    missing: list[str] = []
    for method in settings.methods:
        for training_seed in settings.training_seeds:
            for deployment_seed in settings.deployment_seeds:
                path = (
                    settings.output_dir
                    / method
                    / f"seed_{training_seed}"
                    / f"deployment_seed_{deployment_seed}"
                    / "result.json"
                )
                if not path.exists():
                    missing.append(str(path))
                else:
                    results.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError("Missing robustness results: " + ", ".join(missing))

    rows: list[dict[str, object]] = []
    for result in results:
        for condition in result["conditions"]:
            rows.append(
                {
                    "method": result["method"],
                    "training_seed": result["training_seed"],
                    "deployment_seed": result["deployment_seed"],
                    "condition": condition["name"],
                    "phase_noise_std_rad": condition["phase_noise_std_rad"],
                    "phase_shift_pixels": condition["phase_shift_pixels"],
                    "detector_noise_relative_rms": condition["detector_noise_relative_rms"],
                    "accuracy": condition["accuracy"],
                    "absolute_accuracy_drop": condition["absolute_accuracy_drop"],
                    "accuracy_retention": condition["accuracy_retention"],
                }
            )

    summary: dict[str, object] = {}
    for condition in settings.conditions:
        condition_summary: dict[str, object] = {}
        for method in settings.methods:
            method_rows = [
                row
                for row in rows
                if row["condition"] == condition.name and row["method"] == method
            ]
            condition_summary[method] = {
                "accuracy": _summary([float(row["accuracy"]) for row in method_rows]),
                "absolute_accuracy_drop": _summary(
                    [float(row["absolute_accuracy_drop"]) for row in method_rows]
                ),
                "accuracy_retention": _summary(
                    [float(row["accuracy_retention"]) for row in method_rows]
                ),
            }
        summary[condition.name] = condition_summary

    keyed = {
        (
            str(row["method"]),
            int(row["training_seed"]),
            int(row["deployment_seed"]),
            str(row["condition"]),
        ): float(row["accuracy"])
        for row in rows
    }
    paired: dict[str, object] = {}
    for left, right in (("bp", "fa_pretrained"), ("fa_pretrained", "fa_random")):
        contrast: dict[str, object] = {}
        for condition in settings.conditions:
            deltas = []
            for training_seed in settings.training_seeds:
                for deployment_seed in settings.deployment_seeds:
                    key = (training_seed, deployment_seed, condition.name)
                    deltas.append(
                        keyed[(left, *key)] - keyed[(right, *key)]
                    )
            contrast[condition.name] = _summary(deltas)
        paired[f"{left}_minus_{right}"] = contrast

    output = {
        "config": str(settings.config_path),
        "config_sha256": sha256_file(settings.config_path),
        "split": settings.split,
        "summary": summary,
        "paired_accuracy_deltas": paired,
        "rows": rows,
    }
    comparison_dir = settings.output_dir / "comparison"
    _write_json(comparison_dir / "comparison.json", output)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with (comparison_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed checkpoints under optical deployment errors")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "compare"), required=True)
    parser.add_argument("--methods", nargs="*", choices=METHODS)
    parser.add_argument("--training-seeds", nargs="*", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_robustness_settings(args.config)
    if args.phase == "compare":
        print(json.dumps(compare(settings), indent=2), flush=True)
        return
    methods = tuple(args.methods) if args.methods else settings.methods
    training_seeds = tuple(args.training_seeds) if args.training_seeds else settings.training_seeds
    if not set(methods).issubset(settings.methods):
        raise ValueError(f"Methods must be a subset of {settings.methods}")
    if not set(training_seeds).issubset(settings.training_seeds):
        raise ValueError(f"Training seeds must be a subset of {settings.training_seeds}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    for method in methods:
        for training_seed in training_seeds:
            for deployment_seed in settings.deployment_seeds:
                set_seed(training_seed)
                run_one(
                    settings,
                    method=method,
                    training_seed=training_seed,
                    deployment_seed=deployment_seed,
                    device=device,
                    force=args.force,
                )


if __name__ == "__main__":
    main()
