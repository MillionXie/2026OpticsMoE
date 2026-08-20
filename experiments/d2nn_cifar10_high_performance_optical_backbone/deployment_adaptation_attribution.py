from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .datasets import load_datasets, make_loader
from .deployment_adaptation import evaluate, load_adaptation_settings
from .deployment_robustness import build_differentiable_deployment_state
from .fixed_feedback_training import sha256_file
from .formal_settings import load_formal_settings
from .training import build_model


ADAPTED_METHODS = ("bp", "fa_pretrained", "fa_random")
ATTRIBUTION_MODES = ("full", "phase_updates_only", "electronic_updates_only")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _mixed_state(
    source: dict[str, torch.Tensor],
    adapted: dict[str, torch.Tensor],
    mode: str,
) -> dict[str, torch.Tensor]:
    if mode == "full":
        return adapted
    if mode == "phase_updates_only":
        return {
            name: adapted[name] if name.endswith("raw_phase") else value
            for name, value in source.items()
        }
    if mode == "electronic_updates_only":
        return {
            name: source[name] if name.endswith("raw_phase") else value
            for name, value in adapted.items()
        }
    raise ValueError(f"Unsupported attribution mode: {mode}")


def run(
    config: Path,
    *,
    methods: tuple[str, ...],
    condition_names: tuple[str, ...],
) -> dict[str, object]:
    settings = load_adaptation_settings(config)
    formal = load_formal_settings(settings.formal_config)
    datasets = load_datasets(formal.base, download=False)
    evaluation_dataset = (
        datasets.validation if settings.evaluation_split == "validation" else datasets.test
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condition_lookup = {condition.name: condition for condition in settings.conditions}
    unknown = sorted(set(condition_names) - set(condition_lookup))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    rows: list[dict[str, object]] = []
    for training_seed in settings.training_seeds:
        source_checkpoint = (
            formal.base.output_dir
            / settings.source_method
            / f"seed_{training_seed}"
            / "best.pt"
        )
        source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)["model"]
        source_sha = sha256_file(source_checkpoint)
        for deployment_seed in settings.deployment_seeds:
            loader = make_loader(
                evaluation_dataset,
                formal.base,
                train=False,
                seed=training_seed * 100_000 + deployment_seed + 2,
            )
            for condition_name in condition_names:
                condition = condition_lookup[condition_name]
                for method in methods:
                    checkpoint = (
                        settings.output_dir
                        / method
                        / f"seed_{training_seed}"
                        / f"deployment_seed_{deployment_seed}"
                        / condition_name
                        / "best.pt"
                    )
                    adapted = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
                    for mode in ATTRIBUTION_MODES:
                        model = build_model(formal.base, device)
                        model.load_state_dict(_mixed_state(source, adapted, mode), strict=True)
                        deployment, _ = build_differentiable_deployment_state(
                            model,
                            condition,
                            deployment_seed=deployment_seed,
                            device=device,
                        )
                        metrics = evaluate(
                            model,
                            loader,
                            device,
                            deployment=deployment,
                            max_batches=formal.base.training.max_evaluation_batches,
                        )
                        row = {
                            "method": method,
                            "training_seed": training_seed,
                            "deployment_seed": deployment_seed,
                            "condition": condition_name,
                            "mode": mode,
                            "accuracy": metrics["accuracy"],
                            "loss": metrics["loss"],
                            "source_checkpoint_sha256": source_sha,
                            "adapted_checkpoint": str(checkpoint),
                            "adapted_checkpoint_sha256": sha256_file(checkpoint),
                        }
                        rows.append(row)
                        print(
                            f"[attribution] condition={condition_name} method={method} "
                            f"mode={mode} accuracy={metrics['accuracy']:.4f}",
                            flush=True,
                        )

    summaries: dict[str, object] = {}
    for condition_name in condition_names:
        summaries[condition_name] = {}
        for method in methods:
            summaries[condition_name][method] = {}
            for mode in ATTRIBUTION_MODES:
                values = [
                    float(row["accuracy"])
                    for row in rows
                    if row["condition"] == condition_name
                    and row["method"] == method
                    and row["mode"] == mode
                ]
                summaries[condition_name][method][mode] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
    payload = {
        "config": str(config),
        "config_sha256": sha256_file(config),
        "evaluation_split": settings.evaluation_split,
        "definitions": {
            "phase_updates_only": "Adapted raw optical phases; source electronic head, residual, and gates.",
            "electronic_updates_only": "Source raw optical phases; adapted electronic head, residual, and gates.",
        },
        "rows": rows,
        "summaries": summaries,
    }
    _write_json(settings.output_dir / "attribution" / "attribution.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute deployment recovery to optical/electronic updates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=ADAPTED_METHODS, default=ADAPTED_METHODS)
    parser.add_argument("--conditions", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_adaptation_settings(args.config)
    conditions = (
        tuple(args.conditions)
        if args.conditions
        else tuple(condition.name for condition in settings.conditions)
    )
    payload = run(args.config, methods=tuple(args.methods), condition_names=conditions)
    print(json.dumps(payload["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
