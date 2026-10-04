"""Audit optical contribution and a random final-phase control.

The random control changes exactly one deployable optical parameter:
``serial_optics.raw_global_phase`` (the Language-global/final feature plane).
All routers, earlier phase planes, fusion coefficients, electronic residuals and
the readout remain bit-identical to the source checkpoint.  Several predeclared
seeds are evaluated because a single random phase is a noisy control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from .data import load_single_metric_cache
from .modeling import build_model
from .settings import load_settings
from .training import _loader, evaluate


TARGET_PARAMETER = "serial_optics.raw_global_phase"
CORE_METRICS = ("srcc", "krcc", "plcc", "rmse", "mae")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def uniform_physical_phase_raw(
    shape: Iterable[int], *, seed: int, dtype: torch.dtype
) -> torch.Tensor:
    """Return raw logits whose modeled phase is uniform on ``[0, 2 pi)``."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    # Avoid exact sigmoid endpoints while retaining a practically uniform phase.
    fraction = torch.rand(tuple(shape), generator=generator, dtype=torch.float64)
    epsilon = torch.finfo(torch.float32).eps
    fraction = fraction.clamp(epsilon, 1.0 - epsilon)
    return torch.logit(fraction).to(dtype=dtype)


def _core(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(metrics[name]) for name in CORE_METRICS}


def _summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in CORE_METRICS:
        values = torch.tensor(
            [float(record["metrics"][metric]) for record in records],
            dtype=torch.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(unbiased=False)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def _predictions(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = [float(row["prediction"]) for row in csv.DictReader(handle)]
    return torch.tensor(values, dtype=torch.float64)


def _prediction_change(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise RuntimeError("Prediction files have different or empty sample sets")
    left = reference - reference.mean()
    right = candidate - candidate.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    difference = candidate - reference
    return {
        "pcc_vs_learned_prediction": float((left * right).sum() / denominator),
        "delta_rmse": float(difference.square().mean().sqrt()),
        "delta_mae": float(difference.abs().mean()),
        "delta_max_abs": float(difference.abs().max()),
    }


def run(
    *,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    seeds: list[int],
    device_name: str | None,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one fixed random seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Random seeds must be unique")

    config = config.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(config)
    selected_device = device_name or settings.device
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(selected_device)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError("Checkpoint architecture does not match the config")
    if saved.get("target_name") != settings.target_name:
        raise RuntimeError("Checkpoint target does not match the config")
    state = saved["state_dict"]
    if TARGET_PARAMETER not in state:
        raise RuntimeError(f"Checkpoint is missing {TARGET_PARAMETER}")

    payload = load_single_metric_cache(settings)
    loader = _loader(payload, "test", settings, shuffle=False)
    model = build_model(settings).to(device)
    model.load_state_dict(state, strict=True)
    parameter = dict(model.named_parameters())[TARGET_PARAMETER]
    learned_raw = parameter.detach().cpu().clone()
    learned_raw_sha256 = _tensor_sha256(learned_raw)

    learned_prediction_path = output_dir / "predictions_optical_on.csv"
    learned_metrics = evaluate(
        model,
        loader,
        device,
        optical_enabled=True,
        prediction_path=learned_prediction_path,
    )
    learned_predictions = _predictions(learned_prediction_path)
    bypass_metrics = evaluate(
        model,
        loader,
        device,
        optical_enabled=False,
        prediction_path=output_dir / "predictions_optical_off.csv",
    )

    random_records: list[dict[str, Any]] = []
    reference_random_state: dict[str, torch.Tensor] | None = None
    for seed in seeds:
        random_raw = uniform_physical_phase_raw(
            learned_raw.shape, seed=seed, dtype=learned_raw.dtype
        )
        with torch.no_grad():
            parameter.copy_(random_raw.to(device=parameter.device))
        random_prediction_path = (
            output_dir / f"predictions_random_last_phase_seed{seed}.csv"
        )
        metrics = evaluate(
            model,
            loader,
            device,
            optical_enabled=True,
            prediction_path=random_prediction_path,
        )
        physical_phase = 2.0 * math.pi * torch.sigmoid(random_raw.float())
        record = {
            "seed": int(seed),
            "metrics": _core(metrics),
            "prediction_change": _prediction_change(
                learned_predictions, _predictions(random_prediction_path)
            ),
            "raw_phase_sha256": _tensor_sha256(random_raw),
            "physical_phase_rad": {
                "minimum": float(physical_phase.min()),
                "maximum": float(physical_phase.max()),
                "mean": float(physical_phase.mean()),
                "std_population": float(physical_phase.std(unbiased=False)),
            },
        }
        random_records.append(record)
        print(
            f"seed={seed} random_last_phase_SRCC={metrics['srcc']:.9f}",
            flush=True,
        )
        if seed == seeds[0]:
            reference_random_state = {
                name: value.detach().cpu().clone() for name, value in state.items()
            }
            reference_random_state[TARGET_PARAMETER] = random_raw

    with torch.no_grad():
        parameter.copy_(learned_raw.to(device=parameter.device))
    restored_metrics = evaluate(model, loader, device, optical_enabled=True)
    for metric in CORE_METRICS:
        if not math.isclose(
            float(restored_metrics[metric]),
            float(learned_metrics[metric]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError(f"Learned-state restoration changed {metric}")

    assert reference_random_state is not None
    reference_seed = int(seeds[0])
    random_checkpoint = output_dir / f"random_last_phase_seed{reference_seed}.pt"
    random_payload = {
        "schema_version": 1,
        "architecture": saved["architecture"],
        "target_name": saved["target_name"],
        "prompt": saved.get("prompt", settings.prompt),
        "epoch": saved.get("epoch", -1),
        "state_dict": reference_random_state,
        "settings": saved.get("settings"),
        "selection_policy": "fixed-seed random-phase control; not a selected model",
        "random_phase_ablation": {
            "target_parameter": TARGET_PARAMETER,
            "physical_plane": "Language global; final/fourth feature phase plane",
            "distribution": "independent Uniform[0, 2*pi) physical phase",
            "seed": reference_seed,
            "all_other_state_tensors_unchanged": True,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": _sha256(checkpoint),
        },
    }
    torch.save(random_payload, random_checkpoint)

    random_summary = _summary(random_records)
    report = {
        "schema_version": 1,
        "comparison_contract": {
            "same_checkpoint": True,
            "same_test_split": True,
            "test_count": int(learned_metrics.get("count", len(loader.dataset))),
            "target_parameter": TARGET_PARAMETER,
            "target_plane": "Language global; final/fourth feature phase plane",
            "randomization": "Uniform[0, 2*pi) physical phase via inverse sigmoid",
            "unchanged": [
                "all optical routers",
                "the other five learned phase tensors",
                "four fusion alpha values",
                "electronic residual branch",
                "post-optical readout",
                "20% evaluation zero-order leakage setting",
            ],
            "primary_random_seed_predeclared_not_cherry_picked": reference_seed,
        },
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "learned_last_phase_sha256": learned_raw_sha256,
        "normal_optical_electronic": _core(learned_metrics),
        "same_checkpoint_all_optics_bypassed": _core(bypass_metrics),
        "learned_on_minus_bypass": {
            metric: float(learned_metrics[metric]) - float(bypass_metrics[metric])
            for metric in CORE_METRICS
        },
        "random_last_phase_runs": random_records,
        "random_last_phase_summary": random_summary,
        "learned_minus_random_mean": {
            metric: float(learned_metrics[metric])
            - random_summary[metric]["mean"]
            for metric in CORE_METRICS
        },
        "reference_random_checkpoint": str(random_checkpoint),
        "reference_random_checkpoint_sha256": _sha256(random_checkpoint),
        "restoration_verified": True,
    }
    _write_json(output_dir / "last_phase_ablation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1346, 1347, 1348, 1349, 1350]
    )
    parser.add_argument("--device", dest="device_name")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
