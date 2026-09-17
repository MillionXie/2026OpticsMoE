"""Reproduce the pinned simulation from a standalone LGVQ handoff.

This entry point deliberately consumes only files inside the extracted release:
the strict checkpoint, the exact model runtime, and the frozen Qwen-front test
fields.  It never downloads a model or silently rebuilds a dataset cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_release_files() -> None:
    manifest = json.loads((ROOT / "SHA256.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif _sha256(path) != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    if errors:
        raise RuntimeError("Release integrity check failed:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fields",
        type=int,
        default=0,
        help="0 reproduces all 35 fields / 558 videos; a positive value is only a smoke test.",
    )
    parser.add_argument("--output", default="simulation_reproduction.json")
    args = parser.parse_args()
    if args.fields < 0:
        parser.error("--fields must be nonnegative")

    _verify_release_files()

    import torch

    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import (
        forward,
        load_model,
    )
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import (
        regression_metrics,
    )

    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    if release["target"] != "temporal":
        raise ValueError("This handoff is not the pinned Temporal target")
    model, _ = load_model(
        release["target"], ROOT / "weights" / "best_checkpoint.pt", args.device
    )
    fields = release["fields"]
    if args.fields:
        fields = fields[: args.fields]

    predictions: list[float] = []
    targets: list[float] = []
    reference_predictions: list[float] = []
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for field_index, item in enumerate(fields):
            batch = torch.load(
                ROOT / item["file"], map_location="cpu", weights_only=False
            )
            scores = forward(model, batch)["prediction"].detach().cpu().reshape(-1)
            if scores.numel() != len(item["valid"]):
                raise RuntimeError(f"Slot count changed for {item['key']}")
            for slot, valid in enumerate(item["valid"]):
                if not valid:
                    continue
                score = float(scores[slot])
                reference = float(item["simulation_prediction"][slot])
                target = float(item["targets"][slot])
                predictions.append(score)
                reference_predictions.append(reference)
                targets.append(target)
                rows.append(
                    {
                        "field": item["key"],
                        "slot": slot,
                        "video": item["sample_ids"][slot],
                        "prediction": score,
                        "packaged_reference_prediction": reference,
                        "target": target,
                    }
                )
            print(f"SIMULATED {field_index + 1}/{len(fields)} {item['key']}", flush=True)

    prediction_tensor = torch.tensor(predictions)
    target_tensor = torch.tensor(targets)
    metrics = regression_metrics(prediction_tensor, target_tensor, "temporal")
    maximum_prediction_delta = max(
        abs(value - reference)
        for value, reference in zip(predictions, reference_predictions)
    )
    full_test = len(predictions) == 558
    reference_metrics = release["simulation_metrics"]
    # Different CUDA/PyTorch kernels are not bit-identical.  A 0.01 MOS
    # absolute tolerance is still negligible on the 0--100 label scale, while
    # the metric gates below protect the actual reported scientific result.
    tolerances = {
        "maximum_prediction_delta": 1.0e-2,
        "srcc": 1.5e-4,
        "krcc": 1.5e-4,
        "plcc": 1.0e-4,
        "rmse": 1.0e-3,
        "mae": 1.0e-3,
    }
    metric_deltas = {
        name: abs(float(metrics[name]) - float(reference_metrics[name]))
        for name in ("srcc", "krcc", "plcc", "rmse", "mae")
    }
    passed = maximum_prediction_delta <= tolerances["maximum_prediction_delta"]
    if full_test:
        passed = passed and all(
            metric_deltas[name] <= tolerances[name] for name in metric_deltas
        )
    report = {
        "status": "passed" if passed else "failed",
        "checkpoint_sha256": _sha256(ROOT / "weights" / "best_checkpoint.pt"),
        "device": args.device,
        "full_test": full_test,
        "field_count": len(fields),
        "video_count": len(predictions),
        "maximum_prediction_delta_vs_packaged_reference": maximum_prediction_delta,
        "metric_deltas_vs_packaged_reference": metric_deltas,
        "reproduction_tolerances": tolerances,
        "metrics": metrics,
        "packaged_reference_metrics": reference_metrics,
        "rows": rows,
    }
    output = ROOT / args.output
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not passed:
        raise RuntimeError(f"Simulation reproduction failed; inspect {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
