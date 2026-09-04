from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "d2nn_continuous",
    "d2nn_oeo_sigmoid",
    "moe_continuous_fixed_router",
    "moe_oeo_dynamic_router",
    "moe_oeo_fixed_router",
)
PRIMARY_VARIANTS = set(VARIANTS[:4])


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build four-primary-plus-supplement comparison tables"
    )
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    router_rows = []
    power_rows = []
    for variant in VARIANTS:
        run = args.runs_root / variant
        architecture = _read(run / "architecture_report.json")
        test = _read(run / "student_metrics.json")
        if not test:
            test = _read(run / "metrics" / "student_metrics.json")
        latest = _read(run / "metrics" / "training_latest.json")
        best_observed = _read(run / "metrics" / "best_observed_test.json")
        ema_best_observed = _read(run / "metrics" / "ema_best_observed_test.json")
        diagnostics = _read(run / "metrics" / "multiplane_diagnostics.json")
        rows.append(
            {
                "variant": variant,
                "comparison_role": (
                    "primary" if variant in PRIMARY_VARIANTS else "supplemental"
                ),
                "optical_family": architecture.get("optical_family"),
                "phase_planes_per_modality": architecture.get("phase_planes_per_modality"),
                "intermediate_ccd_boundaries": architecture.get("intermediate_square_law_boundaries"),
                "router_calls_per_modality": architecture.get("router_calls_per_modality"),
                "phase_parameters": architecture.get("parameter_counts", {}).get("optical_phase"),
                "total_trainable_parameters": architecture.get("parameter_counts", {}).get("trainable_total"),
                "train_loss": latest.get("total_loss"),
                "train_top1": latest.get("train_top1"),
                "selected_checkpoint_test_top1": test.get("top1_retrieval_accuracy"),
                "selected_checkpoint_test_top3": test.get("top3_retrieval_accuracy"),
                "selected_checkpoint_test_mrr": test.get("mrr"),
                "final_epoch_test_top1": latest.get("test_top1"),
                "final_epoch_test_top3": latest.get("test_top3"),
                "final_epoch_test_mrr": latest.get("test_mrr"),
                "best_observed_test_top1": best_observed.get("test_top1"),
                "best_observed_test_epoch": best_observed.get("epoch"),
                "ema_best_observed_test_top1": ema_best_observed.get("test_top1"),
                "ema_best_observed_test_epoch": ema_best_observed.get("epoch"),
                "phase_delta_rms_rad": latest.get("phase_delta_run_rms_rad"),
                "epoch_time_sec": latest.get("epoch_time_sec"),
            }
        )
        for modality in ("vision", "language"):
            values = diagnostics.get(modality, {})
            for router in values.get("routers", []):
                selections = router.get("selection_rate", [])
                importance = router.get("mean_probability_importance", [])
                weights = router.get("mean_sparse_weight", [])
                for expert in range(max(len(selections), len(importance), len(weights))):
                    router_rows.append(
                        {
                            "variant": variant,
                            "modality": modality,
                            "router_stage": router.get("stage"),
                            "expert": expert + 1,
                            "selection_rate": selections[expert] if expert < len(selections) else None,
                            "mean_probability_importance": importance[expert] if expert < len(importance) else None,
                            "mean_sparse_weight": weights[expert] if expert < len(weights) else None,
                        }
                    )
            for stage in values.get("optical_stages", []):
                power_rows.append(
                    {
                        "variant": variant,
                        "modality": modality,
                        "optical_stage": stage.get("stage"),
                        "mean_input_power": stage.get("mean_input_power"),
                        "mean_output_or_reload_power": stage.get("mean_output_or_reload_power"),
                        "final_ccd_mean": values.get("final_ccd_mean"),
                    }
                )
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"schema_version": 1, "variants": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, values in (
        ("router_diagnostics.csv", router_rows),
        ("power_diagnostics.csv", power_rows),
    ):
        if not values:
            continue
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
