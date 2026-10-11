"""Validate and summarize a completed validation-only FFA LSP pilot.

Example:
    python -m FFA.summarize --run FFA/runs/<run> --out FFA/reports/<run>

Only JSON records are needed. No checkpoints, datasets, CUDA, or server access
are required. Matplotlib is imported only when rendering the static figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any


FFA_ROOT = Path(__file__).resolve().parent
METHODS = (
    "head_only", "bp_current", "fa_source_connector",
    "source_full_response", "random_full_response",
)
LABELS = {
    "head_only": "Head only",
    "bp_current": "Current BP",
    "fa_source_connector": "Source connector FA",
    "source_full_response": "Frozen source response",
    "random_full_response": "Frozen random response",
}
COLORS = {
    "head_only": "#777777", "bp_current": "#171717",
    "fa_source_connector": "#0072B2", "source_full_response": "#D55E00",
    "random_full_response": "#CC79A7",
}


def _read_json(path: Path, inputs: dict[str, str]) -> Any:
    content = path.read_bytes()  # Missing files are an error, never a zero score.
    inputs[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode("utf-8-sig"))


def _number(value: Any, name: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} is above {maximum}")
    return value


def _pck(value: Any, name: str) -> float:
    return _number(value, name, minimum=0.0, maximum=1.0)


def _same(actual: float, expected: float, name: str, tolerance=1e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{name} conflicts with the recorded history: {actual} vs {expected}")


def _history(value: Any, epochs: int, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != epochs:
        raise ValueError(f"{name} must contain exactly {epochs} completed epochs")
    if [row.get("epoch") for row in value] != list(range(1, epochs + 1)):
        raise ValueError(f"{name} has missing, duplicated, or unordered epochs")
    for row in value:
        _pck(row["validation"]["pck_torso_0p2"], f"{name} epoch {row['epoch']} PCK")
    return value


def _recovery(source: float, bp: float, head: float) -> dict[str, Any]:
    gain = bp - head
    return {
        "bp_gain_over_equal_budget_head_only_pp": 100.0 * gain,
        "source_gain_over_equal_budget_head_only_pp": 100.0 * (source - head),
        "source_recovery_of_bp_gain": (source - head) / gain if gain > 0.0 else None,
        "defined": gain > 0.0,
        "undefined_reason": None if gain > 0.0 else "BP has no positive gain over the equal-budget head-only control.",
        "ratio_is_not_clipped": True,
    }


def collect(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate recorded identities/metrics and return summary and CSV rows."""
    run = run.expanduser().resolve()
    inputs: dict[str, str] = {}
    result = _read_json(run / "result.json", inputs)
    if result.get("status") != "complete":
        raise ValueError("The pilot is not complete; do not summarize a partial run as a final result")
    for flag in ("test_evaluated", "hardware_experiment", "dynamic_feedback_formation"):
        if result.get(flag) is not False:
            raise ValueError(f"This summarizer requires an explicitly false {flag} flag")
    profile = result["profile"]
    if profile.get("task") != "lsp" or profile.get("methods") != list(METHODS):
        raise ValueError("Expected the paired five-method LSP pilot protocol")
    warm_epochs = profile["warmup_epochs"]
    adapt_epochs = profile["adaptation_epochs"]
    if not isinstance(warm_epochs, int) or warm_epochs <= 0 or not isinstance(adapt_epochs, int) or adapt_epochs <= 0:
        raise ValueError("Warmup and adaptation budgets must be positive integer epoch counts")
    warm = _history(_read_json(run / "warmup_history.json", inputs), warm_epochs, "warmup_history")
    common = _pck(result["common_validation_pck"], "selected warmup common validation PCK")
    warm_values = [float(row["validation"]["pck_torso_0p2"]) for row in warm]
    _same(common, max(warm_values), "common_validation_pck")
    root_records = result["results"]
    if not isinstance(root_records, list) or len(root_records) != len(METHODS):
        raise ValueError("Main result must contain exactly five method records")
    records = {record["method"]: record for record in root_records}
    if set(records) != set(METHODS):
        raise ValueError("Main result has missing, duplicated, or unknown methods")

    identities = ("shared_start_sha256", "dataset_manifest_sha256", "source_checkpoint_sha256")
    shared_identity = {key: records[METHODS[0]][key] for key in identities}
    if any(not isinstance(value, str) or len(value) != 64 for value in shared_identity.values()):
        raise ValueError("Shared/source/dataset identities must contain SHA-256 strings")
    method_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    for row in warm:
        curves.append({
            "phase": "head_warmup", "method": "shared_head", "epoch": row["epoch"],
            "validation_pck": float(row["validation"]["pck_torso_0p2"]),
            "validation_pck_percent": 100.0 * float(row["validation"]["pck_torso_0p2"]),
            "training_loss": _number(row["loss"], "warmup training loss"),
            "phase_mean_absolute_rad": None, "phase_rms_rad": None,
        })

    for method in METHODS:
        record = records[method]
        if record.get("status") != "complete" or record.get("test_evaluated") is not False:
            raise ValueError(f"{method} is not a complete validation-only result")
        if record.get("frozen_electronic_parameters_unchanged") is not True:
            raise ValueError(f"{method} lacks a passed frozen-electronics check")
        if record["epochs"] != adapt_epochs:
            raise ValueError(f"{method} has an unequal adaptation budget")
        for key, value in shared_identity.items():
            if record[key] != value:
                raise ValueError(f"{method} differs in {key}")
        history = _history(_read_json(run / method / "history.json", inputs), adapt_epochs, method)
        baseline = _read_json(run / method / "baseline_validation.json", inputs)
        baseline_pck = _pck(baseline["pck_torso_0p2"], f"{method} epoch-0 validation PCK")
        best = _pck(record["best_validation_pck"], f"{method} best PCK")
        last = _pck(record["last_validation_pck"], f"{method} last PCK")
        all_pck = [baseline_pck] + [float(row["validation"]["pck_torso_0p2"]) for row in history]
        _same(best, max(all_pck), f"{method} best validation PCK")
        _same(last, all_pck[-1], f"{method} last validation PCK")
        best_epoch = record["best_epoch"]
        if not isinstance(best_epoch, int) or not 0 <= best_epoch <= adapt_epochs:
            raise ValueError(f"{method} best epoch is outside 0..{adapt_epochs}")
        _same(best, all_pck[best_epoch], f"{method} selected epoch PCK")
        curves.append({
            "phase": "paired_adaptation", "method": method, "epoch": 0,
            "validation_pck": baseline_pck, "validation_pck_percent": 100.0 * baseline_pck,
            "training_loss": None, "phase_mean_absolute_rad": None, "phase_rms_rad": None,
        })
        for row in history:
            phase = row["phase"]
            rms_values = phase["per_stage_rms_rad"]
            rms = math.sqrt(sum(_number(v, "phase RMS", minimum=0.0) ** 2 for v in rms_values) / len(rms_values))
            curves.append({
                "phase": "paired_adaptation", "method": method, "epoch": row["epoch"],
                "validation_pck": float(row["validation"]["pck_torso_0p2"]),
                "validation_pck_percent": 100.0 * float(row["validation"]["pck_torso_0p2"]),
                "training_loss": _number(row["training_loss"], f"{method} training loss"),
                "phase_mean_absolute_rad": _number(phase["mean_absolute_rad"], "phase mean absolute drift", minimum=0.0),
                "phase_rms_rad": rms,
            })
        phase = history[-1]["phase"]
        means = [_number(v, "stage phase mean drift", minimum=0.0) for v in phase["per_stage_mean_absolute_rad"]]
        rms_values = [_number(v, "stage phase RMS drift", minimum=0.0) for v in phase["per_stage_rms_rad"]]
        if len(means) != 8 or len(rms_values) != 8:
            raise ValueError(f"{method} must contain all eight P11 stage drift values")
        phase_mean = _number(phase["mean_absolute_rad"], "terminal phase mean drift", minimum=0.0)
        phase_rms = math.sqrt(sum(v * v for v in rms_values) / len(rms_values))
        terminal = record["last_phase_diagnostic"]
        if method == "head_only":
            if terminal is not None:
                raise ValueError("Head-only group must not claim a phase-gradient diagnostic")
            if phase_mean > 1e-7 or phase_rms > 1e-7:
                raise ValueError("Head-only control moved a frozen optical phase")
            terminal_cosine = None
            per_stage_cosine = [None] * 8
        else:
            if not isinstance(terminal, dict) or len(terminal["per_stage"]) != 8:
                raise ValueError(f"{method} lacks a complete terminal phase-gradient diagnostic")
            terminal_cosine = _number(terminal["phase"]["cosine"], "terminal overall gradient cosine", minimum=-1.000001, maximum=1.000001)
            per_stage_cosine = [_number(stage["cosine"], "terminal stage gradient cosine", minimum=-1.000001, maximum=1.000001)
                                for stage in terminal["per_stage"]]
        for index, (mean, rms, cosine) in enumerate(zip(means, rms_values, per_stage_cosine), start=1):
            gradients.append({
                "method": method, "stage": index,
                "optical_axis": "token" if index % 2 else "channel",
                "terminal_gradient_cosine_to_current_bp": cosine,
                "terminal_mean_absolute_phase_drift_rad": mean,
                "terminal_rms_phase_drift_rad": rms,
                "gradient_defined": method != "head_only",
            })
        method_rows.append({
            "method": method, "label": LABELS[method], "baseline_validation_pck": baseline_pck,
            "best_validation_pck": best, "best_validation_pck_percent": 100.0 * best,
            "best_epoch": best_epoch, "last_validation_pck": last,
            "last_validation_pck_percent": 100.0 * last,
            "best_gain_over_shared_warmup_pp": 100.0 * (best - common),
            "last_gain_over_shared_warmup_pp": 100.0 * (last - common),
            "terminal_phase_mean_absolute_rad": phase_mean,
            "terminal_phase_rms_rad": phase_rms,
            "terminal_gradient_cosine_to_current_bp": terminal_cosine,
            "terminal_stage_gradient_cosines": per_stage_cosine,
            "skipped_optimizer_steps": sum(row["skipped_optimizer_steps"] for row in history),
            "completed_adaptation_epochs": adapt_epochs,
        })

    by_method = {row["method"]: row for row in method_rows}
    recoveries = {}
    for selection in ("best", "last"):
        key = f"{selection}_validation_pck"
        recoveries[selection] = _recovery(by_method["source_full_response"][key],
                                         by_method["bp_current"][key], by_method["head_only"][key])
    _same(_number(result["bp_gain_over_equal_budget_head_only"], "reported BP gain"),
          recoveries["best"]["bp_gain_over_equal_budget_head_only_pp"] / 100.0, "reported BP gain")
    reported_recovery = result["source_full_response_recovery"]
    if recoveries["best"]["defined"]:
        _same(_number(reported_recovery, "reported source recovery"),
              recoveries["best"]["source_recovery_of_bp_gain"], "reported source recovery")
    elif reported_recovery is not None:
        raise ValueError("Source recovery is reported despite nonpositive BP gain")
    baselines = [row["baseline_validation_pck"] for row in method_rows]
    summary = {
        "format": "ffa-validation-pilot-summary-v1", "status": "complete",
        "run_path": str(run), "task": "lsp", "seed": profile["seed"],
        "metric": "Validation PCK@0.2 torso", "metric_unit": "fraction; display values additionally in percent",
        "profile": profile, "shared_identity": shared_identity,
        "shared_warmup_baseline": {
            "selected_validation_pck": common, "selected_validation_pck_percent": 100.0 * common,
            "selected_epoch": warm_values.index(max(warm_values)) + 1,
            "completed_head_warmup_epochs": warm_epochs,
            "body_updates_during_warmup": False, "fixed_training_images_during_warmup": True,
            "adaptation_epoch0_pck_spread_pp": 100.0 * (max(baselines) - min(baselines)),
            "note": "Warmup and adaptation have separate epoch axes and separate update scopes.",
        },
        "methods": method_rows, "source_full_response_recovery": recoveries,
        "phase_drift_definition": "Circular displacement from source; equal-size P11 stage RMS values give overall RMS. This is not an unwrapped accumulated path length.",
        "gradient_definition": "Terminal candidate and exact current-BP gradients at the same model and diagnostic batch; head-only has no phase-gradient diagnostic.",
        "best_selection": "Validation-selected within epochs 0..adaptation_epochs; epoch 0 preserves the shared start.",
        "claim_scope": {
            "validation_development_pilot_only": True, "test_evaluated": False,
            "independent_generalization_assessed": False, "hardware_experiment": False,
            "dynamic_feedback_formation": False, "pretraining_sources": 1,
            "limitations": "Single-source, single-seed, reduced-data short adaptation; no statistical equivalence or hardware-speedup claim.",
        },
        "git_commit": result["git_commit"], "gpu_uuid": result["gpu_uuid"],
        "input_sha256": inputs,
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return summary, curves, gradients


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot export an empty table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue()


def render(summary: dict[str, Any], curves: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "savefig.dpi": 200}):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
        warm = [row for row in curves if row["phase"] == "head_warmup"]
        axes[0].plot([row["epoch"] for row in warm], [row["validation_pck_percent"] for row in warm],
                     color="#0072B2", linewidth=1.8, label="Shared head warmup")
        base = summary["shared_warmup_baseline"]
        axes[0].scatter([base["selected_epoch"]], [base["selected_validation_pck_percent"]],
                        marker="*", s=110, color="#D55E00", zorder=4, label="Selected shared start")
        axes[0].set_title("A  Head warmup: optical body frozen", loc="left", fontsize=11)
        axes[0].set_xlabel("Head warmup epoch (fixed source features)")
        axes[0].set_ylabel("Validation PCK@0.2 torso (%)")
        axes[0].legend(frameon=False, fontsize=9)
        for method in METHODS:
            rows = [row for row in curves if row["phase"] == "paired_adaptation" and row["method"] == method]
            axes[1].plot([row["epoch"] for row in rows], [row["validation_pck_percent"] for row in rows],
                         color=COLORS[method], linewidth=1.7, marker="o", markersize=3,
                         label=LABELS[method], linestyle="--" if method == "head_only" else "-")
        axes[1].set_title("B  Paired adaptation from the shared start", loc="left", fontsize=11)
        axes[1].set_xlabel("Adaptation epoch (same sample/update budget)")
        axes[1].set_ylabel("Validation PCK@0.2 torso (%)")
        axes[1].set_xticks(range(summary["profile"]["adaptation_epochs"] + 1))
        axes[1].legend(frameon=False, fontsize=8.5, loc="best")
        for ax in axes:
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)
        fig.suptitle("LSP validation development pilot", fontsize=13, y=1.0)
        fig.text(0.5, 0.015, "Panels use separate y scales. Single source/seed, reduced data; validation only. No hardware or feedback formation assessment.",
                 ha="center", fontsize=8.5, color="#555555")
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            fig.savefig(temporary, format="png", bbox_inches="tight")
            os.replace(temporary, path)
        finally:
            plt.close(fig)
            if temporary.exists():
                temporary.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a completed validation-only FFA pilot; no server access.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory inside FFA")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out != FFA_ROOT and FFA_ROOT not in out.parents:
        raise ValueError("Output directory must remain inside this repository's FFA directory")
    summary, curves, gradients = collect(args.run)
    out.mkdir(parents=True, exist_ok=True)
    render(summary, curves, out / "learning_curves.png")
    _write_atomic(out / "summary.csv", _csv_text(summary["methods"]))
    _write_atomic(out / "learning_curves.csv", _csv_text(curves))
    _write_atomic(out / "terminal_gradients.csv", _csv_text(gradients))
    summary["exports"] = ["summary.json", "summary.csv", "learning_curves.csv",
                          "terminal_gradients.csv", "learning_curves.png"]
    _write_atomic(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "output": str(out), "files": summary["exports"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
