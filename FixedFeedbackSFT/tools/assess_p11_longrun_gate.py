#!/usr/bin/env python3
"""Assess pre-registered recovery gates for the exploratory P11 long run.

The tool is deliberately read-only.  It never signals, stops, resumes or
modifies a training process; it turns the completed epoch history into a
machine-readable recommendation for manual review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


BASELINE_TOP1 = 0.51350
GATES = (
    (10, 0.49000),
    (20, 0.50500),
    (30, BASELINE_TOP1),
)


def _slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    count = float(len(values))
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / count
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator if denominator else None


def assess(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "status": "waiting",
            "decision": "wait_for_epoch_10",
            "reason": "no completed training epoch",
        }
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"History epochs are not consecutive from one: {epochs}")
    raw = [float(row["validation_raw"]["top1_accuracy"]) for row in history]
    ema = [float(row["validation_ema"]["top1_accuracy"]) for row in history]
    if not all(math.isfinite(value) for value in (*raw, *ema)):
        raise ValueError("History contains non-finite validation accuracy")

    latest_epoch = epochs[-1]
    window = raw[-min(5, len(raw)) :]
    recent_slope = _slope(window)
    reached = [(epoch, threshold) for epoch, threshold in GATES if latest_epoch >= epoch]
    result: dict[str, Any] = {
        "status": "assessed",
        "latest_epoch": latest_epoch,
        "baseline_top1": BASELINE_TOP1,
        "latest_raw_top1": raw[-1],
        "latest_ema_top1": ema[-1],
        "best_trained_raw_top1": max(raw),
        "best_trained_ema_top1": max(ema),
        "raw_gap_to_baseline": raw[-1] - BASELINE_TOP1,
        "recent_raw_slope_per_epoch": recent_slope,
        "slope_window_epochs": epochs[-len(window) :],
        "gates": [
            {"epoch": epoch, "minimum_raw_top1": threshold}
            for epoch, threshold in GATES
        ],
        "automatic_process_action": "none",
    }
    if not reached:
        result.update(
            {
                "decision": "wait_for_epoch_10",
                "reason": "first recovery gate has not been reached",
            }
        )
        return result

    gate_epoch, minimum = reached[-1]
    if raw[-1] < minimum:
        result.update(
            {
                "decision": "manual_review_stop_candidate",
                "active_gate_epoch": gate_epoch,
                "reason": (
                    f"latest raw Top-1 {raw[-1]:.6f} is below the "
                    f"gate minimum {minimum:.6f}"
                ),
            }
        )
    elif recent_slope is None or recent_slope <= 0.0:
        result.update(
            {
                "decision": "manual_review_stop_candidate",
                "active_gate_epoch": gate_epoch,
                "reason": "recent raw Top-1 does not have a positive slope",
            }
        )
    else:
        next_gate = next(
            (epoch for epoch, _ in GATES if epoch > latest_epoch),
            None,
        )
        result.update(
            {
                "decision": "continue_to_next_gate",
                "active_gate_epoch": gate_epoch,
                "next_gate_epoch": next_gate,
                "reason": "accuracy floor and positive-slope requirements are met",
            }
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.history.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("History JSON must contain a list")
    result = assess(payload)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
