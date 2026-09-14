from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .settings import METHODS, TASKS


def summarize(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in root.glob("*/*/seed_*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        rows.append(
            {
                "task": result["task"],
                "category": result["category"],
                "method": result["method"],
                "seed": int(result["seed"]),
                "top1": float(result["test"]["top1"]),
                "balanced_accuracy": float(result["test"]["balanced_accuracy"]),
                "phase_mean_absolute_rad": float(result["phase"]["mean_absolute_rad"]),
                "wall_seconds": float(result["wall_seconds"]),
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"])].append(row)
    groups = []
    for (task, method), values in sorted(grouped.items()):
        scores = [value["top1"] for value in values]
        groups.append(
            {
                "task": task,
                "method": method,
                "n": len(values),
                "top1_mean": statistics.mean(scores),
                "top1_sample_sd": statistics.stdev(scores) if len(scores) > 1 else None,
                "phase_mean_absolute_rad": statistics.mean(
                    value["phase_mean_absolute_rad"] for value in values
                ),
            }
        )
    expected = {(task, method) for task in TASKS for method in METHODS}
    complete = set(grouped)
    return {
        "format": "p14-vtab1k-summary-v1",
        "root": str(root.resolve()),
        "complete_runs": len(rows),
        "complete_task_method_cells": len(complete),
        "missing_task_method_cells": [list(key) for key in sorted(expected - complete)],
        "rows": rows,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(payload["rows"][0]) if payload["rows"] else ["task", "method", "seed"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(json.dumps({key: payload[key] for key in ("complete_runs", "missing_task_method_cells")}, indent=2))


if __name__ == "__main__":
    main()
