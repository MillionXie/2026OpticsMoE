"""Fit the formal 4x4 frozen-D2NN/target-Linear transfer matrix.

The optical tensors in each row stay fixed.  Every cell fits exactly one
target Linear(784, C) on the complete target training split, selects it on the
target validation split, and evaluates it once on the target test split.
"""
import argparse
import json
from pathlib import Path

import torch

from .run import (TASK_ORDER, build_model, fit_mlp_probe, load_tasks, save,
                  selection_score)


def parse_checkpoints(items):
    paths = {}
    for item in items:
        task, raw_path = item.split("=", 1)
        paths[task] = Path(raw_path)
    missing = set(TASK_ORDER) - set(paths)
    if missing:
        raise ValueError(f"missing checkpoints: {sorted(missing)}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="TASK=best_checkpoint.pt; provide all four tasks")
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    checkpoint_paths = parse_checkpoints(args.checkpoint)
    tasks = load_tasks({
        "eurosat": args.eurosat, "clevr": args.clevr,
        "speech": args.speech, "physical": args.physical,
    }, require_full=True)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "status.json", {"status": "running"})
    details = {}
    scores = {}
    try:
        for source_index, source in enumerate(TASK_ORDER):
            row_path = args.out / f"{source}.json"
            row = json.loads(row_path.read_text()) if row_path.exists() else {}
            checkpoint = torch.load(checkpoint_paths[source], map_location=device,
                                    weights_only=False)
            model = build_model("d2nn", cfg, cfg["seed"] + source_index,
                                max_experts=4).to(device)
            model.load_state_dict(checkpoint["model"])
            model.requires_grad_(False)
            for target_index, target in enumerate(TASK_ORDER):
                if target in row:
                    continue
                probe = fit_mlp_probe(model, tasks[target], cfg, device,
                                      cfg["seed"] + source_index * 100 + target_index)
                row[target] = probe
                save(row_path, row)
                print(json.dumps({
                    "source_optics": source,
                    "target_linear": target,
                    "selected_head_epoch": probe["selected_epoch"],
                    "test_balanced_accuracy": selection_score(
                        target, probe["metrics"]["test"]),
                }), flush=True)
            details[source] = row
            scores[source] = {
                target: selection_score(target, row[target]["metrics"]["test"])
                for target in TASK_ORDER
            }
            save(args.out / "matrix.json", {
                "protocol": "frozen_source_d2nn_fit_target_single_linear_on_full_train",
                "task_order": list(TASK_ORDER),
                "optimization": "target Linear(784,C) only",
                "optical_updates": 0,
                "score_matrix": scores,
                "details": details,
            })
        save(args.out / "status.json", {"status": "complete"})
    except Exception as error:
        save(args.out / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
