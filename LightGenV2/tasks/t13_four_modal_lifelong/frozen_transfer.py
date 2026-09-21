"""Evaluate independently trained D2NN optics with only a refitted target MLP."""
import argparse
import json
from pathlib import Path

import torch

from .model import TASK_ORDER
from .run import build_model, fit_mlp_probe, load_tasks, save, seed_all, selection_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    for task in TASK_ORDER:
        p.add_argument("--" + task, type=Path, required=True)
        p.add_argument("--" + task + "-checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    device = torch.device(cfg.get("device", "cuda")); seed_all(cfg["seed"])
    paths = {name: getattr(a, name) for name in TASK_ORDER}
    tasks = load_tasks(paths, require_full=bool(cfg.get("require_full", False)))
    details, matrix = {}, {}
    for source_index, source in enumerate(TASK_ORDER):
        model = build_model("d2nn", cfg, cfg["seed"] + source_index,
                            max_experts=4).to(device)
        checkpoint_path = getattr(a, source + "_checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.requires_grad_(False); details[source] = {}; matrix[source] = {}
        for target_index, target in enumerate(TASK_ORDER):
            result = fit_mlp_probe(model, tasks[target], cfg, device,
                                   cfg["seed"] + source_index * 100 + target_index)
            details[source][target] = result
            matrix[source][target] = selection_score(target, result["metrics"]["test"])
            print(json.dumps({"source_optics": source, "target_mlp": target,
                              "test": matrix[source][target]}), flush=True)
    save(a.out / "results.json", {
        "task_order": list(TASK_ORDER),
        "contract": "independent source-task D2NN optics frozen; target MLP fitted on all target training samples; validation-selected; test accessed afterward",
        "test_matrix": matrix, "details": details,
    })


if __name__ == "__main__":
    main()
