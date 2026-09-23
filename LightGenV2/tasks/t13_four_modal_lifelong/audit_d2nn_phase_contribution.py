"""Post-hoc D2NN validation ablation with each task's original Linear head.

No parameters are trained.  Each task uses its independent full-aperture
checkpoint, then the learned optical phases are replaced by constant or
deterministically initialized phases while its trained readout is fixed.
"""

import argparse
import json
from pathlib import Path

import torch

from .run import TASK_ORDER, build_model, evaluate, load_tasks, save, selection_score, state_sha


def parse_mapping(items):
    return dict(item.split("=", 1) for item in items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    checkpoints = parse_mapping(args.checkpoint)
    data_roots = parse_mapping(args.data)
    if set(checkpoints) != set(TASK_ORDER) or set(data_roots) != set(TASK_ORDER):
        raise ValueError("exactly four checkpoint and data entries are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {}
    for task_index, name in enumerate(TASK_ORDER):
        task = load_tasks({name: data_roots[name]}, require_full=True, names=(name,))[name]
        initial = build_model(
            "d2nn", cfg, cfg["seed"] + task_index,
            max_experts=int(cfg.get("single_task_d2nn_max_experts", 4)),
        )
        model = build_model(
            "d2nn", cfg, cfg["seed"] + task_index,
            max_experts=int(cfg.get("single_task_d2nn_max_experts", 4)),
        ).to(device)
        checkpoint = torch.load(checkpoints[name], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        learned_phases = [model.first_phase, model.global_phase, *model.additional_phases]
        initial_phases = [initial.first_phase, initial.global_phase, *initial.additional_phases]
        learned_copy = [phase.detach().clone() for phase in learned_phases]
        head_before = {
            key: state_sha(value) for key, value in model.named_parameters()
            if key.startswith(f"heads.{name}.")
        }
        phase_stats = []
        for learned, original in zip(learned_copy, initial_phases):
            left, right = learned.float().cpu(), original.detach().float()
            phase_stats.append({
                "shape": list(left.shape),
                "learned_raw_std": float(left.std()),
                "initial_raw_std": float(right.std()),
                "raw_change_rms": float((left - right).square().mean().sqrt()),
                "raw_change_fraction_gt_0.1": float(((left - right).abs() > 0.1).float().mean()),
            })
        conditions = {}
        for condition in ("trained", "flat", "initialized"):
            with torch.no_grad():
                for learned, saved, original in zip(learned_phases, learned_copy, initial_phases):
                    if condition == "trained":
                        learned.copy_(saved)
                    elif condition == "flat":
                        learned.zero_()
                    else:
                        learned.copy_(original.to(device))
            metrics, _, _ = evaluate(model, task, "val", device, cfg["eval_batch"])
            conditions[condition] = selection_score(name, metrics)
            print(json.dumps({"task": name, "condition": condition,
                              "validation_balanced_accuracy": conditions[condition]}), flush=True)
        head_after = {
            key: state_sha(value) for key, value in model.named_parameters()
            if key.startswith(f"heads.{name}.")
        }
        if head_before != head_after:
            raise RuntimeError(f"{name}: readout changed during phase ablation")
        output[name] = {
            "checkpoint": checkpoints[name],
            "checkpoint_epoch": checkpoint.get("epoch"),
            "validation_balanced_accuracy": conditions,
            "phase_stats": phase_stats,
            "head_preserved": True,
        }
        del model, initial, task
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "d2nn_phase_contribution.json", {
        "protocol": "full_validation_inference_only_original_task_linear_head",
        "optimization_steps": 0,
        "results": output,
    })


if __name__ == "__main__":
    main()
