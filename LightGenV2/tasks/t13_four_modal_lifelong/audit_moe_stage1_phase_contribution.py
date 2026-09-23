"""EuroSAT MoE validation ablation with its trained Linear head fixed."""

import argparse
import json
from pathlib import Path

import torch

from .run import build_model, evaluate, load_tasks, save, selection_score, state_sha


def phases(model):
    return [model.router_phase, *model.first_phase[:4], model.global_phase,
            *model.additional_phases]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = load_tasks({"eurosat": args.eurosat}, require_full=True,
                      names=("eurosat",))["eurosat"]
    initial = build_model("moe", cfg, cfg["seed"])
    model = build_model("moe", cfg, cfg["seed"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.active_count.fill_(4)
    phase_tensors = phases(model)
    saved = [p.detach().clone() for p in phase_tensors]
    initial_phases = phases(initial)
    head_before = {k: state_sha(v) for k, v in model.named_parameters()
                   if k.startswith("heads.eurosat.")}
    stats = []
    for learned, original in zip(saved, initial_phases):
        left, right = learned.float().cpu(), original.detach().float()
        stats.append({
            "shape": list(left.shape),
            "learned_raw_std": float(left.std()),
            "initial_raw_std": float(right.std()),
            "raw_change_rms": float((left - right).square().mean().sqrt()),
        })
    scores = {}
    for condition in ("trained", "flat", "initialized"):
        with torch.no_grad():
            for parameter, learned, original in zip(phase_tensors, saved, initial_phases):
                if condition == "trained":
                    parameter.copy_(learned)
                elif condition == "flat":
                    parameter.zero_()
                else:
                    parameter.copy_(original.to(device))
        metrics, _, _ = evaluate(model, task, "val", device, cfg["eval_batch"], 0)
        scores[condition] = selection_score("eurosat", metrics)
        print(json.dumps({"condition": condition,
                          "validation_balanced_accuracy": scores[condition]}), flush=True)
    head_after = {k: state_sha(v) for k, v in model.named_parameters()
                  if k.startswith("heads.eurosat.")}
    if head_before != head_after:
        raise RuntimeError("EuroSAT readout changed during phase ablation")
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "moe_stage1_phase_contribution.json", {
        "checkpoint": str(args.checkpoint),
        "optimization_steps": 0,
        "head_preserved": True,
        "validation_balanced_accuracy": scores,
        "phase_stats": stats,
    })


if __name__ == "__main__":
    main()
