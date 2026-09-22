"""Test whether the EuroSAT head depends on learned D2NN optical phases.

The EuroSAT single-task Linear head stays fixed in every condition.  The
optical phases are independently changed to zero or deterministic random
uniform phases, with no training or test-set model selection.
"""

import argparse
import json
from pathlib import Path

import torch

from .run import build_model, evaluate, load_tasks, save, selection_score, state_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = load_tasks({"eurosat": args.eurosat}, require_full=True, names=("eurosat",))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(
        "d2nn", cfg, cfg["seed"],
        max_experts=int(cfg.get("single_task_d2nn_max_experts", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    head_before = {
        name: state_sha(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("heads.eurosat.")
    }
    phase_parameters = [model.first_phase, model.global_phase, *model.additional_phases]
    results = {}
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    for condition in ("trained", "zero", "random_uniform"):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(original[name])
            if condition == "zero":
                for phase in phase_parameters:
                    phase.zero_()
            elif condition == "random_uniform":
                for phase in phase_parameters:
                    value = torch.empty(phase.shape, device="cpu").uniform_(
                        -torch.pi, torch.pi, generator=generator,
                    )
                    phase.copy_(value.to(device))
        metrics, _, _ = evaluate(model, tasks["eurosat"], "test", device, cfg["eval_batch"])
        results[condition] = {
            "balanced_accuracy": selection_score("eurosat", metrics),
            "accuracy": metrics["accuracy"],
            "phase_hashes": [state_sha(phase) for phase in phase_parameters],
        }
        print(json.dumps({"condition": condition, **results[condition]}), flush=True)

    head_after = {
        name: state_sha(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("heads.eurosat.")
    }
    if head_after != head_before:
        raise RuntimeError("EuroSAT head changed during the optical ablation")
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "eurosat_optics_ablation.json", {
        "checkpoint": str(args.checkpoint),
        "geometry": list(model.first_phase.shape),
        "optimization_steps": 0,
        "head_preserved": True,
        "results": results,
    })


if __name__ == "__main__":
    main()
