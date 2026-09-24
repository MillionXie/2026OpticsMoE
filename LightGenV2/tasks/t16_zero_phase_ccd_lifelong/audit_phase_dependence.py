"""Validation-only check of whether learned optical phases affect predictions."""

import argparse
import json
from pathlib import Path

import torch

from .model import DirectCCDOptics
from .train_eurosat import load_split, source_paths
from .train_other_tasks import CLASSES, evaluate, load_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("batch must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    task = config["task"]
    protocol = Path(config["source_protocol"])
    if task == "eurosat_paired_rgb_sar":
        trainval, holdout = source_paths(protocol)
        data = load_split(trainval, holdout, "val")
        classes = 10
    else:
        data = load_task(protocol, task, "val")
        classes = CLASSES[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectCCDOptics(config["architecture"],
                            activation_order=config["activation_order"]).to(device)
    model.configure_stage(0)
    model.load_state_dict(checkpoint["model"])
    trained = evaluate(model, data, device, args.batch, classes)
    with torch.no_grad():
        model.global_phase.zero_()
        for phase in model.additional_phases:
            phase.zero_()
        if model.architecture == "moe":
            model.router_phase.zero_()
            for phase in model.first_phase:
                phase.zero_()
        else:
            model.first_phase.zero_()
    zero_phase = evaluate(model, data, device, args.batch, classes)
    result = {
        "purpose": "validation_only_phase_dependence_fixed_trained_linear",
        "task": task, "architecture": config["architecture"],
        "checkpoint": str(args.checkpoint), "source_commit": config["model_git_commit"],
        "selected_epoch": checkpoint["epoch"],
        "trained_phase": trained, "all_phase_raw_zero": zero_phase,
        "balanced_accuracy_drop": trained["balanced_accuracy"]
                                  - zero_phase["balanced_accuracy"],
        "limitation": "Zeroing all phases changes the system outside its training distribution; this measures dependence, not attribution of accuracy to optics.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"task": task, "architecture": config["architecture"],
                      "trained_validation_balanced_accuracy": trained["balanced_accuracy"],
                      "zero_phase_validation_balanced_accuracy": zero_phase["balanced_accuracy"],
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
