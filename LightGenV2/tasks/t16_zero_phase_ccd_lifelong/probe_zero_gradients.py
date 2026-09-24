"""Synthetic-target gradient audit for dark CCD windows; no model update."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from LightGenV2.tasks.t14_shared_readout_lifelong.single_task_eurosat import source_paths
from .data import FilledEuroSatFields
from .model import DirectCCDOptics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trainval, holdout = source_paths(args.eurosat)
    data = FilledEuroSatFields(trainval, holdout, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = data[np.array([0])].to(device)
    report = {"trained": False, "target_labels": "synthetic, not source labels",
              "source": str(args.eurosat), "models": {}}
    for architecture in ("moe", "d2nn"):
        model = DirectCCDOptics(architecture).to(device)
        model.configure_stage(0)
        model.eval()
        result = model(field)
        powers = result["window_power"][0]
        report["models"][architecture] = {
            "absolute_powers": powers.detach().cpu().tolist(),
            "normalized_powers": (powers / powers.sum()).detach().cpu().tolist(),
            "current_additive_logit_floor": 1e-20, "gradients": {}}
        for window in (0, 1, 4, 8):
            current_grad = torch.autograd.grad(
                result["logits"][0, window], model.global_phase,
                retain_graph=True, allow_unused=True)[0]
            former_hard_floor_grad = torch.autograd.grad(
                powers[window].clamp_min(1e-12).log(), model.global_phase,
                retain_graph=True, allow_unused=True)[0]
            report["models"][architecture]["gradients"][str(window)] = {
                "current_global_phase_grad_norm": float(current_grad.norm()) if current_grad is not None else None,
                "former_hard_floor_global_phase_grad_norm": float(former_hard_floor_grad.norm()) if former_hard_floor_grad is not None else None}
        print(f"finished {architecture}", flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
