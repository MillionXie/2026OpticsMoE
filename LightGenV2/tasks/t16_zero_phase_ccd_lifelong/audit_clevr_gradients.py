"""Diagnostic only: paired CLEVR score separation and optical gradient flow."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t14_shared_readout_lifelong.data import ClevrRawPairs
from .model import DirectCCDOptics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    if args.pairs < 1 or args.pairs > 32:
        raise ValueError("diagnostic pair count must be 1..32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["task"] != "clevr" or config["architecture"] not in ("moe", "d2nn"):
        raise ValueError("expected t16 CLEVR checkpoint")
    model = DirectCCDOptics(config["architecture"],
                            activation_order=config["activation_order"]).to(device)
    model.configure_stage(0)
    model.load_state_dict(checkpoint["model"])
    model.train()
    data = ClevrRawPairs(args.source, "train")
    indices = np.arange(2 * args.pairs)
    field = data.get_batch(indices, device)
    target = torch.as_tensor(data.labels[indices], device=device)
    output = model(field)
    logits = output["logits"]
    loss = F.cross_entropy(logits, target)
    loss.backward()
    binary_margin = (logits[:, 1] - logits[:, 0]).detach().reshape(-1, 2)
    pair_gap = binary_margin[:, 0] - binary_margin[:, 1]
    grad = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad[name] = float(parameter.grad.norm().detach().cpu())
    result = {
        "purpose": "diagnostic_train_subset_only_not_accuracy",
        "source_checkpoint": str(args.checkpoint),
        "selected_epoch": int(checkpoint["epoch"]),
        "sample_pairs": args.pairs,
        "loss": float(loss.detach().cpu()),
        "positive_minus_negative_margin_mean": float(pair_gap.mean().cpu()),
        "positive_minus_negative_margin_abs_mean": float(pair_gap.abs().mean().cpu()),
        "same_image_router_weight_l1_mean": (
            float((output["route_power"].detach().reshape(-1, 2, 16)[:, 0]
                   - output["route_power"].detach().reshape(-1, 2, 16)[:, 1]
                   ).abs().sum(1).mean().cpu())
            if output["route_power"] is not None else None),
        "gradient_l2_by_parameter": grad,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
