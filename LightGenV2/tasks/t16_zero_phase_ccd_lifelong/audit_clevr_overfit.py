"""Tiny training-only CLEVR memorization probe; never a benchmark score."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t14_shared_readout_lifelong.data import ClevrRawPairs
from .data import ClevrAttributeQueryPairs
from .model import DirectCCDOptics
from .train_other_tasks import clevr_pairwise_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not (1 <= args.pairs <= 32 and 1 <= args.steps <= 500):
        raise ValueError("diagnostic budget exceeded")
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["task"] not in {"clevr", "clevr_attributes"}:
        raise ValueError("expected t16 CLEVR checkpoint")
    model = DirectCCDOptics(config["architecture"],
                            activation_order=config["activation_order"]).to(device)
    model.configure_stage(0)
    model.load_state_dict(checkpoint["model"])
    model.train()
    dataset_type = (ClevrAttributeQueryPairs if config["task"] == "clevr_attributes"
                    else ClevrRawPairs)
    data = dataset_type(args.source, "train")
    indices = np.arange(2 * args.pairs)
    field = data.get_batch(indices, device)
    target = torch.as_tensor(data.labels[indices], device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    history = []
    checkpoints = {0, 1, 10, 25, 50, 100, args.steps}
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(field)
        logits = output["logits"]
        loss = F.cross_entropy(logits, target)
        if config.get("clevr_pairwise_weight", 0):
            loss = loss + float(config["clevr_pairwise_weight"]) * clevr_pairwise_loss(
                logits, target)
        if step in checkpoints:
            binary_margin = (logits[:, 1] - logits[:, 0]).detach().reshape(-1, 2)
            history.append({
                "step": step, "train_subset_loss": float(loss.detach().cpu()),
                "train_subset_correct_of_total": [
                    int((logits.argmax(1) == target).sum().detach().cpu()),
                    len(target)],
                "positive_minus_negative_margin_mean": float(
                    (binary_margin[:, 0] - binary_margin[:, 1]).mean().cpu()),
            })
        if step == args.steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                        if p.requires_grad], 1.0)
        optimizer.step()
    print(json.dumps({
        "purpose": "memorization_on_reused_training_pairs_only_not_generalization",
        "architecture": config["architecture"], "task": config["task"],
        "clevr_pairwise_weight": config.get("clevr_pairwise_weight", 0),
        "pairs": args.pairs,
        "steps": args.steps, "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
