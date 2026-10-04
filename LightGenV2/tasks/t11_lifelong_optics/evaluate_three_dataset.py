"""Reload a three-task checkpoint and reproduce validation metrics without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .continual_three_dataset import GROUP_MASKS, PREFIX_MASKS
from .cross_dataset import evaluate, load_dataset
from .data import sha
from .model import OpticalMoE
from .run import save


def select_by_identity(all_ids, selected_ids, label):
    lookup = {str(identity): i for i, identity in enumerate(all_ids.tolist())}
    try:
        return np.asarray([lookup[str(identity)] for identity in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"{label} identity is absent from the supplied dataset: {error}") from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    for name in "abc":
        parser.add_argument(f"--task-{name}", type=Path, required=True)
        parser.add_argument(f"--task-{name}-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    split = json.loads((args.run / "split.json").read_text())
    recorded = json.loads((args.run / "metrics.json").read_text())
    checkpoint_path = args.run / "C" / "best_checkpoint.pt"
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = OpticalMoE(cfg).to(args.device)
    model.load_state_dict(state["model"])

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "checkpoint_stage": state.get("stage"),
        "checkpoint_epoch": state.get("epoch"),
        "headline_metric": "balanced_accuracy",
        "test_images_read": False,
        "interpretation": "Expert masks alter coherent interference; masked results are diagnostics, not additive knowledge estimates.",
    }
    for j, name in enumerate("ABC"):
        data, _ = load_dataset(getattr(args, f"task_{name.lower()}"), getattr(args, f"task_{name.lower()}_manifest"))
        indices = select_by_identity(data["val_ids"], split[name]["val_ids"], f"task {name} validation")
        images = torch.from_numpy(data["val_images"][indices])
        labels = torch.from_numpy(data["val_labels"][indices]).long()
        variants = [("all", None), ("own_group", GROUP_MASKS[j]), ("learned_prefix", PREFIX_MASKS[j])]
        if j:
            variants.append(("previous_prefix", PREFIX_MASKS[j - 1]))
        for label, mask in variants:
            metrics, _, _ = evaluate(model, images, labels, cfg["batch_size"], mask)
            results[f"{name}_{label}"] = metrics

    results["A_BWT_after_C"] = results["A_all"]["balanced_accuracy"] - recorded["snapshots"]["A"]["metrics"]["A"]["balanced_accuracy"]
    results["B_BWT_after_C"] = results["B_all"]["balanced_accuracy"] - recorded["snapshots"]["B"]["metrics"]["B"]["balanced_accuracy"]
    for name in "ABC":
        expected = recorded[f"{name}_all"]
        actual = results[f"{name}_all"]
        if actual["confusion"] != expected["confusion"] or abs(actual["balanced_accuracy"] - expected["balanced_accuracy"]) > 1e-12:
            raise RuntimeError(f"Reloaded {name} metrics do not match the recorded run")
    save(args.run / "reevaluation_three.json", results)
    print(json.dumps({key: value["balanced_accuracy"] for key, value in results.items() if isinstance(value, dict) and "balanced_accuracy" in value}, indent=2))


if __name__ == "__main__":
    main()
