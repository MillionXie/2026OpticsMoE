"""Plot selected MoE validation routing without changing checkpoints or scoring."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import DirectCCDOptics
from .train_lifelong_moe import load_dataset


TASKS = ("eurosat", "clevr", "speech_binary", "physical_binary_raw")


def plot_aggregate(validation, out):
    mean = np.asarray([validation[t]["router"]["mean_power_by_slot"] for t in TASKS])
    argmax = np.asarray([validation[t]["router"]["argmax_fraction_by_slot"] for t in TASKS])
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), constrained_layout=True)
    for ax, values, title in zip(axes, (mean, argmax),
                                 ("Mean optical power fraction", "Fraction of samples with largest weight")):
        image = ax.imshow(values, vmin=0, vmax=max(0.2, values.max()),
                          aspect="auto", cmap="viridis")
        ax.set_yticks(range(4), ("EuroSAT", "CLEVR", "Speech", "Physical"))
        ax.set_xticks(range(16), range(1, 17))
        ax.set_ylabel(title)
        fig.colorbar(image, ax=ax, fraction=0.02)
    axes[-1].set_xlabel("Physical expert slot (1–16)")
    fig.savefig(out / "expert_distribution_validation.png", dpi=160)
    plt.close(fig)


def plot_examples(model, datasets, device, out, indices=(0, 1)):
    fig, axes = plt.subplots(4, len(indices), figsize=(4 * len(indices), 12),
                             constrained_layout=True, squeeze=False)
    records = []
    for row, task in enumerate(TASKS):
        data = datasets[task]
        for col, index in enumerate(indices):
            field = data.get_batch(np.asarray([index]), device)
            with torch.no_grad():
                output = model(field, return_debug=True)
            q = output["route_power"][0].cpu().numpy()
            axes[row, col].bar(range(1, 17), q)
            axes[row, col].set_xticks(range(1, 17, 2))
            axes[row, col].set_ylim(0, 1)
            axes[row, col].set_title(f"{task} val #{index}: true {int(data.labels[index])}, "
                                     f"pred {int(output['logits'].argmax(1)[0])}")
            records.append({"task": task, "index": index, "true": int(data.labels[index]),
                            "predicted": int(output["logits"].argmax(1)[0]),
                            "power_by_slot": q.tolist(),
                            "active_capture": float(output["router_efficiency"][0])})
    fig.savefig(out / "expert_examples_validation.png", dpi=160)
    plt.close(fig)
    (out / "expert_examples_validation.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["architecture"] != "moe" or config["stage"] != 4:
        raise ValueError("expected a selected stage-D MoE checkpoint")
    if tuple(config["order"]) != TASKS:
        raise ValueError("task order differs from the plotted protocol")
    args.out.mkdir(parents=True, exist_ok=False)
    validation = checkpoint["validation"]
    plot_aggregate(validation, args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectCCDOptics("moe", activation_order="center_out").to(device)
    model.configure_stage(3)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    protocols = dict(zip(TASKS, (args.eurosat, args.clevr, args.speech, args.physical)))
    datasets = {task: load_dataset(protocols, task, "val", args.vision_checkpoint, device)
                for task in TASKS}
    plot_examples(model, datasets, device, args.out)
    (args.out / "source.json").write_text(json.dumps({"checkpoint": str(args.checkpoint),
        "model_git_commit": config["model_git_commit"], "split": "val",
        "selected_epoch": checkpoint["epoch"], "aggregate_n": {
            task: validation[task]["n"] for task in TASKS}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
