"""Save a few validation examples of a trained MoE router and its CCD fields."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from .model import DirectCCDOptics
from .train_eurosat import load_split, source_paths
from .train_other_tasks import load_task


def sample_description(data, task, index, protocol):
    if task == "clevr":
        vocabulary = json.loads((protocol.parent / "vocab.json").read_text())
        inverse = {int(value): word for word, value in vocabulary.items()}
        return " ".join(inverse[int(token)] for token in data.token_ids[index] if token)
    if task == "speech_binary":
        words = data.base.rows[0]["candidate_words"]
        spoken = words[data.base.rows[index // 2]["audio_class"]]
        candidate = words[int(data.candidate_words[index])]
        return f"spoken={spoken}; text={candidate}"
    if task == "physical_binary":
        candidate = int(data.candidate_descriptions[index])
        concept = data.base.CONCEPTS[candidate // 2].replace("_", " ")
        kind = "possible" if candidate % 2 else "impossible"
        return f"text={concept} {kind}"
    return "paired RGB/SAR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["architecture"] != "moe":
        raise ValueError("router visualization requires a MoE checkpoint")
    task = config["task"]
    protocol_path = Path(config["source_protocol"])
    if task == "eurosat_paired_rgb_sar":
        trainval, holdout = source_paths(protocol_path)
        data = load_split(trainval, holdout, "val")
    else:
        data = load_task(protocol_path, task, "val")
    if any(index < 0 or index >= len(data) for index in args.indices):
        raise ValueError("a requested validation index is out of range")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectCCDOptics("moe", activation_order=config["activation_order"]).to(device)
    model.configure_stage(0)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    fig, axes = plt.subplots(len(args.indices), 4, figsize=(15, 3.5 * len(args.indices)),
                             squeeze=False, constrained_layout=True)
    active = model.active_indices[:int(model.active_count)].cpu().numpy()
    for row, sample_index in enumerate(args.indices):
        if hasattr(data, "get_batch"):
            field = data.get_batch(np.asarray([sample_index]), device)
        else:
            field = data[np.asarray([sample_index])].to(device)
        with torch.no_grad():
            output = model(field, return_debug=True)
        route = output["route_power"][0].cpu().numpy()
        router = output["router_ccd"][0].cpu().numpy()
        final = output["final_ccd"][0].cpu().numpy()
        predicted = int(output["logits"].argmax(1)[0])
        true_label = int(data.labels[sample_index])
        description = sample_description(data, task, sample_index, protocol_path)
        rows.append({"index": sample_index, "true_label": true_label,
                     "predicted_label": predicted,
                     "sample_description": description,
                     "active_slot_ids": (active + 1).tolist(),
                     "active_route_weights": route[active].tolist(),
                     "active_capture": float(output["router_efficiency"][0])})
        raw_input = field[0].cpu().numpy()
        if raw_input.min() < 0:
            # Signed temporal differences are real optical *fields*, not
            # nonnegative amplitudes. Preserve the sign with a diverging map.
            scale = max(float(np.quantile(np.abs(raw_input), 0.95)), 1e-20)
            shown = np.arcsinh(raw_input / scale)
            axes[row, 0].imshow(shown, cmap="coolwarm",
                                vmin=-max(abs(shown.min()), abs(shown.max())),
                                vmax=max(abs(shown.min()), abs(shown.max())))
            input_scale_note = "signed field shown on asinh scale"
        else:
            axes[row, 0].imshow(np.log1p(raw_input / max(raw_input.max(), 1e-20) * 100),
                                cmap="magma")
            input_scale_note = "nonnegative input shown on log scale"
        axes[row, 0].set_title(
            f"{task} val #{sample_index}; y={true_label}\n{description}\n{input_scale_note}",
            fontsize=8)
        axes[row, 1].imshow(np.log1p(router / max(router.max(), 1e-20) * 100),
                            cmap="inferno")
        for number, (y, x) in enumerate(model.router_centers):
            half = model.router_side // 2
            axes[row, 1].add_patch(Rectangle((x-half, y-half), model.router_side,
                              model.router_side, fill=False,
                              edgecolor="yellow" if number in active else "cyan",
                              linewidth=1 if number in active else 0.4))
            axes[row, 1].text(x, y, str(number + 1), color="white", ha="center",
                              va="center", fontsize=6)
        axes[row, 1].set_title(f"router CCD; active capture {rows[-1]['active_capture']:.1%}")
        axes[row, 2].bar((active + 1).astype(str), route[active])
        axes[row, 2].set_ylim(0, 1)
        axes[row, 2].set_title("expert optical-power weights")
        axes[row, 2].set_ylabel("fraction")
        axes[row, 3].imshow(np.log1p(final / max(final.max(), 1e-20) * 100),
                            cmap="inferno")
        axes[row, 3].set_title(f"final CCD; predicted {predicted}")
        for col in (0, 1, 3):
            axes[row, col].axis("off")
    args.out.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.out / f"{task}_trained_router.png", dpi=140)
    plt.close(fig)
    (args.out / "samples.json").write_text(json.dumps({
        "task": task, "split": "val", "checkpoint": str(args.checkpoint),
        "source_commit": config["model_git_commit"], "selected_epoch": checkpoint["epoch"],
        "readout": config["readout"], "samples": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
