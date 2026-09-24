"""Label-free zero-phase input and direct-CCD geometry audit (no training)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from LightGenV2.tasks.t14_shared_readout_lifelong.data import (
    ClevrRawPairs, PhysicalPermutedCandidates,
)
from LightGenV2.tasks.t14_shared_readout_lifelong.single_task_eurosat import source_paths
from .data import PairedEuroSatFields, SpeechBinaryPairs
from .model import DirectCCDOptics


def get_input(dataset, index, device):
    if hasattr(dataset, "get_batch"):
        return dataset.get_batch(np.array([index]), device)
    return dataset[np.array([index])].to(device)


def candidates(size):
    c = size // 2
    for side in (64, 96, 128, 160, 192):
        for xp in (128, 160, 192, 224, 256, 288):
            for yp in (160, 200, 240, 280, 320, 360):
                if xp < side + 16 or yp < side + 16:
                    continue
                centers = ([(c-yp, c + k*xp) for k in (-1, 0, 1)]
                           + [(c, c + int(k*xp/2)) for k in (-3, -1, 1, 3)]
                           + [(c+yp, c + k*xp) for k in (-1, 0, 1)])
                half = side // 2
                if any(y-half < 0 or y+half > size or
                       x-half < 0 or x+half > size for y, x in centers):
                    continue
                yield side, xp, yp, centers


def powers_from_prefix(prefix, centers, side):
    half = side // 2
    return np.asarray([
        prefix[y+half, x+half] - prefix[y-half, x+half]
        - prefix[y+half, x-half] + prefix[y-half, x-half]
        for y, x in centers], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    trainval, holdout = source_paths(args.eurosat)
    physical = json.loads(args.physical.read_text())
    datasets = {
        "eurosat": PairedEuroSatFields(trainval, holdout, "train"),
        "clevr": ClevrRawPairs(args.clevr, "train"),
        "speech_binary": SpeechBinaryPairs(args.speech, "train", seed=17),
        "physical": PhysicalPermutedCandidates(physical["source_roots"], "train", seed=17),
    }
    models = {name: DirectCCDOptics(name).to(device).eval()
              for name in ("moe", "d2nn")}
    for model in models.values():
        model.configure_stage(0)
        phase_params = [model.global_phase, *model.additional_phases]
        phase_params += ([model.router_phase, *model.first_phase]
                         if model.architecture == "moe" else [model.first_phase])
        if any(bool(torch.count_nonzero(p)) for p in phase_params):
            raise AssertionError("phase initialization is not exactly zero")
        if any(isinstance(layer, torch.nn.Linear) for layer in model.modules()):
            raise AssertionError("electronic classification layer found")

    layouts = list(candidates(models["moe"].height))
    if not layouts:
        raise RuntimeError("no non-overlapping layout candidates")
    rows = []
    first_examples = {}
    router_stats = {}
    for task, dataset in datasets.items():
        sample_indices = np.linspace(0, len(dataset)-1,
                                     args.samples_per_task, dtype=np.int64)
        for architecture, model in models.items():
            key = f"{task}/{architecture}"
            router_stats[key] = []
            for index in sample_indices:
                field = get_input(dataset, int(index), device)
                with torch.no_grad():
                    output = model(field, return_debug=True)
                intensity = output["final_ccd"][0].cpu().numpy().astype(np.float64)
                total = intensity.sum()
                prefix = np.pad(intensity.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                if key not in first_examples:
                    first_examples[key] = {
                        "input": field[0].cpu().numpy(), "intensity": intensity,
                        "router": (None if output["router_ccd"] is None
                                   else output["router_ccd"][0].cpu().numpy())}
                if output["router_efficiency"] is not None:
                    router_stats[key].append({
                        "sample_index": int(index),
                        "active_capture": float(output["router_efficiency"][0]),
                        "power_weights": output["route_power"][0, :4].cpu().tolist()})
                for side, xp, yp, centers in layouts:
                    powers = powers_from_prefix(prefix, centers, side).clip(min=0)
                    capture = float(powers.sum() / max(total, 1e-20))
                    probabilities = powers / max(powers.sum(), 1e-20)
                    entropy = float(-(probabilities * np.log(probabilities + 1e-30)).sum()
                                    / np.log(10))
                    rows.append((side, xp, yp, key, capture, entropy,
                                 float(probabilities.min())))
            print(f"finished {key}", flush=True)

    grouped = {}
    for side, xp, yp, key, capture, entropy, minimum in rows:
        entry = grouped.setdefault((side, xp, yp), {})
        entry.setdefault(key, []).append((capture, entropy, minimum))
    ranking = []
    for (side, xp, yp), measurements in grouped.items():
        group_means = {key: {
            "capture": float(np.mean(np.asarray(values)[:, 0])),
            "entropy": float(np.mean(np.asarray(values)[:, 1])),
            "min_window_probability": float(np.mean(np.asarray(values)[:, 2]))}
            for key, values in measurements.items()}
        min_capture = min(x["capture"] for x in group_means.values())
        min_entropy = min(x["entropy"] for x in group_means.values())
        ranking.append({"side": side, "x_pitch": xp, "y_pitch": yp,
                        "score": min_capture * (.5 + .5 * min_entropy),
                        "worst_group_capture": min_capture,
                        "worst_group_entropy": min_entropy,
                        "group_means": group_means})
    ranking.sort(key=lambda row: row["score"], reverse=True)
    best = ranking[0]
    report = {"trained": False, "labels_used": False, "split": "train",
              "samples_per_task": args.samples_per_task,
              "phase_raw_initialization": 0.0,
              "final_camera": "direct existing propagation plane; no added FFT",
              "selected_for_visual_inspection_only": best,
              "top_candidates": ranking[:20], "router": router_stats,
              "source_protocols": {"eurosat": str(args.eurosat),
                                   "clevr": str(args.clevr),
                                   "speech": str(args.speech),
                                   "physical": str(args.physical)}}
    (args.out / "scan.json").write_text(json.dumps(report, indent=2) + "\n")
    for architecture, model in models.items():
        model.set_output_geometry(best["side"], best["x_pitch"], best["y_pitch"])
        fig, axes = plt.subplots(4, 3, figsize=(13, 15), constrained_layout=True)
        for row, task in enumerate(datasets):
            example = first_examples[f"{task}/{architecture}"]
            axes[row, 0].imshow(example["input"], cmap="magma")
            axes[row, 0].set_title(f"{task}: fixed 224x224 amplitude")
            router = example["router"]
            if router is None:
                axes[row, 1].axis("off")
            else:
                axes[row, 1].imshow(np.log1p(router * 1e6), cmap="inferno")
                for number, (y, x) in enumerate(model.router_centers):
                    h = model.router_side // 2
                    axes[row, 1].add_patch(Rectangle(
                        (x-h, y-h), model.router_side, model.router_side,
                        fill=False, edgecolor="cyan", linewidth=.5))
                    axes[row, 1].text(x, y, str(number+1), color="white",
                                      ha="center", va="center", fontsize=5)
                axes[row, 1].set_title("zero-phase router CCD / 16 ports")
            axes[row, 2].imshow(np.log1p(example["intensity"] * 1e6), cmap="inferno")
            for number, (y, x) in enumerate(model.output_centers):
                h = model.output_side // 2
                axes[row, 2].add_patch(Rectangle((x-h, y-h),
                                                 model.output_side, model.output_side,
                                                 fill=False, edgecolor="cyan", linewidth=.9))
                axes[row, 2].text(x, y, str(number), color="white", fontsize=8,
                                  ha="center", va="center")
            axes[row, 2].set_title("direct CCD / tentative 3-4-3 windows")
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"{architecture}, raw phase=0, no extra lens; unlabeled train examples")
        fig.savefig(args.out / f"{architecture}_zero_phase.png", dpi=130)
        plt.close(fig)
    euro = datasets["eurosat"]
    index = np.array([0, len(euro)//3])
    original = _rgb_field(euro.images[index]).numpy()
    filled = euro[index].numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    for row in range(2):
        axes[row, 0].imshow(original[row], cmap="magma")
        axes[row, 0].set_title(f"old R/G/B/zero, sample {index[row]}")
        axes[row, 1].imshow(filled[row], cmap="magma")
        axes[row, 1].set_title("proposed R/G/B/luma, same source")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(args.out / "eurosat_input_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"best": best, "report": str(args.out / "scan.json")}), flush=True)


if __name__ == "__main__":
    main()
