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

from LightGenV2.tasks.t13_four_modal_lifelong.data import _rgb_field
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
    # Only compact, centered 3-4-3 readouts are in the current protocol.
    # The prior corner-scattered layouts belong to a historical probe.
    for side in (80, 96, 112):
        for xp in (side + 16, side + 32, side + 48):
            for yp in (side + 32, side + 64, side + 96):
                if xp < side + 16 or yp < side + 16:
                    continue
                centers = ([(c-yp, c + k*xp) for k in (-1, 0, 1)]
                           + [(c, c + int(k*xp/2)) for k in (-3, -1, 1, 3)]
                           + [(c+yp, c + k*xp) for k in (-1, 0, 1)])
                half = side // 2
                if any(y-half < 0 or y+half > size or
                       x-half < 0 or x+half > size for y, x in centers):
                    continue
                yield "compact_center_343", side, xp, yp, centers


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
    parser.add_argument("--activation-order", choices=("quadrant", "center_out"),
                        default="center_out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    trainval, holdout = source_paths(args.eurosat)
    physical = json.loads(args.physical.read_text())
    datasets = {
        "eurosat": PairedEuroSatFields(trainval, holdout, "train"),
        "clevr": ClevrRawPairs(args.clevr, "train"),
        "speech_binary": SpeechBinaryPairs(args.speech, "train"),
        "physical": PhysicalPermutedCandidates(physical["source_roots"], "train", seed=17),
    }
    models = {name: DirectCCDOptics(name, activation_order=args.activation_order).to(device).eval()
              for name in ("moe", "d2nn")}
    for model in models.values():
        model.configure_stage(0)
        phase_params = [model.global_phase, *model.additional_phases]
        phase_params += ([model.router_phase, *model.first_phase]
                         if model.architecture == "moe" else [model.first_phase])
        if any(bool(torch.count_nonzero(p)) for p in phase_params):
            raise AssertionError("phase initialization is not exactly zero")
        if [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)] != [model.shared_head]:
            raise AssertionError("expected exactly one shared Linear readout")

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
                        "active_experts": (model.active_indices[:4] + 1).cpu().tolist(),
                        "power_weights": output["route_power"][0, model.active_indices[:4]].cpu().tolist()})
                for layout, side, xp, yp, centers in layouts:
                    powers = powers_from_prefix(prefix, centers, side).clip(min=0)
                    capture = float(powers.sum() / max(total, 1e-20))
                    probabilities = powers / max(powers.sum(), 1e-20)
                    entropy = float(-(probabilities * np.log(probabilities + 1e-30)).sum()
                                    / np.log(10))
                    rows.append((layout, side, xp, yp, key, capture, entropy,
                                 float(probabilities.min())))
            print(f"finished {key}", flush=True)

    grouped = {}
    for layout, side, xp, yp, key, capture, entropy, minimum in rows:
        entry = grouped.setdefault((layout, side, xp, yp), {})
        entry.setdefault(key, []).append((capture, entropy, minimum))
    ranking = []
    for (layout, side, xp, yp), measurements in grouped.items():
        group_means = {key: {
            "capture": float(np.mean(np.asarray(values)[:, 0])),
            "entropy": float(np.mean(np.asarray(values)[:, 1])),
            "min_window_probability": float(np.mean(np.asarray(values)[:, 2]))}
            for key, values in measurements.items()}
        min_capture = min(x["capture"] for x in group_means.values())
        min_entropy = min(x["entropy"] for x in group_means.values())
        ranking.append({"layout": layout, "side": side,
                        "parameter_a": xp, "parameter_b": yp,
                        "score": min_capture * (.5 + .5 * min_entropy),
                        "worst_group_capture": min_capture,
                        "worst_group_entropy": min_entropy,
                        "group_means": group_means})
    ranking.sort(key=lambda row: row["score"], reverse=True)
    best = ranking[0]
    by_layout = {"compact_center_343": ranking[0]}
    report = {"trained": False, "labels_used": False, "split": "train",
              "samples_per_task": args.samples_per_task,
              "activation_order": args.activation_order,
              "readout": "one independent trainable Linear(784,10) per architecture; untrained here",
              "phase_raw_initialization": 0.0,
              "final_camera": "direct existing propagation plane; no added FFT",
              "selected_for_visual_inspection_only": best,
              "best_by_layout": by_layout,
              "top_candidates": ranking[:20], "router": router_stats,
              "source_protocols": {"eurosat": str(args.eurosat),
                                   "clevr": str(args.clevr),
                                   "speech": str(args.speech),
                                   "physical": str(args.physical)}}
    (args.out / "scan.json").write_text(json.dumps(report, indent=2) + "\n")
    best_key = (best["layout"], best["side"], best["parameter_a"], best["parameter_b"])
    best_centers = next(centers for layout, side, a, b, centers in layouts
                        if (layout, side, a, b) == best_key)
    for architecture, model in models.items():
        model.set_output_windows(best_centers, best["side"])
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
            axes[row, 2].set_title(f"direct CCD / tentative {best['layout']} windows")
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"{architecture}, raw phase=0, no extra lens; unlabeled train examples")
        fig.savefig(args.out / f"{architecture}_zero_phase.png", dpi=130)
        plt.close(fig)
    euro = datasets["eurosat"]
    index = np.array([0, len(euro)//3])
    original = _rgb_field(euro.rgb[index]).numpy()
    paired = euro[index].numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    for row in range(2):
        axes[row, 0].imshow(original[row], cmap="magma")
        axes[row, 0].set_title(f"historical RGB-only R/G/B/zero, pair {index[row]}")
        axes[row, 1].imshow(paired[row], cmap="magma")
        axes[row, 1].set_title("paired R/G/B/SAR, same location")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(args.out / "eurosat_input_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"best": best, "report": str(args.out / "scan.json")}), flush=True)


if __name__ == "__main__":
    main()
