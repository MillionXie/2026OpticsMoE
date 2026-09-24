"""Read-only feasibility scan of moving the final CCD, without a new lens."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from LightGenV2.tasks.t14_shared_readout_lifelong.single_task_eurosat import source_paths
from LightGenV2.tasks.t13_four_modal_lifelong.model import AngularSpectrumPropagator
from .data import FilledEuroSatFields
from .model import DirectCCDOptics
from .probe_geometry import candidates, powers_from_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainval, holdout = source_paths(args.eurosat)
    data = FilledEuroSatFields(trainval, holdout, "train")
    indices = np.linspace(0, len(data)-1, args.samples, dtype=np.int64)
    models = {name: DirectCCDOptics(name).to(device).eval()
              for name in ("moe", "d2nn")}
    for model in models.values():
        model.configure_stage(0)
    h = models["moe"].height
    layouts = list(candidates(h))
    distances = (.1, .25, .5, .75)
    propagators = {d: AngularSpectrumPropagator(
        wavelength_m=5.32e-7, pixel_size_m=1.7e-5,
        grid_size=(h, h), distance_m=d).to(device) for d in distances}
    sums = {(d, k): [] for d in distances for k in range(len(layouts))}
    examples = {}
    border_fraction = {d: [] for d in distances}
    with torch.no_grad():
        for architecture, model in models.items():
            for sample_index in indices:
                field = data[np.array([sample_index])].to(device)
                output = model(field, return_debug=True)
                pre_final = output["pre_final_field"]
                for distance, propagator in propagators.items():
                    intensity = propagator(pre_final).abs().square()[0].cpu().numpy().astype(np.float64)
                    total = intensity.sum()
                    border = np.ones(intensity.shape, dtype=bool)
                    border[40:-40, 40:-40] = False
                    border_fraction[distance].append(float(intensity[border].sum() / total))
                    prefix = np.pad(intensity.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                    key = f"{architecture}/{distance}"
                    if key not in examples:
                        examples[key] = intensity
                    for layout_index, (side, xp, yp, centers) in enumerate(layouts):
                        powers = powers_from_prefix(prefix, centers, side).clip(min=0)
                        capture = float(powers.sum() / max(total, 1e-20))
                        p = powers / max(powers.sum(), 1e-20)
                        entropy = float(-(p * np.log(p + 1e-30)).sum() / np.log(10))
                        sums[(distance, layout_index)].append((architecture, capture,
                                                               entropy, float(p.min())))
            print(f"finished {architecture}", flush=True)
    ranking = {}
    for distance in distances:
        rows = []
        for index, (side, xp, yp, centers) in enumerate(layouts):
            stats = {architecture: np.asarray([
                (capture, entropy, minimum)
                for name, capture, entropy, minimum in sums[(distance, index)]
                if name == architecture])
                for architecture in models}
            mean = {name: {"capture": float(value[:, 0].mean()),
                           "entropy": float(value[:, 1].mean()),
                           "min_window_probability": float(value[:, 2].mean())}
                    for name, value in stats.items()}
            minimum_capture = min(row["capture"] for row in mean.values())
            minimum_entropy = min(row["entropy"] for row in mean.values())
            rows.append({"side": side, "x_pitch": xp, "y_pitch": yp,
                         "score": minimum_capture * (.5 + .5 * minimum_entropy),
                         "group_means": mean})
        ranking[str(distance)] = sorted(rows, key=lambda row: row["score"],
                                        reverse=True)[:10]
    report = {"label_free": True, "trained": False,
              "changed_component_in_probe": "final free-space distance only",
              "warning": "FFT propagation is periodic; large-distance edge energy may indicate aliasing",
              "source": str(args.eurosat), "sample_indices": indices.tolist(),
              "ranking": ranking,
              "mean_40px_border_energy_fraction": {
                  str(distance): float(np.mean(values))
                  for distance, values in border_fraction.items()}}
    (args.out / "scan.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, len(distances), figsize=(16, 8), constrained_layout=True)
    for row, architecture in enumerate(models):
        for column, distance in enumerate(distances):
            top = ranking[str(distance)][0]
            model = models[architecture]
            model.set_output_geometry(top["side"], top["x_pitch"], top["y_pitch"])
            ax = axes[row, column]
            ax.imshow(np.log1p(examples[f"{architecture}/{distance}"] * 1e6), cmap="inferno")
            for number, (y, x) in enumerate(model.output_centers):
                half = model.output_side // 2
                ax.add_patch(Rectangle((x-half, y-half), model.output_side,
                                       model.output_side, fill=False,
                                       edgecolor="cyan", linewidth=.7))
                ax.text(x, y, str(number), color="white", ha="center", va="center", fontsize=7)
            ax.set_title(f"{architecture}: final distance {distance} m")
            ax.axis("off")
    fig.savefig(args.out / "distance_comparison.png", dpi=130)
    plt.close(fig)
    print(json.dumps({"best_by_distance": {k: v[0] for k, v in ranking.items()},
                      "border_fraction": report["mean_40px_border_energy_fraction"]}),
          flush=True)


if __name__ == "__main__":
    main()
