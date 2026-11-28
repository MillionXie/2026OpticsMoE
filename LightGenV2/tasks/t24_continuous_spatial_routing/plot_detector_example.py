"""Illustrate measured CCD windows and expert-plane footprints of two fixed VAL IDs."""
import argparse
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.nn import functional as F

from .constrained_routing import FootprintRoutingLG
from .quantize_positions import load_validation, read_json
from .train import REPO, TASK, require_authorized_gpu, sha256


def verified(directory, names):
    manifest = read_json(directory / "manifest.json")
    for name in names:
        if sha256(directory / name) != manifest[name]:
            raise ValueError(f"Run manifest mismatch at {directory / name}")
    return {name: manifest[name] for name in names}


@torch.inference_mode()
def example(run, audit, device):
    noise_seeds = [1069, 1999]
    run_hash = verified(run, ["metadata.json", "selection_lock.json", "best_checkpoint.pt", "predictions_val.npz"])
    audit_hash = verified(audit, ["result.json", "canonical.npz", *[f"noise_{s}.npz" for s in noise_seeds]])
    result, meta = read_json(audit / "result.json"), read_json(run / "metadata.json")
    cfg = meta["config"]
    if (result["best_checkpoint_sha256"] != run_hash["best_checkpoint.pt"] or
            result["val_predictions_sha256"] != run_hash["predictions_val.npz"] or
            result["seed"] != 37 or result["mode"] != "constrained" or result["split"] != "val"):
        raise ValueError("Audit and trained run do not identify this VAL checkpoint")
    uuid = require_authorized_gpu(cfg["authorized_gpu_uuids"]) if device.type == "cuda" else None
    model = FootprintRoutingLG(cfg["num_classes"]).eval()
    for file in ("model.py", "constrained_routing.py"):
        if sha256(TASK / file) != meta["source_sha256"][(TASK / file).relative_to(REPO).as_posix()]:
            raise ValueError("Model changed since training: " + file)
    model.load_state_dict(torch.load(run / "best_checkpoint.pt", map_location="cpu", weights_only=False)["model"])
    model.to(device)
    fields, labels, ids = load_validation(cfg, device)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Expected unique validation IDs")
    idx = min(range(len(ids)), key=lambda i: hash_bytes(str(ids[i]).encode("utf-8")).digest())
    with np.load(run / "predictions_val.npz", allow_pickle=False) as data:
        if not np.array_equal(ids, data["ids"]) or not np.array_equal(labels.numpy(), data["labels"]):
            raise ValueError("Validation identity mismatch")
        expected = data["positions"][idx]
    with np.load(audit / "canonical.npz", allow_pickle=False) as data:
        if not np.array_equal(data["ids"], ids) or not np.allclose(data["positions"][idx], expected, atol=.002, rtol=0):
            raise ValueError("Canonical audit row mismatch")
    noisy = []
    for seed in noise_seeds:
        with np.load(audit / f"noise_{seed}.npz", allow_pickle=False) as data:
            if len(data["positions"]) != len(ids):
                raise ValueError("Incomplete audit capture")
            noisy.append(data["positions"][idx].tolist())
    # Replay the original validation batch shape for the same FFT implementation.
    batch = cfg["batch"]
    start = idx // batch * batch
    amplitude = fields[start:start + batch].to(device)
    amplitude = amplitude / amplitude.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)
    intensity = model.propagator(F.pad(amplitude * model.transmission(model.router_phase), (147,) * 4)).abs().square()
    scores = model.window_scores(intensity)
    candidates, ranking = model._candidate_positions(scores)
    anchors = model._select_pair(candidates, ranking)
    positions, power, _ = model.route_from_ccd(intensity)
    row = idx - start
    coords = positions[row].cpu().numpy()
    if not np.allclose(coords, expected, rtol=0, atol=.002):
        raise ValueError(f"Regenerated coordinates disagree for {ids[idx]}")
    values = scores[row].cpu().numpy()
    info = dict(dataset=cfg["dataset"], run=run.name, seed=37, split="val", id=str(ids[idx]),
                row=idx, id_sha256=hash_bytes(str(ids[idx]).encode()).hexdigest(),
                data_manifest_sha256=cfg["manifest_sha256"], run_files_sha256=run_hash, audit_files_sha256=audit_hash,
                trained_commit=meta["source_commit"], device=str(device), gpu_uuid=uuid,
                max_abs_replay_coordinate_error_px=float(np.abs(coords - expected).max()),
                clean_yx=coords.tolist(), gate_power=power[row].cpu().tolist(),
                anchors_ccd_center_yx=(model.anchor_yx[anchors[row]] + 132).cpu().tolist(),
                noisy_landing_yx=noisy, noise_seeds=noise_seeds,
                photons_per_unit_incident_energy=result["photons_per_unit_incident_energy"],
                raw_window_powers=values.tolist(), center_axis_px=(model.axis + 132).cpu().tolist())
    return info


def panels(fig, axes, row, title):
    ccd, expert = axes
    centers = np.array(row["center_axis_px"])
    mid = (centers[1:] + centers[:-1]) / 2
    edges = np.r_[2 * centers[0] - mid[0], mid, 2 * centers[-1] - mid[-1]]
    values = np.array(row["raw_window_powers"])
    mesh = ccd.pcolormesh(edges, edges, values, cmap="magma", shading="flat", rasterized=True)
    fig.colorbar(mesh, ax=ccd, fraction=.044, pad=.03, format="%.2g")
    ccd.set(xlim=(edges[0], edges[-1]), ylim=(edges[-1], edges[0]), aspect="equal",
            xticks=[132, 259, 386], yticks=[132, 259, 386], ylabel="CCD y (px)", xlabel="CCD x (px)")
    colors = ("#00768a", "#b25317")
    points = np.array(row["clean_yx"])
    for i, (y, x) in enumerate(points):
        expert.add_patch(Rectangle((x, y), 224, 224, facecolor=colors[i], alpha=.28))
        expert.add_patch(Rectangle((x, y), 224, 224, fill=False, edgecolor=colors[i], lw=1.6))
        expert.text(x+6, y+20, f"{i+1}", color=colors[i], fontsize=10, weight="bold")
    anchors = np.asarray(row["anchors_ccd_center_yx"])
    ccd.scatter(anchors[:, 1], anchors[:, 0], marker="s", s=130, facecolors="none", edgecolors="w", lw=1)
    ccd.scatter(points[:, 1]+132, points[:, 0]+132, marker="o", s=60, facecolors="none", edgecolors="#5eead4", lw=2)
    for noisy, marker, color in zip(row["noisy_landing_yx"], ("+", "x"), ("#e0ff38", "#88bfff")):
        p = np.asarray(noisy) + 132
        ccd.scatter(p[:, 1], p[:, 0], marker=marker, color=color, s=42, lw=1.3)
    for i, (y, x) in enumerate(points+132):
        ccd.annotate(str(i+1), (x, y), xytext=(7,-8), textcoords="offset points", fontsize=9,
                     color="white", bbox=dict(fc="#252630", ec="none", pad=1))
    overlap = np.maximum(224 - np.abs(points[0]-points[1]), 0).prod()/224**2
    expert.set(xlim=(0,478), ylim=(478,0), aspect="equal", xticks=[0, 224, 478], yticks=[0, 224, 478],
               xlabel="Expert x (px)", ylabel="Expert y (px)")
    ccd.set_title(f"{title}   ·   ID {row['id']}", fontsize=11)
    expert.set_title(f"Clean footprints   ·   {overlap:.1%} overlap", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory below T24")
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(TASK):
        parser.error("Use an output directory inside the task")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.6))
    fig.subplots_adjust(left=.10, right=.96, bottom=.16, top=.89, hspace=.42, wspace=.34)
    fig.suptitle("Router CCD window power and input placement", fontsize=16, y=.973)
    fig.text(.5,.936,"One validation image per dataset   ·   constrained routing   ·   seed 37", ha="center")
    examples = []
    for pair_axes, key, title in zip(axes, ("kather", "blood"), ("Kather2016", "BloodMNIST128")):
        run_name = f"{key}_s37_constrained"
        row = example(args.runs / run_name, args.audits / run_name, torch.device(args.device))
        panels(fig, pair_axes, row, title)
        examples.append(row)
    marks = [("s", "Detector anchor", "#494e59"), ("o", "Clean centroid", "#098476"),
             ("+", "Poisson A", "#8a8900"), ("x", "Poisson B", "#4675af")]
    legend = [Line2D([], [], ls="none", marker=m, color=c, markerfacecolor="none", label=l) for m,l,c in marks]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5,.066), ncol=4, frameon=False)
    fig.text(.5,.029,"60 × 60 px detector integrals on a 17 × 17 grid. Each clean footprint is 224 × 224 px.\n"
             "CCD centroid (y, x) maps to expert top-left (y − 132, x − 132). Pixel pitch: 17 µm.",
             ha="center", fontsize=9)
    out = args.out / "detector_example.svg"
    plt.rcParams["svg.hashsalt"] = "t24-detector-example"
    fig.savefig(out, dpi=120, metadata={"Date":None})
    plt.close(fig)
    (args.out / "detector_example.json").write_text(json.dumps(dict(
        selection="lowest SHA256 of UTF-8 validation ID within each dataset",
        script_sha256=sha256(Path(__file__)), svg_sha256=sha256(out), examples=examples), indent=2)+"\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
