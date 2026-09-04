"""Analyze one physical phase plane across five-epoch snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .phase_snapshots import load_phase_snapshot


def _select(value: torch.Tensor, plane: str, frame: int, expert: int) -> torch.Tensor:
    if value.ndim == 2:
        if frame or expert:
            raise ValueError("A global/router 2-D plane has no frame/expert index")
        return value
    if plane == "parallel_optics.raw_expert_phase":
        index = frame * 4 + expert
    elif plane == "parallel_router.raw_router_phase":
        if expert:
            raise ValueError("Parallel router has one phase per frame, not per expert")
        index = frame
    elif plane == "serial_optics.raw_expert_phase":
        if frame:
            raise ValueError("Serial experts are sequence-level, not frame-level")
        index = expert
    else:
        raise ValueError(f"Unsupported indexed plane: {plane}")
    if not 0 <= index < value.shape[0]:
        raise IndexError(f"Selected index {index} exceeds shape {tuple(value.shape)}")
    return value[index]


def analyze(snapshot_dir: Path, output_dir: Path, plane: str, frame: int, expert: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pt under {snapshot_dir}")
    payloads = [load_phase_snapshot(path) for path in paths]
    phases = torch.stack(
        [_select(payload["planes"][plane]["phase_rad"].float(), plane, frame, expert) for payload in payloads]
    ).numpy()
    epochs = np.asarray([int(payload["epoch"]) for payload in payloads])
    if any(payload["architecture"] != payloads[0]["architecture"] for payload in payloads):
        raise RuntimeError("Snapshot directory mixes architectures")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "phase_rad_matrix.npy", phases.reshape(len(phases), -1))
    circular = np.concatenate((np.cos(phases), np.sin(phases)), axis=1).reshape(len(phases), -1)
    centered = circular - circular.mean(0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u * singular[None, :]
    np.save(output_dir / "circular_pca_scores.npy", scores)
    np.save(output_dir / "circular_pca_components.npy", vt)
    variance = singular**2
    variance_ratio = variance / max(float(variance.sum()), 1e-12)
    delta = np.angle(np.exp(1j * (phases - phases[:1])))
    rows = []
    for index, epoch in enumerate(epochs):
        rows.append(
            {
                "epoch": int(epoch),
                "phase_mean_rad": float(phases[index].mean()),
                "phase_std_rad": float(phases[index].std()),
                "wrapped_delta_rms_rad": float(np.sqrt(np.mean(delta[index] ** 2))),
                "fraction_changed_over_0p05_rad": float(np.mean(np.abs(delta[index]) > 0.05)),
                "circular_pc1": float(scores[index, 0]),
                "circular_pc2": float(scores[index, 1]) if scores.shape[1] > 1 else 0.0,
            }
        )
    with (output_dir / "one_expert_evolution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})
    columns = min(5, len(phases)); selected = np.linspace(0, len(phases) - 1, columns).round().astype(int)
    fig, axes = plt.subplots(1, columns, figsize=(6.8, 1.45), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, selected):
        image = axis.imshow(phases[index], cmap="twilight", vmin=0, vmax=2 * np.pi)
        axis.set_title(f"epoch {epochs[index]}"); axis.set_xticks([]); axis.set_yticks([])
    colorbar = fig.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
    colorbar.set_label("phase (rad)")
    fig.savefig(output_dir / "one_expert_phase_snapshots.png", dpi=300)
    fig.savefig(output_dir / "one_expert_phase_snapshots.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.8), constrained_layout=True)
    axes[0].plot(epochs, [row["wrapped_delta_rms_rad"] for row in rows], marker="o", ms=2)
    axes[0].set(xlabel="epoch", ylabel="wrapped RMS (rad)")
    axes[1].plot(epochs, [row["fraction_changed_over_0p05_rad"] for row in rows], marker="o", ms=2)
    axes[1].set(xlabel="epoch", ylabel="changed fraction")
    axes[2].plot(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores)), marker="o", ms=2)
    for index, epoch in enumerate(epochs):
        if index in {0, len(epochs) - 1}:
            axes[2].annotate(str(epoch), (scores[index, 0], scores[index, 1] if scores.shape[1] > 1 else 0))
    axes[2].set(xlabel="circular PC1", ylabel="circular PC2")
    fig.savefig(output_dir / "one_expert_evolution_metrics.png", dpi=300)
    fig.savefig(output_dir / "one_expert_evolution_metrics.pdf")
    plt.close(fig)
    report = {
        "schema_version": 1,
        "plane": plane,
        "frame": frame,
        "expert": expert,
        "snapshot_count": len(paths),
        "epochs": epochs.tolist(),
        "phase_shape": list(phases.shape[1:]),
        "pca_representation": "flatten(concat(cos(phase), sin(phase)))",
        "explained_variance_ratio_first_five": variance_ratio[:5].tolist(),
        "architecture": payloads[0]["architecture"],
    }
    (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", default="phase_snapshots")
    parser.add_argument("--output-dir", default="analysis_one_expert")
    parser.add_argument("--plane", default="parallel_optics.raw_expert_phase")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.snapshot_dir), Path(args.output_dir), args.plane, args.frame, args.expert), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

