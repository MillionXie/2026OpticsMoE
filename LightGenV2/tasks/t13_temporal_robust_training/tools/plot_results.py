"""Plot measured five-condition story plus optional simulation sensitivity.

Input: {conditions:[{id,scope,metrics:{srcc},checkpoint_sha256,session}],
        noise:[{group,scale,srcc}], pitch:[{group,pitch_um,srcc}]}.
G1 must be simulation; G2-G5 hardware. No synthetic expected numbers supported.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def validate_conditions(rows):
    lookup = {row["id"]: row for row in rows}
    if len(rows) != 5 or set(lookup) != {"G1", "G2", "G3", "G4", "G5"}:
        raise ValueError("Require exactly five measured/reportable condition rows")
    for key, row in lookup.items():
        scope = "simulation" if key == "G1" else "hardware"
        value = row["metrics"]["srcc"]
        if row["scope"] != scope or value is None or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError(f"Wrong scope or invalid SRCC for {key}")
        if not row.get("checkpoint_sha256") or (scope == "hardware" and not row.get("session")):
            raise ValueError("Checkpoint/session provenance is required")
    if lookup["G1"]["checkpoint_sha256"] != lookup["G2"]["checkpoint_sha256"]:
        raise ValueError("G1 must reuse G2 checkpoint")
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = validate_conditions(data["conditions"])
    if args.output.exists():
        parser.error("Output directory must be new")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = 1 + bool(data.get("noise")) + bool(data.get("pitch"))
    fig, axes = plt.subplots(1, panels, figsize=(5*panels, 4), squeeze=False)
    ax = axes[0, 0]
    scores = [rows[f"G{i}"]["metrics"]["srcc"] for i in range(1, 6)]
    ax.scatter([0], scores[:1], marker="o", facecolors="none", edgecolors="#777777", s=75, label="Ideal simulation (same G2 weights)")
    ax.plot([0, 1], scores[:2], linestyle="--", color="#bbbbbb")
    ax.plot(range(1, 5), scores[1:], "o-", color="#2a6f97", label="Hardware")
    for i in range(2, 5):
        ax.annotate(f"{scores[i]-scores[i-1]:+.3f}", (i-0.5, (scores[i]+scores[i-1])/2), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set_xticks(range(5), ["G1\nIdeal", "G2\nPost", "G3\n+CCD", "G4\n+DC", "G5\nIn-train"])
    ax.set_ylabel("SRCC")
    ax.set_title("a  Deployment recovery (observed, not fitted)")
    ax.legend(fontsize=8)
    panel = 1
    for key, xname, title in (("noise", "scale", "CCD noise sensitivity (simulation)"),
                              ("pitch", "pitch_um", "Device pitch sensitivity (simulation)")):
        if not data.get(key):
            continue
        ax = axes[0, panel]
        for group in sorted({row["group"] for row in data[key]}):
            curve = sorted((row for row in data[key] if row["group"] == group), key=lambda row: row[xname])
            if any(not math.isfinite(row["srcc"]) or not -1 <= row["srcc"] <= 1 for row in curve):
                raise ValueError("Invalid curve SRCC")
            ax.plot([row[xname] for row in curve], [row["srcc"] for row in curve], "o-", label=group)
        ax.set_xlabel("Empirical noise scale" if key == "noise" else "Device pixel pitch (um)")
        ax.set_ylabel("SRCC")
        ax.set_title(title)
        ax.legend(fontsize=8)
        panel += 1
    for ax in axes[0]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "robustness_main.svg")
    fig.savefig(args.output / "robustness_main.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
