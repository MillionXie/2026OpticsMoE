"""Four predeclared examples: native amplitude, simulated CCD, raw CCD, display-only contrast."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
REPORT = HERE.parents[1] / "reports" / "20260924_mnist_dvp"
PACKAGE = ROOT / "MNIST_10cm_8um_Bench_Test_20260923" / "02_mnist_10cm" / "inputs_40_fixed"
REFERENCE = ROOT / "ABO_Lab_SHS_8um" / "assets" / "mnist_native8_completed_20260913" / "paired"
KEYS = ["mnist_i00013_y0", "mnist_i00014_y1", "mnist_i00038_y2", "mnist_i00032_y3"]

ref = json.loads((REFERENCE / "pair_reference.json").read_text(encoding="utf-8"))
analysis = json.loads((REPORT / "mnist40_12ms_analysis.json").read_text(encoding="utf-8"))
rows = {r["key"]: r for r in analysis["selected_rows"]}
lookup = {r["key"]: i for i, r in enumerate(ref["rows"])}
with np.load(REFERENCE / "pair_reference.npz") as z:
    sim = np.stack([z["ccd_B"][lookup[k]] for k in KEYS])
sim_top = float(np.percentile(sim, 99.5))

fig, axes = plt.subplots(4, 4, figsize=(12.4, 12.4), constrained_layout=True)
titles = ["Amplitude BMP (0-255)", f"Simulation (shared 0-{sim_top:.2g})", "CCD linear (0-255)", "CCD display (0-120)"]
for j, title in enumerate(titles):
    axes[0, j].set_title(title, fontsize=11)
for i, key in enumerate(KEYS):
    amp = np.asarray(Image.open(PACKAGE / (key + ".bmp")))
    amp = amp[32:1048, 452:1468]
    measured = np.flipud(np.asarray(Image.open(REPORT / "mnist40_12ms_canonical" / (key + ".png"))))
    labels = [amp, sim[i], measured, measured]
    maxima = [255, sim_top, 255, 120]
    for j, (a, high) in enumerate(zip(labels, maxima)):
        ax = axes[i, j]
        ax.imshow(a, cmap="gray", vmin=0, vmax=high, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 3:
            for x0, y0, x1, y1 in ref["detector_bounds"]:
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               fill=False, edgecolor="red", linewidth=1.2))
    r = rows[key]
    axes[i, 0].set_ylabel(f"{key}\ntrue={r['label']}  pred={r['prediction']}\nPCC={r['pcc']:.3f}", fontsize=10)
fig.suptitle("MNIST-4 | 8 um native B phase | 12 ms DVP exposure | 240 ms SLM wait\nCCD geometry: 2026-09-24 four corners + vertical flip", fontsize=13)
out = REPORT / "four_examples_12ms.png"
fig.savefig(out, dpi=150)
print(out)
