"""Redraw the verified EuroSAT figure without the D2NN A+B series."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / ".codex_plot_deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path("D:/陈课题组/LGII_f2_code/图2d代码/EuroSAT_三模型性能与路由_20260914")
OUT.mkdir(parents=True, exist_ok=True)
completed = ROOT / "completed"
perf_path = completed / "archive/moe_root/results/PERFORMANCE.json"
routing_path = completed / "ROUTING_TEST_VERIFICATION.json"
perf = json.loads(perf_path.read_text(encoding="utf-8"))
verified = json.loads(routing_path.read_text(encoding="utf-8"))
assert verified["passed"]
assert json.loads((completed / "LOCAL_VERIFICATION.json").read_text(encoding="utf-8"))["passed"]
routing = verified["routing_test"]
assert routing["A"]["samples"] == routing["B"]["samples"] == 5429

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"width_ratios": [1.25, 1]})
order = ["moe", "A_only", "B_only"]
names = ["光路由 MoE A+B", "D2NN A-only", "D2NN B-only"]
x = np.arange(len(order))
width = .36
export_data = {"accuracy_percent": {}, "mean_input_power_percent": {}, "performance_source": str(perf_path), "routing_source": str(routing_path), "excluded_model": "D2NN A+B"}
for k, (domain, color) in enumerate([("A", "#2764aa"), ("B", "#db873b")]):
    values = [100 * perf["models"][m]["tests"][domain]["accuracy"] for m in order]
    export_data["accuracy_percent"][domain] = dict(zip(names, values))
    bars = axes[0].bar(x + (k - .5) * width, values, width, label=domain + (" 光学" if domain == "A" else " SAR"), color=color)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
axes[0].set(xticks=x, xticklabels=names, ylim=(0, 105), ylabel="测试准确率（%）", title="三模型固定检查点的双域测试")
axes[0].tick_params(axis="x", labelsize=9)
axes[0].legend(loc="upper right", frameon=False)
axes[0].grid(axis="y", alpha=.18)
axes[0].set_axisbelow(True)
arr = np.array([routing[d]["mean_power"] for d in ("A", "B")]) * 100
assert arr.shape == (2, 4) and np.allclose(arr.sum(axis=1), 100, atol=1e-5)
im = axes[1].imshow(arr, vmin=0, vmax=100, cmap="Blues", aspect="auto")
for i in range(2):
    export_data["mean_input_power_percent"][("A", "B")[i]] = arr[i].tolist()
    for j in range(4):
        axes[1].text(j, i, f"{arr[i, j]:.2f}%", ha="center", va="center", color="white" if arr[i, j] > 60 else "#152334")
axes[1].set(xticks=range(4), xticklabels=["E0（A）", "E1（A）", "E2（B）", "E3（B）"], yticks=[0, 1], yticklabels=["A 测试", "B 测试"], title="MoE 平均输入功率份额")
fig.colorbar(im, ax=axes[1], label="功率份额（%）", fraction=.05, pad=.04)
fig.suptitle("EuroSAT 光学/SAR 十分类 · seed 42 · A/B 各测试 5,429 张", fontsize=13)
fig.tight_layout()
stem = "EuroSAT性能与路由_不含D2NN_AB"
fig.savefig(OUT / f"{stem}.png", dpi=300)
fig.savefig(OUT / f"{stem}.svg")
plt.close(fig)
(OUT / "figure_data.json").write_text(json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"png": str(OUT / f"{stem}.png"), "svg": str(OUT / f"{stem}.svg"), "data": export_data}, ensure_ascii=False, indent=2))
