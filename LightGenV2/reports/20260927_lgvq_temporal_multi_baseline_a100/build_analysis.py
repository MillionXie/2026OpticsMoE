from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_remote"
HOST_POWER_W = 338.2
OURS = {
    "model": "LightGenV2 Ours",
    "batch_size": 16,
    "formal_srcc": 0.8044,
    "measured_videos": 200,
    "measured_calls": 200,
    "wall_mean_ms_per_batch": 10.52378531679213,
    "wall_p95_ms_per_batch": None,
    "a100_board_power_w": 62.842,
    "host_plus_board_power_w": None,
    "calls_for_16_videos": 1,
    "time_16_videos_ms": 10.52378531679213,
    "energy_16_videos_j": 1.2066879347692436,
    "videos_per_j": 16 / 1.2066879347692436,
    "speedup_vs_ours": 1.0,
    "energy_gain_vs_ours": 1.0,
    "performance_source": "fixed full LGVQ test split",
    "timing_source": "prior unified A100 200-sample narrow-boundary audit",
}
PERFORMANCE = {
    "Qwen3-VL-2B": 0.7663428036522781,
    "DeepSeek-VL2-Tiny": 0.5688169,
    "CLIP ViT-B/32": 0.7251568,
    "YOLO11s": 0.7015822,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def visual_result(model: str, path: Path, result_index: int = 0) -> dict:
    report = read_json(path)
    item = report["results"][result_index]
    batch = int(item["batch_size_videos"])
    time_16 = float(item["equivalent_16_video_time_ms"])
    energy = float(item["equivalent_16_video_host_plus_board_energy_j"])
    return {
        "model": model,
        "batch_size": batch,
        "formal_srcc": PERFORMANCE[model],
        "measured_videos": int(item["measured_videos"]),
        "measured_calls": int(item["measured_calls"]),
        "wall_mean_ms_per_batch": float(item["synchronized_wall_ms_per_batch"]["mean"]),
        "wall_p95_ms_per_batch": float(item["synchronized_wall_ms_per_batch"]["p95"]),
        "a100_board_power_w": float(item["active_a100_board_power_w"]["mean"]),
        "host_plus_board_power_w": HOST_POWER_W + float(item["active_a100_board_power_w"]["mean"]),
        "calls_for_16_videos": int(item["calls_for_16_videos"]),
        "time_16_videos_ms": time_16,
        "energy_16_videos_j": energy,
        "videos_per_j": 16.0 / energy,
        "speedup_vs_ours": time_16 / OURS["time_16_videos_ms"],
        "energy_gain_vs_ours": energy / OURS["energy_16_videos_j"],
        "performance_source": report["performance_source"],
        "timing_source": str(path.relative_to(ROOT)),
    }


def qwen_result(batch: int) -> dict:
    path = RAW / "qwen3_vl_2b_wall_independent" / f"batch_{batch}" / "formal" / f"batch_{batch:02d}" / "report.json"
    report = read_json(path)
    wall_summary = report["model_boundary_all_batches_synchronized_wall_ms"]
    wall = float(wall_summary["mean"])
    power = float(report["telemetry"]["active_mean_w"])
    calls = 16 // batch
    time_16 = wall * calls
    energy = (HOST_POWER_W + power) * time_16 / 1000.0
    return {
        "model": "Qwen3-VL-2B",
        "batch_size": batch,
        "formal_srcc": PERFORMANCE["Qwen3-VL-2B"],
        "measured_videos": int(report["test_videos"]),
        "measured_calls": int(report["full_batches"]),
        "wall_mean_ms_per_batch": wall,
        "wall_p95_ms_per_batch": float(wall_summary["p95"]),
        "a100_board_power_w": power,
        "host_plus_board_power_w": HOST_POWER_W + power,
        "calls_for_16_videos": calls,
        "time_16_videos_ms": time_16,
        "energy_16_videos_j": energy,
        "videos_per_j": 16.0 / energy,
        "speedup_vs_ours": time_16 / OURS["time_16_videos_ms"],
        "energy_gain_vs_ours": energy / OURS["energy_16_videos_j"],
        "performance_source": "fixed full 558-video LGVQ test split",
        "timing_source": str(path.relative_to(ROOT)),
    }


rows = [dict(OURS)]
rows.extend(qwen_result(batch) for batch in (1, 2, 4, 8))
rows.append(visual_result("CLIP ViT-B/32", RAW / "clip_vit_b32" / "report.json", 0))
for batch in (2, 4, 8):
    rows.append(visual_result("CLIP ViT-B/32", RAW / "clip_vit_b32_independent" / f"batch_{batch}" / "report.json"))
for model, folder in (("DeepSeek-VL2-Tiny", "deepseek_vl2_tiny_independent"), ("YOLO11s", "yolo11s_independent")):
    for batch in (1, 2, 4, 8):
        rows.append(visual_result(model, RAW / folder / f"batch_{batch}" / "report.json"))

fieldnames = list(rows[0])
with (ROOT / "calculated_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "calculated_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

colors = {
    "LightGenV2 Ours": "#7B2CBF",
    "Qwen3-VL-2B": "#0072B2",
    "DeepSeek-VL2-Tiny": "#D55E00",
    "CLIP ViT-B/32": "#009E73",
    "YOLO11s": "#E69F00",
}
markers = {"LightGenV2 Ours": "*", "Qwen3-VL-2B": "o", "DeepSeek-VL2-Tiny": "s", "CLIP ViT-B/32": "D", "YOLO11s": "^"}


def plot(x_key: str, xlabel: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    for model in colors:
        selected = [row for row in rows if row["model"] == model]
        selected.sort(key=lambda row: row["batch_size"])
        x = [row[x_key] for row in selected]
        y = [100.0 * row["formal_srcc"] for row in selected]
        ax.plot(x, y, color=colors[model], marker=markers[model], linewidth=1.8,
                markersize=9 if model == "LightGenV2 Ours" else 5.5, label=model)
        for row in selected:
            label = "Optical" if model == "LightGenV2 Ours" else f"B={row['batch_size']}"
            offsets = {1: (5, 6), 2: (5, -13), 4: (-27, 6), 8: (-27, -13)}
            offset = (5, 6) if model == "LightGenV2 Ours" else offsets[row["batch_size"]]
            ax.annotate(label, (row[x_key], 100.0 * row["formal_srcc"]), xytext=offset,
                        textcoords="offset points", fontsize=7, color=colors[model])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Temporal quality SRCC (%)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_ylim(52, 83)
    fig.tight_layout()
    fig.savefig(ROOT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


plot("time_16_videos_ms", "Latency for 16 videos (ms)", "srcc_vs_latency_16videos")
plot("energy_16_videos_j", "Energy for 16 videos (J)", "srcc_vs_energy_16videos")

hash_rows = []
for path in sorted(
    list(RAW.rglob("*"))
    + [ROOT / "calculated_summary.csv", ROOT / "calculated_summary.json",
       ROOT / "srcc_vs_latency_16videos.png", ROOT / "srcc_vs_energy_16videos.png",
       ROOT / "LGVQ_temporal_multi_baseline_A100.xlsx", ROOT / "README.md",
       ROOT / "protocol.json"]
):
    if not path.is_file():
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest})
with (ROOT / "SHA256SUMS.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(hash_rows)

print(json.dumps({"rows": len(rows), "output": str(ROOT)}, ensure_ascii=False))
