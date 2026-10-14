"""ABO image-to-image: fixed physical rank72 vs frozen Qwen-64D.

Read saved predictions only. No inference, training or hardware operations.
Category aggregation preserves exact-SKU correctness and all original queries.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "LightGenV2").is_dir())
SOURCE = {
    "ours_report.json": "handoffs/abo_latestfresh35_lab_20260930/rank72_physical_full_report_20261001.json",
    "protocol.json": "handoffs/abo_latestfresh35_lab_20260930/rank72_dataset_protocol.json",
    "qwen_report.json": "LightGenV2/reports/baseline_plotting_20260922/_server_raw/abo_image_to_image/final_report.json",
    "qwen_predictions.csv": "LightGenV2/reports/baseline_plotting_20260922/_server_raw/abo_image_to_image/normal_predictions.csv",
    "dataset_manifest.csv": "LightGenV2/tasks/t07_abo_image_retrieval/releases/abo_qwen_baseline_reproduction_20260914/evidence/abo_similarity10_manifest.csv",
}
DISPLAY = {"chair": "Chairs", "rug": "Rugs", "sofa": "Sofas", "wall art": "Wall art",
           "light fixture": "Light fixtures", "stool seating": "Stools", "pillow": "Pillows",
           "bed": "Beds", "home mirror": "Mirrors", "vase": "Vases"}
GREEN, BLUE = "#16835F", "#3976B5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(name):
    with (REPO / SOURCE[name]).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def unique(rows):
    result = {r["sample_id"]: r for r in rows}
    assert len(result) == len(rows), "Duplicate sample IDs"
    return result


def analyze():
    load = lambda name: json.loads((REPO / SOURCE[name]).read_text(encoding="utf-8-sig"))
    ours, qwen, protocol = [load(n) for n in ("ours_report.json", "qwen_report.json", "protocol.json")]
    assert ours["checkpoint_sha256"] == "25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22"
    assert ours["status"] == qwen["status"] == "complete"
    assert ours["physical_to_physical"] is True
    assert qwen["frozen"] is True and qwen["fitted_on_this_dataset"] is False
    assert qwen["descriptor_dimension"] == 64
    assert digest(REPO / SOURCE["protocol.json"]) == qwen["manifest_sha256"]
    assert digest(REPO / SOURCE["dataset_manifest.csv"]) == protocol["parent_manifest_sha256"]
    meta = unique(csv_read("dataset_manifest.csv"))
    pr = unique(protocol["rows"])
    query = {k: v for k, v in pr.items() if v["split"] == "query"}
    gallery = {k: v for k, v in pr.items() if v["split"] == "train"}
    op, qp = unique(ours["predictions"]), unique(csv_read("qwen_predictions.csv"))
    assert len(query) == len(op) == len(qp) == 800
    assert len(gallery) == 1600 and not set(gallery) & set(query)
    assert set(op) == set(qp) == set(query)
    assert all(v == 2400 for v in ours["ccd_counts"].values()) and len(ours["ccd_counts"]) == 6
    assert len(set(r["product_id"] for r in query.values())) == 200
    assert set(Counter(r["product_id"] for r in query.values()).values()) == {4}
    assert set(Counter(r["product_id"] for r in gallery.values()).values()) == {8}
    paired = []
    for sid, qr in query.items():
        o, q, m = op[sid], qp[sid], meta[sid]
        assert o["product_id"] == q["product_id"] == qr["product_id"] == m["product_id"]
        assert qr["category_id"] == int(m["category_id"])
        for pred in (o, q):
            target = gallery[pred["top1_sample_id"]]
            assert target["product_id"] == pred["top1_product_id"]
            assert int(pred["hit_at_1"]) == int(pred["product_id"] == pred["top1_product_id"])
        rank = int(o["rank_of_first_relevant"])
        assert (rank == 1) == bool(o["hit_at_1"])
        paired.append({"sample_id": sid, "product_id": qr["product_id"],
                       "category_id": qr["category_id"], "dataset_category": m["class_name"],
                       "category": DISPLAY[m["class_name"]],
                       "ours_hit1": int(o["hit_at_1"]), "qwen_hit1": int(q["hit_at_1"]),
                       "ours_first_relevant_rank": rank, "ours_top1_product_id": o["top1_product_id"],
                       "qwen_top1_product_id": q["top1_product_id"]})
    cats = []
    for cid in sorted(set(r["category_id"] for r in paired)):
        rows = [r for r in paired if r["category_id"] == cid]
        assert len(rows) == 80 and len(set(r["product_id"] for r in rows)) == 20
        assert len(set(r["dataset_category"] for r in rows)) == 1
        oc, qc = [sum(r[k] for r in rows) for k in ("ours_hit1", "qwen_hit1")]
        cats.append({"category_id": cid, "dataset_category": rows[0]["dataset_category"],
                     "category": rows[0]["category"], "products": 20, "queries": 80,
                     "ours_correct": oc, "qwen_correct": qc,
                     "ours_accuracy": oc / 80, "qwen_accuracy": qc / 80,
                     "difference_pp": (oc - qc) / 80 * 100})
    overall = []
    for k in (1, 5, 10):
        oc = sum(r["ours_first_relevant_rank"] <= k for r in paired)
        assert abs(oc / 800 - ours["metrics"][f"recall_at_{k}"]) < 1e-12
        qrate = qwen["metrics"]["normal"][f"hit_at_{k}"]
        qc = round(qrate * 800)
        assert abs(qc / 800 - qrate) < 1e-12
        if k == 1:
            assert sum(r["qwen_hit1"] for r in paired) == qc == 681
            assert oc == 649
        overall.append({"metric": f"Hit@{k}", "queries": 800, "ours_correct": oc,
                        "qwen_correct": qc, "ours_rate": oc / 800, "qwen_rate": qrate,
                        "difference_pp": (oc - qc) / 800 * 100})
    outcome = Counter((r["ours_hit1"], r["qwen_hit1"]) for r in paired)
    return {"categories": cats, "overall": overall, "queries": paired,
            "paired_outcomes": {"both_correct": outcome[1, 1], "ours_only_correct": outcome[1, 0],
                                "qwen_only_correct": outcome[0, 1], "both_wrong": outcome[0, 0]},
            "notes": ["Exact-SKU retrieval, 800 held-out views of 200 enrolled products, 1600 gallery views.",
                      "Ours uses real six-stage CCD features for both gallery and query; Qwen uses original RGB images.",
                      "Categories are inherited class_name labels, not manually relabeled or category-level relevance.",
                      "Qwen Top-5/10 are saved full-report aggregates; paired Qwen input exposes Top-1 only.",
                      "Historical TEST-selected model and same-spin views; not independent unseen-product generalization.",
                      "Single fixed-run descriptive results; bars have no fabricated repeats or confidence intervals."]}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(d):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    cats = d["categories"]
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(9.8, 7.1))
    o = np.array([r["ours_accuracy"] for r in cats]) * 100
    q = np.array([r["qwen_accuracy"] for r in cats]) * 100
    ax.barh(y - .18, o, height=.34, color=GREEN, label="Ours (experiment)")
    ax.barh(y + .18, q, height=.34, color=BLUE, label="Qwen (64D)")
    for yy, a, b in zip(y, o, q):
        ax.text(a + .7, yy - .18, f"{a:.2f}", va="center", fontsize=9)
        ax.text(b + .7, yy + .18, f"{b:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, [r["category"] for r in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.09))
    save(fig, "01_category_top1")
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for values, shift, color, label in [([r["ours_rate"]*100 for r in d["overall"]], -.18, GREEN, "Ours (experiment)"),
                                        ([r["qwen_rate"]*100 for r in d["overall"]], .18, BLUE, "Qwen (64D)")]:
        bars = ax.bar(x + shift, values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)
    ax.set_xticks(x, [r["metric"] for r in d["overall"]])
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Retrieval success (%)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.13))
    save(fig, "02_overall_retrieval")


def main():
    data = analyze()
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    data["sources"] = []
    for name, relative in SOURCE.items():
        src = REPO / relative
        dst = raw / name
        if dst.exists():
            assert digest(dst) == digest(src), f"Refusing to overwrite different source: {dst}"
        else:
            shutil.copyfile(src, dst)
        data["sources"].append({"local_name": name, "source": relative, "sha256": digest(src)})
    for name, rows in [("category_accuracy.csv", data["categories"]), ("overall_metrics.csv", data["overall"]),
                       ("paired_queries.csv", data["queries"])]:
        csv_write(HERE / name, rows)
    (HERE / "statistics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(data)
    sections = "".join(f'<h2>{title}</h2><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a><br><img src="{name}.png">' for name, title in [("01_category_top1", "1. 各类别 Top-1 对比（推荐）"), ("02_overall_retrieval", "2. 总体 Top-1 / Top-5 / Top-10 对比")])
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in data["notes"])
    (HERE / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>ABO 图搜图实拍对比</title><style>body{font-family:Arial,Microsoft YaHei;max-width:1000px;margin:32px auto}img{max-width:100%}h2{margin-top:40px}</style><h1>ABO 图搜图：实拍 Ours 与冻结 Qwen</h1><p>同一组800查询、1600图库、200商品。10个原数据集类别，每类20商品、80查询。命中仍须为同一商品，而不是同类。</p><p>Ours 649/800 = 81.125%；Qwen-64D 681/800 = 85.125%。只读统计原报告，没有重新训练或推理。</p><p><a href="ABO_I2I_statistics.xlsx">Excel</a> · <a href="category_accuracy.csv">类别数据</a> · <a href="overall_metrics.csv">总体数据</a> · <a href="paired_queries.csv">逐查询配对</a> · <a href="raw/">原始报告</a></p>'+sections+'<h2>口径说明</h2><ul>'+notes+'</ul>', encoding="utf-8")
    print(json.dumps({"categories":data["categories"],"overall":data["overall"],"paired_outcomes":data["paired_outcomes"]}, indent=2))


if __name__ == "__main__":
    main()
