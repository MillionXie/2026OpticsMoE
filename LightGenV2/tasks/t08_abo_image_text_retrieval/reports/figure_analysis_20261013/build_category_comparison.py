"""Simple category-level ABO I2T figures from paired saved predictions.

Categories are manual display groups based on official product titles, not model
targets. Correctness remains exact product/SKU retrieval among all 100 titles.
No training, weights, dataset labels or predictions are modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "simple_category_comparison"
# Defined from titles before calculating category performance. Labels are the
# stable easy100 labels, NOT a clustering of successes/failures.
GROUPS = [
    ("家具", "Furniture", [5,8,9,10,11,16,17,22,29,31,43,45,54,56,59,62,76,79,83,84,92,93,94,95,96,98,99]),
    ("收纳用品", "Storage", [6,13,24,42,52,75,80,81,85]),
    ("墙面装饰", "Wall decor", [3,21,23,27,46,50,65,87,97]),
    ("花器与摆件", "Decorative objects", [4,18,30,37,68,74,86]),
    ("灯具", "Lighting", [49,57,58]),
    ("家电", "Appliances", [2,28,34,63,88]),
    ("地毯与地垫", "Rugs & floor mats", [12,48,55,67,72]),
    ("抱枕", "Throw pillows", [14,47,66]),
    ("数码配件", "Tech accessories", [0,1,7,19,20,25,26,35,40,44,51,60,69,70,71,77,90]),
    ("厨房餐具", "Kitchenware", [15,32,33,53,73]),
    ("运动与户外", "Sports & outdoor", [36,39,41,78,89]),
    ("文具与办公", "Office supplies", [38,61,64,82,91]),
]
FILES = ["test.csv", "train.csv", "titles.csv",
         "ours_same_weight_replay_predictions.csv", "qwen_dynamic2048_predictions.csv"]


def read(name):
    with (HERE / "raw" / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, records):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def keyed(rows):
    d = {r["sample_id"]: r for r in rows}
    assert len(d) == len(rows), "Duplicate sample IDs"
    return d


def assemble():
    mapping = {i: (cn, en) for cn, en, ids in GROUPS for i in ids}
    all_ids = [i for _, _, ids in GROUPS for i in ids]
    assert len(all_ids) == len(set(all_ids)) == 100
    assert set(all_ids) == set(range(100))
    test, train, titles = read("test.csv"), read("train.csv"), read("titles.csv")
    ours = keyed(read(FILES[3]))
    qwen = keyed(read(FILES[4]))
    assert set(keyed(test)) == set(ours) == set(qwen)
    assert len(test) == 2400 and len(train) == 4800 and len(titles) == 100
    title_map = {int(r["label"]): r for r in titles}
    train_count = Counter(int(r["label"]) for r in train)
    details = []
    for r in test:
        sid, label = r["sample_id"], int(r["label"])
        o, q = ours[sid], qwen[sid]
        assert int(o["true_label"]) == int(q["true_label"]) == label
        assert o["product_id"] == q["true_product_id"] == r["product_id"]
        assert title_map[label]["product_id"] == r["product_id"]
        assert (int(o["true_rank"]) == 1) == (int(o["predicted_label"]) == label)
        assert (int(q["true_rank"]) == 1) == (int(q["predicted_label"]) == label)
        details.append({"sample_id": sid, "label": label, "product_id": r["product_id"],
                        "category": mapping[label][0], "ours_correct": int(int(o["true_rank"]) == 1),
                        "qwen_correct": int(int(q["true_rank"]) == 1)})
    products = []
    for label in range(100):
        rows = [d for d in details if d["label"] == label]
        assert len(rows) == 24 and train_count[label] == 48
        products.append({"category": mapping[label][0], "category_en": mapping[label][1],
                         "label": label, "product_id": title_map[label]["product_id"],
                         "title": title_map[label]["title"], "train_n": train_count[label],
                         "test_n": len(rows), "ours_correct": sum(d["ours_correct"] for d in rows),
                         "qwen_correct": sum(d["qwen_correct"] for d in rows)})
    categories = []
    for cn, en, ids in GROUPS:
        rows = [p for p in products if p["label"] in ids]
        n = sum(p["test_n"] for p in rows)
        oc, qc = [sum(p[k] for p in rows) for k in ("ours_correct", "qwen_correct")]
        categories.append({"category": cn, "category_en": en, "product_n": len(rows),
                           "test_n": n, "ours_correct": oc, "qwen_correct": qc,
                           "ours_accuracy": oc / n, "qwen_accuracy": qc / n,
                           "difference_pp": (oc - qc) / n * 100})
    total = {"product_n": 100, "test_n": 2400,
             "ours_correct": sum(d["ours_correct"] for d in details),
             "qwen_correct": sum(d["qwen_correct"] for d in details)}
    assert total["ours_correct"] == 1931 and total["qwen_correct"] == 1769
    total["ours_accuracy"] = total["ours_correct"] / total["test_n"]
    total["qwen_accuracy"] = total["qwen_correct"] / total["test_n"]
    total["difference_pp"] = (total["ours_correct"] - total["qwen_correct"]) / total["test_n"] * 100
    return {"categories": categories, "products": products, "queries": details, "total": total}


def plot(data):
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        from matplotlib.font_manager import fontManager
        fontManager.addfont(str(font))
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.unicode_minus": False})
    cats = data["categories"]
    labels = [f'{r["category"]}\n(n={r["test_n"]})' if font.is_file()
              else f'{r["category_en"]}\n(n={r["test_n"]})' for r in cats]
    o, q = [np.array([r[k] * 100 for r in cats]) for k in ("ours_accuracy", "qwen_accuracy")]
    green, blue = "#16835F", "#3976B5"
    y = np.arange(len(cats))
    for kind in ("bar", "scatter", "line"):
        if kind != "line":
            fig, ax = plt.subplots(figsize=(10, 8))
            if kind == "bar":
                ax.barh(y - .18, o, height=.34, color=green, label="Ours")
                ax.barh(y + .18, q, height=.34, color=blue, label="Qwen (2048D)")
                for yy, a, b in zip(y, o, q):
                    ax.text(a + .8, yy - .18, f"{a:.1f}", va="center", fontsize=9)
                    ax.text(b + .8, yy + .18, f"{b:.1f}", va="center", fontsize=9)
            else:
                ax.scatter(o, y - .12, s=65, color=green, label="Ours", zorder=3)
                ax.scatter(q, y + .12, s=65, marker="s", color=blue, label="Qwen (2048D)", zorder=3)
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set_xlim((0, 108) if kind == "bar" else (-2, 102))
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("Top-1 准确率 (%)" if font.is_file() else "Top-1 accuracy (%)")
            ax.grid(axis="x", alpha=.2)
        else:
            fig, ax = plt.subplots(figsize=(13, 5.4))
            ax.plot(y, o, "o-", color=green, lw=1.8, label="Ours")
            ax.plot(y, q, "s-", color=blue, lw=1.8, label="Qwen (2048D)")
            ax.set_xticks(y, labels, fontsize=10)
            ax.set_ylim(0, 108)
            ax.set_yticks(range(0, 101, 20))
            ax.set_ylabel("Top-1 准确率 (%)" if font.is_file() else "Top-1 accuracy (%)")
            ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=2, frameon=False)
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            fig.savefig(OUT / f"category_{kind}.{ext}", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = assemble()
    data["sources"] = [{"file": name, "sha256": hashlib.sha256((HERE/"raw"/name).read_bytes()).hexdigest()}
                       for name in FILES]
    data["method"] = "Title-based manual display categories; exact-SKU Top-1 correctness; micro-averaged within category."
    data["historical_ours"] = {"correct": 1934, "n": 2400, "accuracy": 1934/2400,
                                "per_query_available": False}
    data["selection_note"] = "Historical TEST-selected model, same-product spin-view split; not independent product generalization."
    (OUT/"category_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv("category_accuracy.csv", data["categories"])
    write_csv("product_category_mapping.csv", data["products"])
    write_csv("paired_queries.csv", data["queries"])
    plot(data)
    import html
    links = "".join(f'<h2>{title}</h2><a href="category_{kind}.svg">SVG</a> · <a href="category_{kind}.pdf">PDF</a><br><img src="category_{kind}.png">'
                    for kind, title in [("bar", "柱状图"), ("scatter", "散点图"), ("line", "折线图")])
    (OUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>ABO 类别对比</title><style>body{font-family:Microsoft YaHei,Arial;max-width:1100px;margin:30px auto}img{max-width:100%}h2{margin-top:45px}</style><h1>ABO 图搜文：按类别比较</h1><p>100 商品归入 12 个展示类别；每类统计检索到正确商品的比例，不是把同类商品算作命中。</p><p>逐样本回放：Ours 80.4583%，冻结 Qwen 动态2048D 73.7083%。历史 Ours 80.5833% 无完整逐样本排名，本图不混用。</p><p><a href="ABO_category_comparison.xlsx">Excel</a> · <a href="category_accuracy.csv">类别绘图数据</a> · <a href="product_category_mapping.csv">商品归类表</a> · <a href="paired_queries.csv">逐样本数据</a></p>'+links, encoding="utf-8")
    print(json.dumps({"categories": data["categories"], "total": data["total"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
