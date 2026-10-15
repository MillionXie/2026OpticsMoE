"""English quantitative figures: adopted physical T2I 85% vs frozen Qwen 82%.

No model is loaded here. Ours per-title rows come from the separately SHA-pinned
CPU readout replay, not a different training budget or a simulated checkpoint.
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
    "ours_report.json": "handoffs/abo_text_to_image_10cm_alpha040_20260925/finetune10/finetune_report.json",
    "qwen_report.json": "LightGenV2/reports/baseline_plotting_20260922/_server_raw/abo_text_to_image/report.json",
    "qwen_predictions.json": "LightGenV2/reports/baseline_plotting_20260922/_server_raw/abo_text_to_image/dynamic_2048_predictions.json",
    "titles.csv": "LightGenV2/tasks/t08_abo_image_text_retrieval/reports/figure_analysis_20261013/raw/titles.csv",
    "test.csv": "LightGenV2/tasks/t08_abo_image_text_retrieval/reports/figure_analysis_20261013/raw/test.csv",
    "category_mapping.csv": "LightGenV2/tasks/t08_abo_image_text_retrieval/reports/figure_analysis_20261013/simple_category_comparison/product_category_mapping.csv",
}
GREEN, BLUE = "#16835F", "#3976B5"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(name):
    with (REPO / SOURCE[name]).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze():
    load = lambda n: json.loads((REPO / SOURCE[n]).read_text(encoding="utf-8-sig"))
    ours_report, qreport, qp = [load(n) for n in ("ours_report.json", "qwen_report.json", "qwen_predictions.json")]
    replay = json.loads((HERE / "raw/ours_cpu_replay.json").read_text(encoding="utf-8"))
    assert replay["readout_sha256"] == ours_report["best_checkpoint_sha256"] == "89e25360c8f12c9d13dbceba49e8d81073d4dd75c6acf667ab12a38b8fd293c5"
    assert replay["body_sha256"] == ours_report["source_checkpoint_sha256"]
    assert replay["device"] == "cpu" and not replay["training"]
    assert replay["query_count"] == 100 and replay["gallery_count"] == 2400
    assert qreport["training"] is False and qreport["trainable_parameters"] == 0
    assert ours_report["training"]["epochs"] == 10 and ours_report["training"]["selected_epoch"] == 9
    for name in ("test", "titles"):
        assert sha(REPO / SOURCE[f"{name}.csv"]) == qreport["data_sha256"][f"{name}.csv"] == replay["data_sha256"][name]
    titles, test, mapping = [read_csv(n) for n in ("titles.csv", "test.csv", "category_mapping.csv")]
    tm = {r["sample_id"]: r for r in test}
    titlemap = {r["product_id"]: r for r in titles}
    mm = {r["product_id"]: r for r in mapping}
    om = {r["product_id"]: r for r in replay["predictions"]}
    qm = {r["product_id"]: r for r in qp}
    assert len(titles) == len(mm) == len(om) == len(qm) == 100
    assert set(titlemap) == set(om) == set(qm) == set(mm)
    assert len(tm) == 2400 and set(tm) == set(qreport["gallery_image_sha256"])
    assert set(Counter(r["product_id"] for r in test).values()) == {24}
    paired, top10 = [], []
    for t in titles:
        pid = t["product_id"]
        o, q, m = om[pid], qm[pid], mm[pid]
        assert o["title"] == q["title"] == t["title"] == m["title"]
        assert int(t["label"]) == int(m["label"]) == o["title_label"]
        ranks = [int(o["first_positive_rank"]), int(q["first_positive_rank"])]
        assert len(o["positive_ranks"]) == 24
        for method, item, rank in zip(("ours", "qwen"), (o, q), ranks):
            ids = item["top10_sample_ids"]
            assert len(ids) == len(set(ids)) == 10 and set(ids) <= set(tm)
            matches = [i+1 for i, sid in enumerate(ids) if tm[sid]["product_id"] == pid]
            assert (matches[0] == rank) if rank <= 10 else (not matches)
            for position, sid in enumerate(ids, 1):
                top10.append({"method": method, "title_label": int(t["label"]), "query_product_id": pid,
                              "rank": position, "image_sample_id": sid, "retrieved_product_id": tm[sid]["product_id"],
                              "same_sku": int(tm[sid]["product_id"] == pid)})
        paired.append({"title_label": int(t["label"]), "product_id": pid, "category": m["category_en"],
                       "title": t["title"], "ours_first_rank": ranks[0], "qwen_first_rank": ranks[1],
                       **{f"{method}_hit{k}": int(rank <= k) for method, rank in zip(("ours", "qwen"), ranks) for k in (1,5,10)},
                       "ours_reciprocal_rank": 1/ranks[0], "qwen_reciprocal_rank": 1/ranks[1],
                       "ours_average_precision": o["average_precision"]})
    historical = ours_report["selected"]["untouched_test"]
    qb = qreport["metrics"]["dynamic_2048"]
    overall = []
    for k, expected in zip((1,5,10), ((85,82),(97,90),(97,96))):
        oc, qc = [sum(p[f"{method}_hit{k}"] for p in paired) for method in ("ours", "qwen")]
        assert (oc, qc) == expected
        assert abs(oc/100 - historical[f"hit_at_{k}"]) < 1e-6
        assert abs(qc/100 - qb[f"hit_at_{k}"]) < 1e-10
        overall.append({"metric":f"Hit@{k}", "queries":100, "ours_correct":oc, "qwen_correct":qc,
                        "ours_rate":oc/100, "qwen_rate":qc/100, "difference_pp":oc-qc})
    categories = []
    for cat in dict.fromkeys(r["category_en"] for r in mapping):
        rows = [r for r in paired if r["category"] == cat]
        oc, qc = [sum(r[k] for r in rows) for k in ("ours_hit1", "qwen_hit1")]
        categories.append({"category":cat,"queries":len(rows),"ours_correct":oc,"qwen_correct":qc,
                           "ours_accuracy":oc/len(rows),"qwen_accuracy":qc/len(rows),"difference_pp":(oc-qc)/len(rows)*100})
    # Preserve the same twelve display-category order as the earlier I2T figure.
    order = ["Furniture","Storage","Wall decor","Decorative objects","Lighting","Appliances",
             "Rugs & floor mats","Throw pillows","Tech accessories","Kitchenware","Sports & outdoor","Office supplies"]
    assert set(order) == {c["category"] for c in categories}
    categories.sort(key=lambda c: order.index(c["category"]))
    mrr = {method:sum(r[f"{method}_reciprocal_rank"] for r in paired)/100 for method in ("ours","qwen")}
    assert abs(mrr["ours"]-historical["mrr"])<1e-6 and abs(mrr["qwen"]-qb["mrr"])<1e-10
    full_metrics = [{"metric":k,"ours":historical[k],"qwen":qb[k],"ours_cpu_replay":replay["metrics"][k]}
                    for k in ("hit_at_1","hit_at_5","hit_at_10","mrr","map")]
    outcomes = Counter((r["ours_hit1"],r["qwen_hit1"]) for r in paired)
    return {"categories":categories,"overall":overall,"queries":paired,"top10":top10,"full_metrics":full_metrics,
            "paired_outcomes":{"both_correct":outcomes[1,1],"ours_only_correct":outcomes[1,0],
                               "qwen_only_correct":outcomes[0,1],"both_wrong":outcomes[0,0]},
            "notes":["100 official-title queries rank 2400 TEST images; 24 exact-SKU positives per title.",
                     "Ours is the adopted physical-feature model after 10-epoch readout adaptation, validation-selected epoch 9.",
                     "Ours per-title statistics are a new SHA-pinned CPU replay of the original measured-feature cache, not historical saved predictions.",
                     "Qwen is frozen Qwen3-VL-Embedding-2B, native-aspect RGB, 2048D. The 64D baseline is not the 82% result.",
                     "Twelve manually assigned title-based display groups, identical to the earlier I2T display mapping; exact-SKU relevance is unchanged.",
                     "One query per product. Small category n reflects titles, not the 24 relevant images. No invented repeats or confidence intervals.",
                     "Original TEST has development exposure. Cache-to-raw-CCD numerical reconstruction is not revalidated here."]}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png","svg","pdf"):
        fig.savefig(HERE/f"{name}.{ext}",dpi=400,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def plot(d):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
    cats=d["categories"]; y=np.arange(len(cats))
    fig, ax=plt.subplots(figsize=(10,8))
    for shift,key,color,label in [(-.18,"ours_accuracy",GREEN,"Ours (experiment, adapted)"),(.18,"qwen_accuracy",BLUE,"Qwen (2048D)")]:
        vals=np.array([r[key]*100 for r in cats]); ax.barh(y+shift,vals,height=.34,color=color,label=label)
        for yy,v in zip(y+shift,vals): ax.text(v+.7,yy,f"{v:.1f}",va="center",fontsize=9)
    ax.set_yticks(y,[f'{r["category"]}\n(n={r["queries"]})' for r in cats]); ax.invert_yaxis()
    ax.set_xlim(0,109); ax.set_xticks(range(0,101,20)); ax.set_xlabel("Top-1 hit rate (%)")
    ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.09))
    save(fig,"01_category_top1")
    fig,ax=plt.subplots(figsize=(7.1,4.6)); x=np.arange(3)
    for shift,key,color,label in [(-.18,"ours_rate",GREEN,"Ours (experiment, adapted)"),(.18,"qwen_rate",BLUE,"Qwen (2048D)")]:
        vals=[r[key]*100 for r in d["overall"]]; bars=ax.bar(x+shift,vals,width=.34,color=color,label=label)
        ax.bar_label(bars,labels=[f"{v:.0f}" for v in vals],padding=4)
    ax.set_xticks(x,[r["metric"] for r in d["overall"]]); ax.set_ylim(0,110); ax.set_yticks(range(0,101,20))
    ax.set_ylabel("Title-query success (%)"); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.13))
    save(fig,"02_overall_retrieval")


def main():
    d=analyze(); d["sources"]=[]
    raw=HERE/"raw"; raw.mkdir(exist_ok=True)
    for name,rel in SOURCE.items():
        src=REPO/rel; dst=raw/name
        if dst.exists(): assert sha(dst)==sha(src), f"Refusing to overwrite source {dst}"
        else: shutil.copyfile(src,dst)
        d["sources"].append({"file":name,"source":rel,"sha256":sha(src)})
    d["sources"].append({"file":"ours_cpu_replay.json","sha256":sha(raw/"ours_cpu_replay.json")})
    for name,key in [("category_accuracy.csv","categories"),("overall_metrics.csv","overall"),
                     ("paired_titles.csv","queries"),("top10_matches.csv","top10"),("all_metrics.csv","full_metrics")]:
        write_csv(name,d[key])
    (HERE/"statistics.json").write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
    plot(d)
    sections="".join(f'<h2>{title}</h2><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a><br><img src="{name}.png">' for name,title in [("01_category_top1","1. 类别 Top-1 对比"),("02_overall_retrieval","2. 总体 Top-1/5/10 对比")])
    notes="".join(f"<li>{html.escape(n)}</li>" for n in d["notes"])
    (HERE/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>ABO 文搜图实拍定量结果</title><style>body{font-family:Arial,Microsoft YaHei;max-width:1050px;margin:32px auto}img{max-width:100%}h2{margin-top:40px}</style><h1>ABO 文搜图：实拍采用版 85% 与冻结 Qwen 82%</h1><p>100条标题查询 → 2400张候选图片，每条24张相关图片。Ours为实拍特征＋10轮末端读出适配版本，不是未微调实拍。逐条排名来自原SHA缓存和权重的CPU回放，Top-1/5/10复现85/97/97。</p><p>类别图沿用图搜文的12个人工展示大类；n为标题数，不是图片数。没有改变“同商品才正确”的判定。</p><p><a href="ABO_T2I_statistics.xlsx">Excel</a> · <a href="category_accuracy.csv">类别数据</a> · <a href="overall_metrics.csv">总体数据</a> · <a href="paired_titles.csv">逐标题配对</a> · <a href="top10_matches.csv">Top10</a> · <a href="raw/">原始报告</a></p>'+sections+'<h2>说明</h2><ul>'+notes+'</ul>',encoding="utf-8")
    print(json.dumps({"overall":d["overall"],"categories":d["categories"],"paired_outcomes":d["paired_outcomes"]},indent=2))


if __name__=="__main__": main()
