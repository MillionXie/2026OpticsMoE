"""Export delivered per-joint pose measurements and two descriptive research figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .personal_prepare import sha256

KINDS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
CHINESE = dict(zip(KINDS, ["肩", "肘", "腕", "髋", "膝", "踝"]))
COLORS = ["#0072B2", "#E69F00"]


def export(delivery, output, overwrite=False):
    delivery, output = Path(delivery), Path(output)
    summary = json.loads((delivery / "SUMMARY.json").read_text(encoding="utf-8"))
    raw, persons, identities = [], [], {}
    for path in sorted(delivery.glob("photo_*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        identities[str(path.relative_to(delivery))] = sha256(path)
        persons.append(m)
        for j in m["joints"]:
            side, kind = j["name"].split("_", 1)
            error = j["error_px_224"]
            norm = error / m["torso_scale_224"] if j["valid"] else None
            if j["valid"] and bool(norm <= .2) != j["pck_hit"]:
                raise ValueError("Recorded hit differs from error threshold")
            raw.append({"sample_id": m["sample_id"], "original_sample_id": m["original_sample_id"],
                        "capture_group": m["capture_group"], "joint": j["name"],
                        "joint_cn": ("左" if side == "left" else "右") + CHINESE[kind],
                        "side": side, "joint_type": kind, "valid": int(j["valid"]),
                        "error_px_224": error, "torso_scale_224": m["torso_scale_224"],
                        "normalized_error": norm, "pck_hit": int(j["pck_hit"]) if j["valid"] else None})
    valid = [r for r in raw if r["valid"]]
    hits = sum(r["pck_hit"] for r in valid)
    assert len(valid) == summary["evaluated_joints"] and hits == summary["correct_joints"]
    assert len(persons) == summary["person_samples"]
    assert abs(np.mean([r["error_px_224"] for r in valid]) - summary["mean_pixel_error_224"]) < 1e-5
    stats = []
    for kind in KINDS:
        for side in ["left", "right"]:
            group = [r for r in valid if r["joint"] == side + "_" + kind]
            errors = np.array([r["normalized_error"] for r in group])
            q1, med, q3 = np.quantile(errors, [.25, .5, .75], method="linear")
            iqr = q3 - q1
            inside = errors[(errors >= q1 - 1.5*iqr) & (errors <= q3 + 1.5*iqr)]
            stats.append({"joint": side + "_" + kind, "joint_cn": group[0]["joint_cn"],
                          "joint_type": kind, "side": side, "valid_count": len(group),
                          "correct_count": sum(r["pck_hit"] for r in group),
                          "pck": sum(r["pck_hit"] for r in group) / len(group),
                          "mean_error_px_224": float(np.mean([r["error_px_224"] for r in group])),
                          "mean_normalized_error": float(errors.mean()), "q1": float(q1),
                          "median": float(med), "q3": float(q3),
                          "lower_whisker": float(inside.min()), "upper_whisker": float(inside.max()),
                          "min": float(errors.min()), "max": float(errors.max()),
                          "outlier_count": int(len(errors) - len(inside))})
    output.mkdir(parents=True, exist_ok=overwrite)
    for name, rows in [("joint_summary.csv", stats), ("joint_observations.csv", raw)]:
        with (output/name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    person_rows = [{k:p[k] for k in ["sample_id", "original_sample_id", "capture_group", "evaluated_joints", "correct_joints", "pck_at_0.2_torso", "mean_pixel_error_224", "normalized_mean_error_torso"]} for p in persons]
    with (output/"sample_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(person_rows[0]));w.writeheader();w.writerows(person_rows)
    bundle = {"summary": summary, "joint_summary": stats, "joint_observations": raw, "sample_summary": person_rows,
              "threshold": .2, "source_metrics_sha256": identities,
              "definitions": {"PCK": "correct keypoints / valid keypoints; error <= 0.2 * torso scale",
                              "normalized_error": "Euclidean error in 224 model coordinates / per-person torso scale",
                              "boxplot": "median and Q1-Q3; whiskers at observations within 1.5 IQR; all data shown",
                              "sampling": "54 photos, 55 persons, correlated capture groups; descriptive distributions, not independent training repeats",
                              "GT": "RTMPose-assisted labels; human review confirmed by user; original stored coordinates retained",
                              "scope": "curated personal-photo simulation; test used for checkpoint selection"}}
    (output/"plot_data.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": .8,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out"})
    x=np.arange(6);width=.34
    fig, ax=plt.subplots(figsize=(8.4,4.6))
    for k,side in enumerate(["left","right"]):
        s=[v for v in stats if v['side']==side]
        pos=x+(k-.5)*width
        bars=ax.bar(pos,[v['pck']*100 for v in s],width,color=COLORS[k],label=side.title())
        for b,v in zip(bars,s):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.2,f"{v['pck']*100:.1f}",ha='center',va='bottom',fontsize=9)
            ax.text(b.get_x()+b.get_width()/2,3,f"n={v['valid_count']}",ha='center',color='white',fontsize=7.5)
    ax.axhline(summary['pck_at_0.2_torso']*100,color='#666666',ls='--',lw=1,label='Overall 83.33%')
    ax.set(ylim=(0,105),xticks=x,xticklabels=[k.title() for k in KINDS],ylabel='PCK@0.2 (%)')
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=9)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e5e5e5',lw=.6)
    fig.text(.02,.95,'a',fontweight='bold',fontsize=15)
    fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.86)
    for suffix in ['png','svg']:fig.savefig(output/f'figure_1_joint_pck.{suffix}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.8));rng=np.random.default_rng(42)
    all_errors=[]
    for k,side in enumerate(['left','right']):
        groups=[np.array([r['normalized_error'] for r in valid if r['joint']==side+'_'+kind]) for kind in KINDS]
        pos=x+(k-.5)*.36
        bp=ax.boxplot(groups,positions=pos,widths=.29,patch_artist=True,showfliers=False,
            medianprops={'color':'#202020','lw':1.4},whiskerprops={'lw':.9},capprops={'lw':.9})
        for b in bp['boxes']:b.set(facecolor=COLORS[k],edgecolor=COLORS[k],alpha=.5)
        for p,e in zip(pos,groups):
            ax.scatter(p+rng.uniform(-.07,.07,len(e)),e,s=10,color=COLORS[k],alpha=.45,linewidths=0,zorder=2)
            all_errors.extend(e)
    ax.axhline(.2,color='#666666',ls='--',lw=1)
    ax.text(5.55,.21,'PCK threshold',ha='right',va='bottom',fontsize=8,color='#555555')
    ax.set_yscale('log')
    ax.set(xticks=x,xticklabels=[k.title() for k in KINDS],xlim=(-.6,5.6),ylim=(min(all_errors)*.7,max(all_errors)*1.3),ylabel='Error / torso scale (log axis)')
    ax.set_yticks([.001,.01,.1,1]);ax.set_yticklabels(['0.001','0.01','0.1','1'])
    ax.legend(handles=[Patch(facecolor=COLORS[k],alpha=.6,label=s.title()) for k,s in enumerate(['left','right'])],frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01))
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e5e5e5',lw=.6)
    fig.text(.02,.95,'b',fontweight='bold',fontsize=15)
    fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.86)
    for suffix in ['png','svg']:fig.savefig(output/f'figure_2_joint_error_boxplot.{suffix}',dpi=300)
    plt.close(fig)
    guide="""# LightGen2 定量绘图说明

图1：按肩、肘、腕、髋、膝、踝分组，分别展示左右关节PCK@0.2。纵轴从0开始，虚线为总体83.33%；n是该关节有效标注点数。各关节分母略不同，总体应以545/654计算，不简单平均各柱。

图2：同样分组的躯干归一化定位误差箱形图。箱体为Q1–Q3，中央线为中位数，须为1.5IQR范围内的最远观测；散点显示全部有效观测，包括离群点。纵轴采用对数刻度，便于同时展示小误差和长尾值；表格数值未取对数，箱形统计量仍按原始归一化误差计算。虚线0.2就是PCK的判定阈值，越低越好。左右颜色在两图中一致。

joint_summary.csv是两图的汇总数据；joint_observations.csv保存55人体×12关节共660条记录（654条有效、6条无效），含像素误差、躯干尺度、归一化误差及命中标记。无效点为空，不当成0。sample_summary.csv为55个人体的指标。Excel给出汇总、箱形图宽表及完整逐点记录，适合直接交给画图同学。

这是当前交付的自摄照片仿真结果，与标准LSP、CCD实拍分别报告。GT为RTMPose辅助标注，人工校验由用户确认；本次不改标注和预测。测试集参与选模，结果按开发指标报告。散点表示单个人体关节观测，不能解释为多次独立训练；同组连拍存在相关性，本版不加入误差棒或显著性标记。

图注示例：LightGen2在自采人体姿态测试集上的逐关节性能。（a）左右肩、肘、腕、髋、膝、踝的PCK@0.2，总体545/654=83.33%。（b）以躯干尺度归一化的关键点定位误差分布。测试集包含54张照片、55个人体实例，654个有效四肢关节点。
"""
    (output/'README.md').write_text(guide,encoding='utf-8')
    (output/'PREVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>LightGen2 quantitative results</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}p{line-height:1.6}</style><h1>LightGen2 定量结果</h1><p>54张照片 / 55个人体 / 654个有效关节点。两图展示同一模型相对GT的逐关节结果。</p><img src="figure_1_joint_pck.png"><img src="figure_2_joint_error_boxplot.png"><p><a href="LightGen2_joint_plot_data.xlsx">绘图Excel</a> · <a href="README.md">绘图口径与图注</a></p>',encoding='utf-8')
    print(json.dumps({'hits':hits,'valid':len(valid),'joint_stats':stats},ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--overwrite',action='store_true',help='Refresh the generated quantitative files only')
    a=p.parse_args();export(a.delivery,a.output,a.overwrite)
