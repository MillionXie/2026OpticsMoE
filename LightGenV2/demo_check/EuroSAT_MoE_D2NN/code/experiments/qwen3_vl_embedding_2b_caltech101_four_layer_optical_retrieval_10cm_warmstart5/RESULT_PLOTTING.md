# 5% 模型结果整理与论文绘图

本入口只读取磁盘上真实存在的仿真、CCD 和微调结果，不生成或补齐任何实验指标。尚未采集 CCD 时，固定仿真测试仍可直接绘图；所有硬件项会在 `results_report.json` 中明确标记为 `false/unavailable`。

## 一条命令生成报告

在实验室压缩包解压后的根目录执行：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report --root . --output-dir result_report
```

论文正式导出必须加 `--require-arial`；如果系统未安装 Arial，命令会直接失败，避免
回退字体被误当成正式图。调试预览可以不加该参数，实际回退字体仍会写入 QA：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report --root . --output-dir result_report --require-arial
```

默认自动读取：

- `reference/training_evidence/stage_b/metrics/evaluation_summary.json`：固定仿真测试；
- `payload/quick210/04_language_global/offline_results/metrics.json`：末层 CCD 离线微调结果（若存在）；
- `payload/quick210/04_language_global/offline_results/pre_finetune_metrics.json` 与
  `post_finetune_metrics.json`：同一固定 test split 的微调前/后指标；
- `payload/quick210/04_language_global/offline_results/predictions.csv`：每个 test query
  两行、按 manifest key 配对的 pre/post 预测、rank 与 similarity margin；
- `payload/quick210/**/ccd_captured/*.png`：CCD 帧质控（若存在）。

若四层实验保存在另一个 session，可重复传入 `--session-dir`：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report `
  --root . `
  --session-dir payload/quick210 `
  --session-dir D:\qwen_lab\four_layer_run1 `
  --output-dir result_report
```

在服务器或源码工程中，若固定测试 JSON 不在解压包结构内，可显式指定：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report \
  --root . \
  --baseline-json experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/metrics/evaluation_summary.json \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four_layer_run1 \
  --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/result_report
```

## 输出契约

每个可用图默认同时生成：

- SVG：文字保持可编辑；
- PDF：TrueType 文字嵌入；
- 600 dpi PNG；
- 600 dpi、LZW 压缩 TIFF。

字体优先 Arial；系统没有 Arial 时回退到 Liberation Sans 或 DejaVu Sans，并把实际字体写入 `results_report.json` 和 `QA_REPORT.md`。单栏图宽 89 mm、高 55–60 mm；overview 为 183 × 100 mm；正文标称 7 pt。

图和数据包括：

1. `fig01_overall_metrics`：Top-1、Top-3、MRR；
2. `fig02_per_class_top1`：10 类 Top-1；
3. `fig03_confusion_matrix`：行归一化底色与原始计数；
4. `fig04_stage_progression`：有真实逐层结果时生成；
5. `fig05_ccd_quality_control`：有 CCD 时生成均值、P01–P99、近黑及近饱和比例；
6. `fig06_paired_query_changes`：完成 quick210 离线微调后，由自动保存的同 key
   pre/post 逐 query CSV 生成；
7. `fig07_overview`：论文选图用 overview；缺失硬件数据的面板会写明 unavailable。

`source_data/` 保存与每张图一一对应的 CSV/JSON；`figure_manifest.json` 和 `figure_manifest.csv` 保存文件 SHA-256；`QA_REPORT.md` 保存字体、分辨率、可编辑文字和缺失数据检查。请把这些 source data 与最终论文图一并移交。

图例会从每个真实 metric 文件动态写入 query、gallery 和 class 数量，CCD 图例会动态写入各 stage 的实际帧数。当前 baseline 是单一固定 checkpoint 在一次 sealed split 上的描述性结果，不存在 seed/fold 重复，因此不绘制会误导读者的误差条；若以后补跑多 seed，应以独立统计文件扩展，而不是从 200 个 query 人为构造“模型重复”。

## 可识别的真实产物

- 仿真/硬件检索指标：`top1_retrieval_accuracy`、`top3_retrieval_accuracy`、`mrr`、`per_sku` 或 `per_class`、`confusion_matrix`；
- quick210：`04_language_global/offline_results/metrics.json`；
- 四层：`01_vision_expert` 至 `04_language_global` 下的 `finetune_metrics.json`；
- CCD：各 stage 的 `ccd_captured/*.png`；
- quick210 逐样本 CSV：`04_language_global/offline_results/predictions.csv`。离线微调
  会为每个固定 test query 自动保存两行，`system=quick210_0_pre_finetune` 和
  `quick210_1_post_finetune`，并包含 `sample_id`、精确 manifest `key`、真实/预测标签、
  `top1_correct`、`similarity_margin` 与 `rank`。因此运行过新版离线微调后 `fig06`
  可直接生成；未运行时仍标记 unavailable，不能由 aggregate 指标反推。

当前正式仿真结果的 5% 是融合系数下限，不是实测“光能占比”。报告会保留这个语义说明，避免论文中误写。

## 快速生成固定仿真预览

打包前可把服务器上的真实 stage-B JSON 预渲染到待打包目录：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report \
  --root STAGING_BUNDLE_ROOT \
  --baseline-json STAGE_B_RUN/metrics/evaluation_summary.json \
  --output-dir STAGING_BUNDLE_ROOT/reference/fixed_simulation_report
```

该目录可直接装入实验室 ZIP；其中报告仍记录原始 JSON 的 SHA-256，不依赖 Qwen 或 Torch。
