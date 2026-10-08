# Baseline 绘图数据包（2026-09-22）

这个目录是给绘图同学的最小交付：先看 `summary.csv`，需要散点、箱线、分组或失败案例时再进入相应任务目录。每个任务最多只有 `data.csv`、`metrics.csv`、`SOURCE.json`、`source_report.json` 和可选 `samples/`。

| 任务 | 主指标 | 数值 | 样本数 | 状态 |
|---|---:|---:|---:|---|
| LGVQ temporal | SRCC | 0.766343 | 558 | complete |
| LGVQ spatial | SRCC | 0.690773 | 558 | complete |
| ABO image-to-text | R@1 | 0.735833 | 2400 | complete |
| ABO image-to-image | R@1 | 0.851250 | 800 | complete |
| ABO text-to-image | Hit@1 | 0.820000 | 100 | complete; latency not measured |
| LSP keypoint detection | PCK@0.2 torso | 0.722571 | 1000 | aggregate exact; per-joint source differs by 0.001 |
| SALICON saliency | CC | 0.874838 | 5000 | complete; only CC retained per image |
| OpenMoji semantic interaction | changed-cell accuracy | 0.812000 | 1000 | complete; exact-checkpoint latency not measured |

## 列与单位

- 比例、相关系数、Recall、Hit、MRR、mAP、IoU、F1 均为 0–1 的无量纲小数，绘图时需要百分数可乘 100。
- LGVQ `target_mos`、`prediction`、`*_error_mos` 是原始 MOS 分数，不是百分比。
- 所有 `*_ms` 是毫秒；LSP 坐标及误差是 224×224 输入平面上的像素。
- `timing_measured=false` 表示该样本没有进入独立的 200 条计时子集，不能以均值回填单点。
- 样本级文件保留代码实际落盘的指标；缺失项不从汇总值反推，也不造数据。

## 两个容易混淆的版本

1. OpenMoji 0.8120 对应 layered-scene baseline、epoch 30、checkpoint SHA256 `0be775...cf8`；当前混合分支中的旧 `qwen_shared_s73` 是 0.8420，不能互换。
2. ABO 文搜图 0.8200 是独立的“100 标题 query → 2400 TEST 图片 gallery”冻结 Qwen 2048D run，源码固定在 Git commit `1e218991a`，不是把图搜文矩阵转置。

## 已知边界

- LSP 请求值 0.722571 来自后续 A100 完整汇总；服务器保留的逐关节 CSV 属于同一 epoch-37 teacher 的较早评测，重算为 0.721571。两者在 `lsp/SOURCE.json` 明确分开。
- SALICON 的正式 run 只保留逐图 CC；KLD、SIM、NSS、AUC-Judd、MAE 只有完整 5000 图汇总。
- 文搜图 0.8200 与 OpenMoji 0.8120 的精确 run 没有单次推理计时，不能借用其他协议的延迟。
