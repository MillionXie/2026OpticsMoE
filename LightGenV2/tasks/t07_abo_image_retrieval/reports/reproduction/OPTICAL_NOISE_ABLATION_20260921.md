# ABO 图搜图 83.125%：同权重光学噪声消融

日期：2026-09-21。

## 固定对象

- 协议：`abo200_enrolled_sku_hash8train4query_v1`。
- 任务：200 个已登记 SKU 的不同视角图搜图；1600 张 gallery、800 张 query，每个 query 有 8 个同 SKU 正例。
- 正式 checkpoint SHA256：`c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0`。
- 所有条件使用同一权重、同一 query/gallery、同一 64 维描述子与同一检索规则；没有针对消融重新训练。
- 随机噪声种子：42。

## 消融定义

选中光学层的输出先替换为零均值、单位方差高斯噪声，随后继续经过模型原有的**逐样本 RMS 尺度匹配**。因此：

- 保留原融合 alpha 和噪声注入后的光支路 RMS；
- 保留光 Router、Top-2 选择和六次捕获的控制流程；
- 只破坏选中光学特征输出所携带的样本信息；
- 不把“噪声振幅变小”混作性能下降原因。

模型的特征光学层顺序为 `V1 -> V2 -> L1 -> L2`。本报告中的“最后一层”严格指最后的 `L2 / language global`，不是任意挑选的一层。

`remove_optical` 则跳过全部光学特征支路，让同一 checkpoint 的电子直通独立推理。它不是独立训练的纯电子模型。

## 完整 800-query 结果

| 条件 | 被替换的光学特征层 | Hit@1 | 命中数 | 相对正常光 |
|---|---|---:|---:|---:|
| 正常光电 | 无 | **83.125%** | 665/800 | — |
| 仅最后一层换噪声 | L2 | **41.375%** | 331/800 | -41.750 pp |
| 每个模态的全局层换噪声 | V2、L2 | **32.875%** | 263/800 | -50.250 pp |
| 所有特征光学层换噪声 | V1、V2、L1、L2 | **17.250%** | 138/800 | -65.875 pp |
| 去光、保留电子直通 | 全部光支路关闭 | **76.375%** | 611/800 | -6.750 pp |

补充指标：

| 条件 | Hit@5 | Hit@10 | mAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| 正常光电 | .92000 | .95125 | .730395 | .788058 |
| 仅 L2 噪声 | .70250 | .81125 | .212659 | .329047 |
| V2、L2 噪声 | .64000 | .75500 | .151665 | .257510 |
| 四层全噪声 | .40375 | .54125 | .062926 | .127858 |
| 去光 | .90500 | .94000 | .530625 | .628314 |

## 解释边界

结果支持两点：

1. 光学特征确实携带对正式检索有用的信息；去光下降 6.75 个百分点。
2. 在保留约 0.445 融合系数时，把有信息光特征换成同尺度噪声会形成强干扰，因此全噪声远差于干净去光。这不是“光贡献等于 65.875 个百分点”，也不能把 alpha 直接解释为贡献百分比。

本次是固定 seed=42 的确定性消融。它足以复现这套权重在指定噪声下的结果，但不是多随机种子的均值或置信区间。Router 本身没有被随机化；四层全噪声指四个**特征光学层**均被替换，以保持原捕获/路由预算。

## 复现命令

在仓库根目录执行，`RELEASE` 指向 `abo_i2i_83125_20260917` 独立交付目录：

```bash
RELEASE=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/releases/abo_i2i_83125_20260917
OUT=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/analysis/abo_i2i_83125_optical_noise_seed42_20260921

CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen optical \
  --assets "$RELEASE/assets" --data "$RELEASE/data" \
  --manifest "$RELEASE/protocol.json" --checkpoint "$RELEASE/assets/best.pt" \
  --expected-checkpoint-sha256 c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0 \
  --batch-size 4 --device cuda --output "$OUT" \
  --optical-noise-ablations --noise-seed 42
```

正式证据在服务器的上述 `OUT/final_report.json`；五份 `*_predictions.csv` 保存全部 800 条逐 query 预测。
