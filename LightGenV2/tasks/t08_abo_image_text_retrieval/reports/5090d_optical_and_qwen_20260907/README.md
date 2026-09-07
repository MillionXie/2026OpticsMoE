# ABO 强均衡光学 MoE 与冻结 Qwen：RTX 5090 D 正式对照

## 任务和性能

本任务是图搜文：每张 ABO test 商品图作为 query，在 100 个固定官方英文标题中检索唯一正确标题。完整 test 有 2,400 张图；训练集 4,800 张图只用于光学 MoE。冻结 Qwen baseline 不使用训练集。

| 方法 | R@1 | R@5 | R@10 | MRR | 时间 | 平均功率 | 能量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 光 Router Top-2 MoE，强均衡 | **0.7983** | **0.9538** | **0.9858** | **0.8676** | **9.374 ms/query** | **109.37 W**（组合代理） | **1.025 J/query**（组合代理） |
| 冻结 Qwen3-VL-Embedding-2B | 0.7371 | 0.9338 | 0.9604 | 0.8230 | 25.691 ms/query | 154.20 W（实测） | 3.962 J/query（实测） |

在当前合同下，光学 MoE 的 R@1 高 6.125 个百分点；Qwen/光学 MoE 的模型核心时间比为 2.74×，实测 Qwen 能量/光学组合能量代理为 3.86×。

## 修正后的光学时间边界

上一版的 10.138 ms 把“下一层 SLM 场重建”和 Router 电子后处理也串进了主路径，超出了本次需要的统计口径。本版只统计：

1. 每个特征 block 的 `CCD → 非负截断 → 单帧均值归一化 → 相对强度截断 → log1p → AdaptiveAvgPool → LayerNorm → ReLU → Linear → 同尺度融合 → block 输出`；
2. 四个特征 block 结束后的一次最终 64D 检索头。

并行电子残差、下一层 SLM 重建以及 Router 的电子后处理均不计入该电子开销。

RTX 5090 D 每项预热 50 次、正式同步计时 1,000 次。中位数为：

| 分量 | 中位数 | P95 | 是否计入 |
| --- | ---: | ---: | --- |
| Vision：CCD 到 block 输出 | 0.3530 ms | 0.3673 ms | 是，2 个 block |
| Language：CCD 到 block 输出 | 0.3525 ms | 0.3647 ms | 是，2 个 block |
| Vision/Language 平均 | **0.3527 ms/block** | — | — |
| 最终 64D 检索头 | **0.0791 ms/query** | 0.0818 ms | 是，1 次 |
| Vision 并行电子残差 | 0.2935 ms | 0.3085 ms | 否，由光路覆盖 |
| Language 并行电子残差 | 0.2773 ms | 0.2883 ms | 否，由光路覆盖 |

因此，四个特征 block 的 CCD 后电子处理共 `2×0.3530 + 2×0.3525 = 1.4109 ms`；加最终读出头后，需要串行计入的电子时间为 **1.4900 ms/query**。

本报告按明确的器件分解值计算六次物理光场：

`6 × (0.714 ms 传播 + 0.100 ms 相位 SLM + 0.500 ms CCD) = 7.884 ms`

所以修正后的完整主路径为：

`7.884 + 1.4109 + 0.0791 = 9.3740 ms/query`

若表格继续沿用此前给定的“六层光路合计 9.084 ms”实测口径，则不要再使用 7.884 ms；对应总时间应写为 `9.084 + 1.4900 = 10.5740 ms/query`。这两个数来自不同的物理光路口径，不能相加或混用。

## 功率与能量口径

光学设备功率使用给定值 80.388 W；GPU 组件功率来自 RTX 5090 D 板卡采样。按修正后的 9.374 ms 主路径：

- 光学设备持续上电能量：0.7536 J/query；
- 被计入的 CCD 后处理和任务头 active GPU 能量：0.1462 J/query；
- 物理光传播期间按 GPU idle 15.92 W 补入后，GPU 板卡能量：0.2717 J/query；
- 光学设备与 GPU 板卡合计：**1.0252 J/query**，折合平均功率 **109.37 W**；
- 若极端假设 RTX 5090 D 在完整 9.374 ms 内始终达到额定 575 W，则绝对上界为 6.1436 J/query、655.388 W。它不是实测平均值。

冻结 Qwen 为直接板卡测量：50 次预热，类别均衡 200 张计时/功率样本；CUDA mean/median/P95 为 25.691/25.075/27.657 ms，active mean/peak 为 154.20/158.05 W，实测 active energy 为 3.962 J/query，575 W 额定上界为 14.772 J/query。

## Baseline 合同

baseline 使用原始 `Qwen3-VL-Embedding-2B`，2,127,532,032 个参数全部冻结，可训练参数为 0；没有 LoRA、额外读出头或微调。100 个标题 embedding 预先计算；在线查询执行原生 Vision/Language blocks、2048D L2 归一化、100 个余弦相似度和完整排序。性能在全部 2,400 张 test 图上计算。

## 证据文件

- `summary.json`、`comparison_summary.csv`：可直接填表的汇总；
- `abo_5090d_comparison.png/.pdf`：性能、时间、能量三联图；
- `optical_moe/`：每个电子分量 1,000 次计时、active 功率样本和原始 JSON/CSV；
- `qwen_baseline/`：2,400 条预测、200 条计时、功率样本和总报告；
- `evidence_manifest.json`：关键文件 SHA256。

重新绘图：

```powershell
python LightGenV2\tasks\t08_abo_image_text_retrieval\reports\5090d_optical_and_qwen_20260907\plot.py
```
