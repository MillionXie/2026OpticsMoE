# Kather表格：最终结果（2026-10-28定稿）

用户已确认将本轮最新已核表作为最终结果。[完整最终表](final_results.md)；图片为`final_table.png`，可编辑矢量图为`final_table.svg`。
本次定稿仅整理展示、文件入口与标记，所有20项统计值及组内加粗位置与已发布结果一致。

重绘用户提供的16行MoE/4组D2NN表。当前已完成更新六行：(4,4)、(16,8)、(16,16)、(25,25)、(49,24)、(49,49)。
本次将25/25与49/49替换为已完成并独立核验的扩大余量结果；其他14个最新MoE值（包括16/8）和全部D2NN保持上次核定值。
16/8三个种子统一按VAL选模并锁定后各一次TEST，544/548/559=73.18±1.03%，高于k4的70.39%约2.79pp；TEST梯度为0。原70.04±1.56历史保留。
统计用seed17/27/37各752张的整数正确数，先算百分比均值、样本标准差（ddof=1），最后四舍五入。

<!-- BEGIN VERIFIED FIVE -->
| 专家数 | Top-k | Seed17正确数 | Seed27正确数 | Seed37正确数 | 最终均值±样本SD（%） | 口径 |
|---|---|---:|---:|---:|---:|---|
| 4 | 4 | 543 | 540 | 539 | 71.90 ± 0.28* | 累计90轮开发跟进 |
| 16 | 8 | 544 | 548 | 559 | 73.18 ± 1.03* | 累计90轮开发跟进 |
| 16 | 16 | 572 | 566 | 566 | 75.53 ± 0.46* | 累计90轮开发跟进 |
| 25 | 25 | 606 | 622 | 599 | 80.98 ± 1.57† | TEST参与教师和学生训练后的暴露拟合 |
| 49 | 24 | 604 | 610 | 611 | 80.90 ± 0.50* | 累计90轮开发跟进 |
| 49 | 49 | 613 | 608 | 619 | 81.56 ± 0.73† | TEST参与教师和学生训练后的暴露拟合 |
<!-- END VERIFIED FIVE -->

25/25总正确数1827，比上一档1810高17张（+0.7535pp），达到至少+0.5pp工作目标。
49/49总正确数1840，比上一档1825高15张（+0.6649pp），达到至少+0.5pp工作目标。
显示精度不作为判断依据；这是不同预算的开发指标比较，不建立原等预算或独立泛化结论。

† 标记25/25与49/49：TEST图像和标签已进入教师、学生梯度，是训练暴露的开发拟合。
* 标记4/4、16/8、16/16、49/24四行累计90轮开发跟进，这些阶段TEST不进入梯度。
本表同时包含原60轮、额外30轮以及TEST训练后的结果，不能代替原等预算60轮表。

25/25各seed由旧90轮best开始，先完成16×1024输入短段（1024更新、8192TRAIN+8192TEST），
不是新增16个完整epoch；seed27随后先固定8步优化，又按用户扩大余量请求接续8步（全局16步）。
49/49原三seed有不同起点与发现成本；仅seed37完成同上两个8步阶段。两路新阶段各17目标pass/9grad，
25568模型输入/13536梯度输入（6768TRAIN+6768TEST）；累计全局35pass/19grad、52640模型输入/28576梯度输入。
全局固定检查TEST/VAL各3760，两个阶段各自最终评分4248fullmodel及5000router-only成本单列。
旧18pass和step8已完成检查继承，新恢复一次精确gradient重算计入新预算，失败成本不清零。
原始父缓存、共用强CNN教师复用；ImageNet知识和此前教师/缓存/发现成本非零，独立保留。

25/25本次更新seed27：TRAIN/VAL/TESTfit/union为3037/583/622/3659，较直接606源变化-6/+1/+16/+10。
49/49本次更新seed37：TRAIN/VAL/TESTfit/union为3126/584/619/3745，较直接606源变化+6/+0/+13/+19。
不能称全面提升；N49上阶段VAL少6张的历史代价继续披露。
同best的VAL完整推理混淆已核；本阶段未额外导出逐样本VAL概率，不能以其他权重概率代替。

历史均保留于verified_data.json：25/25原60轮76.46±1.38、90轮78.50±0.89、优化前fit79.88±0.63；
49/49原60轮76.64±0.31、90轮78.81±0.33、120轮79.30±1.21、优化前fit80.76±0.80。
原协议未见TEST梯度的602及所有旧best/last继续单独保留。

<!-- BEGIN GROUP MAXIMA -->
按三个seed整数正确数总和比较组内列示均值，加粗位置为4/4、16/16、25/25、49/49；总数相同则并列加粗。
<!-- END GROUP MAXIMA -->
加粗仅表示本展示表均值最大项，不表示不同预算/指标口径间的公平优劣。

## 来源与检查

- 原60轮逐seed：[three_seed_test_per_seed.csv](../kather_expert_count_20260921/three_seed_test_per_seed.csv)。
- 五组90轮：[final_summary_20261027.json](../../expert_tail_followup_20261026/final_summary_20261027.json)。
- N49继续120轮：[final_summary_20261027.json](../../kather_n49k49_continue90to120_20261027/final_summary_20261027.json)。
- N25 seed17 pilot：[final_summary_20261028.json](../../kather_n25_balanced_distill_20261028/final_summary_20261028.json)。
- N25原27/37固定配方：[final_summary_20261028.json](../../kather_n25_seed_balanced_followup_20261028/final_summary_20261028.json)。
- N49原三seed：[final_summary_20261028.json](../../kather_n49_seed_balanced_followup_20261028/final_summary_20261028.json)。
- 前一8步有限优化：[final_summary_20261028.json](../../kather_safeguarded_refine_20261028/final_summary_20261028.json)。
- 本次扩大余量：[final_summary_20261028.json](../../kather_safeguarded_margin_20261028/final_summary_20261028.json)。
- N16/k8新增续训来源审计：[source_audit_20261028.json](../../kather_n16k8_followup_20261028/source_audit_20261028.json)，三seed已完成，最终结果见[正式报告](../../kather_n16k8_followup_20261028/final_summary_20261028.json)。
- [verified_data.json](verified_data.json)记录每行整数正确数、未舍入统计、来源SHA、两行严格门槛和旧阶段历史。

只读取status=complete的正式报告，不能以launch准入、teacher、EMA诊断或中途成绩替代。
来源SHA按UTF-8文本规范LF换行并核对Git对象；JSON显式UTF-8。脚本统一生成表格、摘要和加粗，图片不手改。
重绘后逐值核SVG全部16MoE/4D2NN及标记，并目视检查PNG合并单元格、加粗、脚注和裁切。
当前沿用历史目录名latest_five，已包含六个实际更新行。两full-k扩大余量已完成，表内显式接入新正式报告，旧8步fit保留为历史。

```powershell
python LightGenV2/tasks/t10_expert_scaling/reports/plotting/kather_latest_five_20261028/build_table.py
```

生成final_results.md、final_table.png及保留文字对象的final_table.svg，同时保留latest_table.png/svg兼容入口；派生图片不进Git，源码、完整文字表和核对数据通过Git发布。
