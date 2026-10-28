# Kather表格：最新五组结果整理（2026-10-28）

本目录重新绘制用户提供的16行MoE/4组D2NN表。仅更新(4,4)、(16,16)、(25,25)、(49,24)、(49,49)五行；
未更新的MoE行和全部D2NN保持原表。统计使用seed17/27/37各752张的整数正确数，
先计算百分比均值和样本标准差（ddof=1），最后保留两位小数，不平均已经四舍五入的百分数。

<!-- BEGIN VERIFIED FIVE -->
| 专家数 | Top-k | Seed17正确数 | Seed27正确数 | Seed37正确数 | 最新均值±样本SD（%） | 口径 |
|---|---|---:|---:|---:|---:|---|
| 4 | 4 | 543 | 540 | 539 | 71.90 ± 0.28* | 累计90轮开发跟进 |
| 16 | 16 | 572 | 566 | 566 | 75.53 ± 0.46* | 累计90轮开发跟进 |
| 25 | 25 | 606 | 597 | 599 | 79.88 ± 0.63† | TEST参与教师和学生训练后的暴露拟合 |
| 49 | 24 | 604 | 610 | 611 | 80.90 ± 0.50* | 累计90轮开发跟进 |
| 49 | 49 | 613 | 608 | 601 | 80.76 ± 0.80† | TEST参与教师和学生训练后的暴露拟合 |
<!-- END VERIFIED FIVE -->

只有N25的27/37正式报告状态为complete、各自最终selected已审计后，才能执行重绘；
脚本用同一份核对数据同步生成此摘要与组内加粗说明，不能用中途检查点数值填表。

† 标记25/25与49/49：原TEST图像和标签均已进入教师及学生梯度，属于训练暴露的开发拟合，
不能解释为独立TEST泛化或原等预算Top-k结果。49/49仍为已定稿的80.76±0.80，
其seed17有不同适配起点及前序方法开发成本，因此该行仅作描述性三seed汇总。
* 仅标记4/4、16/16、49/24三行累计90轮开发跟进；这些阶段TEST不进入梯度。
本展示表同时包含原60轮、额外30轮及TEST训练后的结果，标题/行标记/脚注均明确这些区别。
若用于原等预算分析，应保留原60轮表，而不是用本展示表替代。

25/25每seed从各自旧90轮训练选出的best出发，再执行固定16×1024输入短段，
即1024次学生更新、8192原TRAIN+8192原TEST梯度输入；不是新增16个完整数据epoch，
不能写成累计106个完整epoch。每seed有独立父缓存和来源校准；共同强CNN教师直接复用，
本N25阶段没有重训教师，但教师包含额外ImageNet预训练知识和此前教师训练/缓存成本，
这些成本不是零，不能并入原等预算表。seed17取已完整交付pilot，27/37取完成后的冻结配方跟进。

25/25历史继续保留：原60轮76.46±1.38；旧累计90轮78.50±0.89，
后者正确数598/587/586。新暴露拟合替换展示值时，这些旧统计仍保存于`full25_history`。

49/49各阶段记录继续分别保留：原60轮76.64±0.31；累计90轮78.81±0.33；
累计120轮、TEST不进梯度的开发跟进79.30±1.21；最新TEST训练后的拟合80.76±0.80。

<!-- BEGIN GROUP MAXIMA -->
按三个seed整数正确数总和比较组内列示均值，加粗位置为4/4、16/16、25/12、49/24；总数相同则并列加粗。
<!-- END GROUP MAXIMA -->
这些加粗只标示本展示表数值最大项，不建立不同预算/指标口径间的公平优劣结论。

## 来源与检查

- 原表逐seed：[three_seed_test_per_seed.csv](../kather_expert_count_20260921/three_seed_test_per_seed.csv)。
- 五组累计90轮：[final_summary_20261027.json](../../expert_tail_followup_20261026/final_summary_20261027.json)。
- 49/49累计120轮：[final_summary_20261027.json](../../kather_n49k49_continue90to120_20261027/final_summary_20261027.json)。
- 49/49已定稿三seed：[final_summary_20261028.json](../../kather_n49_seed_balanced_followup_20261028/final_summary_20261028.json)。
- 25/25 seed17已完成pilot：[final_summary_20261028.json](../../kather_n25_balanced_distill_20261028/final_summary_20261028.json)，取`selected.TEST_fit_correct`及`selected.TEST_total`。
- 25/25 seed27/37最终跟进：[final_summary_20261028.json](../../kather_n25_seed_balanced_followup_20261028/final_summary_20261028.json)，只在顶层`status=complete`时，按`students[].seed`读取各自`selected.TEST_fit_correct`、`selected.TEST_total`。
- [verified_data.json](verified_data.json)含每行整数正确数、未舍入均值/样本SD、来源SHA256和检查结果。

来源UTF-8文本SHA按规范LF换行计算，并与发布Git对象的内容核对，避免Windows CRLF与服务器LF产生伪差异。
全部JSON显式按UTF-8读取；新N25两个正式来源均加入SHA清单，不以启动报告的旧准入值或中途history替代。

原表16个MoE值和4个D2NN值均从逐seed来源重算，与截图抄录值逐项一致；
五个更新配置保持不变；本轮仅替换25/25的最新统计和†标记，其余15个MoE值、4组D2NN值和49/49全部历史保持不变。
不使用VAL、教师准确率或单seed成绩替代三seed统计；不按四舍五入后的显示值决定加粗，整数总正确数相同则并列。
生成后核对SVG中全部16个MoE值、4个D2NN值及标记，并目视检查PNG中的合并单元格、加粗、脚注及裁切。

从仓库根目录执行：

```powershell
python LightGenV2/tasks/t10_expert_scaling/reports/plotting/kather_latest_five_20261028/build_table.py
```

生成`latest_table.png`及保留文字对象的`latest_table.svg`；派生图片不进入Git，源码与核对数据由Git发布。
