# 三张正式矩阵：汇报版

**汇报更正（2026-09-24）**：图中的 `full replay` 是历史文件/标题用语，实际只保留每个旧任务
512 个样本并循环回放，不是完整旧训练集。图中“音文/视频文”两项的文字候选表每个样本都相同；
当前实验检验音频/视频分类，尚未证明逐样本文字信息被使用。详情见
`ARCHITECTURE_DATA_AUDIT_20260924.md`。

三张图统一使用完整测试集 balanced accuracy、0–100% 色标、986×986 有效相位口径，电子读出均为
单层 `Linear(784, C)`。深色表示准确率更高；加框单元是各阶段新学习任务或独立模型的同任务结果。

## 一页总览

- 中文：`figures/three_formal_matrices_zh.{png,pdf,svg}`
- English: `figures/three_formal_matrices.{png,pdf,svg}`

## 单图

1. MoE full replay：`figures/matrix_1_moe_full_replay_zh.{png,pdf,svg}`
2. 独立 D2NN 纯推理 4×4：`figures/matrix_2_frozen_d2nn_cross_modal_zh.{png,pdf,svg}`
3. 可重构 D2NN、无 replay：`figures/matrix_3_d2nn_no_replay_zh.{png,pdf,svg}`

对应英文文件去掉文件名末尾的 `_zh`。

## 汇报时的严谨结论

MoE 的十个下三角单元平均为 72.06%，无 replay D2NN 为 64.58%，提高 7.49 个百分点；最终四任务
平均为 70.07%，比无 replay D2NN 的 62.44% 高 7.63 个百分点。最终阶段 MoE 在 EuroSAT、Speech、
Physical 上更高，在 CLEVR 上低 3.36 个百分点。因此结果支持“MoE 总体终身学习性能和多数任务保持
更好”，不支持“MoE 每个单元都优于 D2NN”。

独立 D2NN 4×4 矩阵没有进行逐单元微调：每行固定一个任务独立训练出的光学权重，每列接目标任务
原本训练好的单层 Linear，直接推理目标测试集。该矩阵用于展示不可重构光学权重的跨模态适配不稳定，
而不是声称所有异源组合都会完全失效。
