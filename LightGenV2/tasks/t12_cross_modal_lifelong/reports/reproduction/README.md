# T12 复现状态

> **2026-09-21 数据合同更正：本页以下数值全部是历史子集诊断，不是正式结果。**
> SEN12MS 原始包的许可文本并非 CC BY 4.0，现已从活动协议移除并由完整 Kather2016
> 替代。CLEVR、SONYC 和 Physical Concepts 的历史 run 也没有使用各自全集。活动协议、
> 精确全集计数和正式入口以任务根目录 [README](../../README.md) 为准。旧 run 为审计保留，
> 不进入新主表或结论。

当前代码实现已建立。2026-09-20 的三任务 CPU smoke 在旧 12 槽几何上完成，只作开发
记录，不能作为四任务结果。

四任务 16 槽 GPU smoke 已在 commit `e99846b8590550f2b02d11e8630e7cf12ebb9fab`
通过，服务器记录为 `runs/smoke/real_four_data_gpu_s17_v1`。它使用 RTX 4090 和下列真实
固定数据包：

- SEN12MS：2,048 / 512 / 768；
- CLEVR：6,000 / 750 / 750；
- SONYC-UST：5,712 / 2,072 / 216；
- Physical Concepts continuity：2,892 / 528 / 676，数据 manifest SHA256 为
  `d8df84711c58477b7974356a6356ce9f0217ef10ae2b6c6fe045ecc9d8873361`。

smoke 对 MoE 和 D2NN 的四任务各执行一次真实光学前向、损失和反向，所有应训练参数均有
有限梯度；它只验证计算合同，不报告分类性能。首轮 30 epoch 正式仿真已使用同一 commit、
同一数据和 `configs/initial_s17.json` 完成，run ID 为
`runs/simulation/initial_s17_v1`；性能见下表。

初始 run 的第二阶段检查发现：旧专家和旧读出头虽保持逐位不变，router 却会把旧任务
样本改送到后来专家，造成额外遗忘。commit
`bb3e49d5f` 增加任务容量保护：新任务仍可复用全部旧专家，旧任务 replay/评估只使用其
学习时已开放的前 4/8/12/16 槽。服务器的改进 run ID 为
`runs/simulation/capacity_guard_s17_v1`，只重训 ours。

## 2026-09-20 历史初始结果（已退出主协议）

`runs/simulation/initial_s17_v1` 已完成。联合 D2NN 在第 26 epoch 取得最佳验证均值，
测试主指标平均值为 0.442562；初始 lifelong MoE 的测试主指标平均值为 0.407443，未超过
baseline。各任务测试主指标如下：

联合 D2NN 同时访问四个任务，不属于终身学习。它从 2026-09-21 起不再进入活动训练入口、
主表或结论；既有 run 按审计规则保留，不删除服务器证据。

| 模型 | SEN12MS macro-F1 | CLEVR balanced accuracy | SONYC macro-AP | Video balanced accuracy | 平均值 |
|---|---:|---:|---:|---:|---:|
| Joint D2NN | 0.239913 | 0.505333 | 0.525002 | 0.500000 | 0.442562 |
| Sequential D2NN + replay | 0.159180 | 0.500000 | 0.426577 | 0.500000 | 0.396439 |
| Initial lifelong MoE | 0.227007 | 0.474667 | 0.428097 | 0.500000 | 0.407443 |
| Capacity-guard lifelong MoE | 0.242133 | 0.500000 | 0.435211 | 0.500000 | 0.419336 |

Initial MoE 的验证分数在各任务刚学完时为 0.356268 / 0.500000 / 0.346617 /
0.500000；最终前三项变为 0.217750 / 0.498667 / 0.278060，平均 backward transfer
为 -0.069469。该 run 证明四类真实数据可以通过同一光学传播和电子 MLP 完成端到端训练，
但没有证明有效的跨模态兼容性或优于 D2NN 的终身学习能力；CLEVR 和视频仍接近机会水平。

改进 run `capacity_guard_s17_v1` 已完成。容量保护把第二阶段的 SEN12MS 从初始 run 的
0.242678 提高到 0.351802；最终测试均值从 0.407443 提高到 0.419336。平均 backward
transfer 从 -0.069469 改善到 -0.043792，但仍低于 Joint D2NN 0.023226，且 CLEVR/视频
仍为机会水平，因此不能据此声称总体超过离线 baseline。

## 2026-09-21 历史 replay 归因对照

为了判断终身学习表现来自 MoE 专家结构还是仅来自 replay，commit `0e791798f` 增加
Sequential D2NN control。最终比较固定为：

1. Joint D2NN：四任务离线联合训练；
2. Sequential D2NN：相同任务顺序、每任务 30 epoch、512 条/旧任务 replay、50% 当前
   任务损失、相同任务 MLP 与验证选模；共享两层相位持续更新，旧任务头冻结；
3. Sequential Optical MoE：与第 2 项相同的顺序/replay/读出合同，另有固定槽位扩展、
   旧专家冻结和每个新专家组 3 epoch warmup。

Sequential D2NN 没有新专家，因此不执行 expert warmup；报告训练时间和更新次数时必须单列。
正式 run ID 为 `runs/simulation/sequential_d2nn_replay_s17_v1`，源码 commit
`0e791798f`，在 `capacity_guard_s17_v1` 完成后使用同一 RTX 4090 运行并已正常完成。

旧统一比较由 commit `06bad5e01` 的脚本从三个 `results.json` 自动生成。该比较器已经退出
活动源码；输入文件 SHA256 仍保留用于历史审计：

- Joint D2NN：`15b9fc75311f03c091a391589cf8d03cf04b20027b2a9836f6ff3b5894282f50`；
- Sequential D2NN：`9c7db5d51903a4cfa62622546bac09e0a793937035de2162659d0b6dc1f372ec`；
- Capacity-guard lifelong MoE：`57b797979111f9241dec786ae292e948138a9b129b8a5fab6c68b017af2d0570`。

Sequential D2NN 的测试主指标平均值为 0.396439。其各任务刚学完时的验证主指标为
0.364104 / 0.500000 / 0.342861 / 0.500000，训练全部结束后为
0.310318 / 0.500000 / 0.359289 / 0.500000，平均 backward transfer 为 -0.012453。
各阶段都验证了旧任务 MLP 参数逐位不变，而共享 first/global phase 确实发生更新。

Capacity-guard lifelong MoE 比 Sequential D2NN 的测试均值高 0.022896；逐任务差值为
+0.082952 / 0.000000 / +0.008633 / 0.000000。这是当前最直接支持“可扩展专家结构比同协议
顺序 D2NN 更适合终身学习”的结果。它的平均 backward transfer 为 -0.043792，反而差于
Sequential D2NN 的 -0.012453，因此现有证据只支持最终任务集合上的平均性能优势，尚不支持
“遗忘更少”。MoE 同时比离线 Joint D2NN 低 0.023226；CLEVR 和视频在三个模型中都接近
机会水平。下一轮应先修复这两个任务的模态编码/监督信号，再重复三模型对照，不能仅增加 epoch
后宣称架构优越。

## 2026-09-21 新活动协议

活动协议改为：Single-task D2NN 可学性上限与冻结光学 MLP probe、Sequential D2NN 无 replay、
Sequential D2NN 同量 replay、Sequential Optical MoE 同量 replay。三个顺序模型在每个任务
结束后保存所有已学任务的 validation/test 指标，形成下三角 `continual_matrix.json`。

64 样本 overfit 仅保留为实现诊断。正式训练必须使用 Kather2016 全部 5,000 张图、CLEVR
全部 85,000 张带 scene graph 图像、SONYC CSV 中全部 18,510 段录音，以及 Physical
Concepts 全部 5,000 个 continuity quadruplet。新的顺序结果只能在全集 manifest 校验和四个
单任务准入完成后产生。
