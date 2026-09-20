# T12 复现状态

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
有限梯度；它只验证计算合同，不报告分类性能。首轮 30 epoch 正式仿真正在同一 commit、
同一数据和 `configs/initial_s17.json` 下运行，run ID 为
`runs/simulation/initial_s17_v1`。完成并通过复算前不填写性能数字。

初始 run 的第二阶段检查发现：旧专家和旧读出头虽保持逐位不变，router 却会把旧任务
样本改送到后来专家，造成额外遗忘。commit
`bb3e49d5f` 增加任务容量保护：新任务仍可复用全部旧专家，旧任务 replay/评估只使用其
学习时已开放的前 4/8/12/16 槽。服务器的改进 run ID 为
`runs/simulation/capacity_guard_s17_v1`，只重训 ours。

## 2026-09-20 初始四任务结果

`runs/simulation/initial_s17_v1` 已完成。联合 D2NN 在第 26 epoch 取得最佳验证均值，
测试主指标平均值为 0.442562；初始 lifelong MoE 的测试主指标平均值为 0.407443，未超过
baseline。各任务测试主指标如下：

| 模型 | SEN12MS macro-F1 | CLEVR balanced accuracy | SONYC macro-AP | Video balanced accuracy | 平均值 |
|---|---:|---:|---:|---:|---:|
| Joint D2NN | 0.239913 | 0.505333 | 0.525002 | 0.500000 | 0.442562 |
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

## 三模型归因对照

为了判断终身学习表现来自 MoE 专家结构还是仅来自 replay，commit `0e791798f` 增加
Sequential D2NN control。最终比较固定为：

1. Joint D2NN：四任务离线联合训练；
2. Sequential D2NN：相同任务顺序、每任务 30 epoch、512 条/旧任务 replay、50% 当前
   任务损失、相同任务 MLP 与验证选模；共享两层相位持续更新，旧任务头冻结；
3. Sequential Optical MoE：与第 2 项相同的顺序/replay/读出合同，另有固定槽位扩展、
   旧专家冻结和每个新专家组 3 epoch warmup。

Sequential D2NN 没有新专家，因此不执行 expert warmup；报告训练时间和更新次数时必须单列。
正式 run ID 为 `runs/simulation/sequential_d2nn_replay_s17_v1`，源码 commit
`0e791798f`，已在 `capacity_guard_s17_v1` 完成后使用同一 GPU 启动。完成前不填写性能数字。
