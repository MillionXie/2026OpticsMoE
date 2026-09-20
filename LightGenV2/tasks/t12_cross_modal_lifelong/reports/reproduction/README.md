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
