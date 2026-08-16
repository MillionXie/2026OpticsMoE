# Caltech101 Language Two-Block MoE4 Dual Fusion

本实验从 Vision-2D、no-DeepStack 电子 checkpoint 初始化，占用两个连续的 Qwen
Language replacement slot。保留旧 Grocery MoE4：`2×2` 个 `224×224` 专家、pitch
`254`、top-k `2` router、有效区 `478×478`、FFT canvas `518×518`。router 参与
检索梯度训练，仅关闭 balance/importance 辅助 loss。

两个 Block 使用同构但参数独立的 CCD 后处理：原始非负强度 `478×478` → 每帧均值尺度归一化
（不做背景扣除）→ clip/log1p → pool `478→224` → row LayerNorm → `224→192`。

Language Block 1：电子 Mixer 1 与 MoE4 专家相位/传播并行。专家 CCD 经本层 readout
得到 `[B,T,192]`，再执行
`F1 = electronic1 + sigmoid(gate1) × expert_delta`。`F1` 被重新编码、按同一 router
写入 2×2 区域，成为 Block 2 global phase 的输入。

Language Block 2：Mixer 2 处理 `F1`；global phase/传播产生最终 CCD，经同构但独立的
readout 得到 `[B,T,192]`，执行
`F2 = LN(electronic2 + sigmoid(gate2) × global_delta)`，最后映射回 2048 hidden。

硬件快速接入会完整仿真 Block 1 和第一次光电融合，导出由 `F1` 重新编码得到的
global 输入。实际光路只替换 Block 2 的 global phase→CCD，随后微调 gate2 和下游
电子 readout。物理接口见 `HARDWARE_BRIDGE.md`，命令见 `RUN_COMMANDS.md`。
