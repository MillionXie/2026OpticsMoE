# Caltech101 Language Block-2 MoE4 Optical Residual

本实验从已训练的 Vision-2D、no-DeepStack 电子 checkpoint 初始化。Vision 和
Language 电子主路默认冻结，并增加一条跨两个 Language Mixer Block 的连续光学支路。

光学支路保留旧 Grocery 光路的 MoE4：`2×2` 个 `224×224` 专家、pitch `254`、
top-k `2` router、逻辑有效区 `478×478`、FFT canvas `518×518`。router 正常参与
前向和检索梯度训练，只关闭 router balance/importance 两项辅助 loss。

Language Block 1 对应 `192→224` 非负场编码、MoE4 router、专家相位、传播和 OEO；
其光场继续送入 Language Block 2 的 global phase、传播和 `478×478` CCD。CCD 再经
稳健归一化、池化到 `224×224`、`224→192`，最后与电子 Block 2 输出按可学习 gate
相加。硬件快速接入保留 Block 1 正常仿真前向，只替换 Block 2 的
“global phase 输入→最终 CCD”部分。

物理接口和文件约定见 `HARDWARE_BRIDGE.md`，命令见 `RUN_COMMANDS.md`。
