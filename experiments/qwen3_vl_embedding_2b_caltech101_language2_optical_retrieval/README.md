# Caltech101 Language Block-2 MoE4 Optical Residual

本实验从已训练的 Vision-2D、no-DeepStack 电子 checkpoint 初始化。Vision 和
Language 电子主路默认冻结；Language Mixer 第 2 个 block 内加入一个并联光学残差。

光学支路保留旧 Grocery 光路的 MoE4：`2×2` 个 `224×224` 专家、pitch `254`、
top-k `2` router、逻辑有效区 `478×478`、FFT canvas `518×518`。router 正常参与
前向和检索梯度训练，只关闭 router balance/importance 两项辅助 loss。

流程为：`192→224` 非负场编码 → MoE4 router → 专家相位/传播/OEO → 全局相位 →
CCD `478×478` → 稳健强度归一化 → 面积池化到 `224×224` → `224→192` → 与电子
Block 2 输出按可学习 gate 相加。硬件快速接入只替换“全局相位输入到最终 CCD”这段；
前面的专家级先用仿真生成。

物理接口和文件约定见 `HARDWARE_BRIDGE.md`，命令见 `RUN_COMMANDS.md`。
