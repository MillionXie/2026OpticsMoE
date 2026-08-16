# Caltech101 Language Block-2 Optical Residual

该实验从已训练的 Vision-2D、no-DeepStack 电子 checkpoint 初始化。Vision 和
Language 电子主路默认冻结，只在 Language Mixer 第 2 个 block 内增加单相位面
光学残差。没有 MoE、router 或 router loss。

光路输入是 Language Block 2 的 192 维 token，经过 `192 -> 224` 非负振幅编码，
形成 `224 x 224` SLM 输入。仿真传播在 `518 x 518` padding canvas 上执行，CCD
读取中心 `224 x 224` 区域。CCD 后使用与实测完全相同的增益/背景稳健归一化。

物理光路接入及文件约定见 `HARDWARE_BRIDGE.md`，所有命令见 `RUN_COMMANDS.md`。

