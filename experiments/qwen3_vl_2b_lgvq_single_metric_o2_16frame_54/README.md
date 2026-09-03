# Qwen3-VL front + 16-frame O2 LGVQ

本工程实现两个互相独立的单指标视频质量模型：Spatial 与 Temporal。它保留
Qwen3-VL-2B-Instruct 的官方图像处理、冻结 patch/position embedding，以及文本
chat template、tokenizer、冻结词 embedding；学生推理网络不含 Transformer 或
Attention。16 帧在 478 有效光场中按 4×4 并行，Vision 单专家为 54×54，Router
固定为光学 Top-2，随后经过四层光电凸融合和目标专属电子读出头输出一个连续 MOS。

- 完整结构与每一步维度：[ARCHITECTURE.md](ARCHITECTURE.md)
- 唯一执行顺序：[RUN_COMMANDS.md](RUN_COMMANDS.md)
