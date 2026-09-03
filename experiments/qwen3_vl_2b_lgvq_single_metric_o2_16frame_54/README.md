# Qwen3-VL front + Spatial-4 / Temporal-16 O2 LGVQ

本工程实现两个互相独立的单指标视频质量模型：Spatial 与 Temporal。它保留
Qwen3-VL-2B-Instruct 的官方图像处理、冻结 patch/position embedding，以及文本
chat template、tokenizer、冻结词 embedding；学生推理网络不含 Transformer 或
Attention。Spatial 使用 4 帧、2×2 lane、109×109 Vision 专家；Temporal 使用
16 帧、4×4 lane、54×54 Vision 专家。Router 固定为光学 Top-2，随后经过四层
光电凸融合和目标专属电子读出头输出一个连续 MOS。训练时相干未调制功率比例在
20%–35% 随机扰动，正式仿真固定为 20%。

- 完整结构与每一步维度：[ARCHITECTURE.md](ARCHITECTURE.md)
- 唯一执行顺序：[RUN_COMMANDS.md](RUN_COMMANDS.md)
- 主服务器缓存、并行训练和监控：[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)
- 训练后六阶段相位可视化与硬件 BMP：[EXPORT_COMMANDS.md](EXPORT_COMMANDS.md)
- 每 5 epoch 相位快照格式与读取：[MASK_EVOLUTION.md](MASK_EVOLUTION.md)

真实 `Qwen3-VL-2B-Instruct` 权重的接口验证结果为：一帧 `448×448` 经官方
processor 得到 `grid_thw=[1,28,28]`；冻结 patch+position 前端输出
`[784,1024]`，无参数池化后为 `[49,1024]`；当前 Spatial prompt 加入完整 user / 
assistant chat template 后为 38 个 token，冻结词表输出 `[1,38,2048]`。这些维度
不是用假张量推测出来的。
