# CIFAR-10 纯电子 Qwen 检索

该工程将 Caltech10 的无教师紧凑电子结构原样迁移到 CIFAR-10，用于检查结论是否能跨数据集复现。

正式数据划分：官方训练集每类固定抽取 3 张作为 gallery，其余 49,970 张全部训练；官方 10,000 张测试集全部用于评测。gallery、train、test 完全不重叠。32×32 原图会稳定导出到数据目录，再由公共 Qwen 数据管线放大到 224×224。

模型不使用教师、attention、MoE、DeepStack 或光学层。Vision 使用 2 个 192 通道的 `3×3 depthwise Conv2D + pointwise Linear + channel MLP` block；Language 使用 2 个 `kernel=5` 的 causal Conv1D mixer block。只保留冻结 Qwen main merger，最终对有效多模态 token 做 mean+max pooling，再投影为 64 维归一化 embedding。

损失保持为 `1.0 × supervised contrastive + 1.0 × episodic prototype CE`。正式训练 12 epoch；由于每个 epoch 约包含五万张图，它的 optimizer step 数显著多于 Caltech101 的 60 epoch，不能只按 epoch 数比较训练量。

命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
