# Caltech101 Target-10 纯电子 Qwen 检索

这个工程用于回答一个明确的对照问题：在相同 Qwen3-VL、相同 10 类数据划分、相同 64 维教师监督和 gallery 检索协议下，去掉所有衍射、相位、错位、k 空间及 MoE 路由后，纯电子学生能达到什么性能。

## 模型

- 冻结的 `Qwen/Qwen3-VL-Embedding-2B` 仍负责教师 embedding，并提供被替换前后的 Qwen 多模态执行框架。
- 学生不载入任何 all-101 或光学 checkpoint，视觉和语言替换模块均独立随机初始化。
- 每个模态先把 Qwen hidden 投影到 256 维，再经过 2 个 pre-norm block。每个 block 是 8 头 self-attention 加 SwiGLU FFN；视觉使用双向 attention，语言使用 causal attention。
- 模块输出通过一个可学习 sigmoid 门控残差映射回 Qwen hidden。最终语言有效 token 的 256 维电子特征进入小型 gated-SwiGLU 残差 head，再投影并归一化为 64 维 embedding。
- 没有专家、router、相位面、FFT/Fresnel 传播或光路增强。共享训练器里显示的单路 routing 仅是无参数诊断占位，不参与计算。

## 数据与损失

固定 10 类，每类最多 3 张 gallery、30 张训练、20 张测试，总计 30/300/200。每个 epoch 使用自然数据量，PK batch 为 10 类乘每类 3 张。

总损失沿用可比较的检索目标：

`5.0 * cosine KD + 0.5 * relational KD + 1.0 * supervised contrastive + 0.5 * student gallery CE + 0.5 * teacher-gallery CE`

电子主体、输入输出 adapter、embedding head 的初始学习率分别为 `1.5e-4`、`1e-4`、`2e-4`，AdamW weight decay 为 `0.01`，gradient clipping 为 `1.0`，EMA 为 `0.995`。每个 epoch 都评测测试集，因此能直接比较训练/测试曲线判断过拟合。

运行方式见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
