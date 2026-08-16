# Caltech101 Target-10 纯电子 Qwen 检索

这个工程用于回答一个明确的对照问题：在相同 Qwen3-VL、相同 10 类数据划分、相同 64 维教师监督和 gallery 检索协议下，去掉所有衍射、相位、错位、k 空间及 MoE 路由后，纯电子学生能达到什么性能。

## 模型

- 冻结的 `Qwen/Qwen3-VL-Embedding-2B` 只提供多模态执行框架；训练不读取教师 embedding，也不依赖 teacher cache。
- 学生不载入任何 all-101 或光学 checkpoint，视觉和语言替换模块均独立随机初始化。
- 每个模态先把 Qwen hidden 投影到 128 维，再经过 2 个共享 token-wise residual MLP block：`LayerNorm -> Linear(128,256) -> GELU -> Dropout -> Linear(256,128)`。
- 每个 block 和模态输出都使用可学习 sigmoid 残差比例。语言端对所有有效多模态 token 做无参数 mean pooling，保证没有 attention 时视觉 token 仍能进入最终表示，然后经过 `LayerNorm -> Linear(128,64) -> L2Normalize`。
- 没有 self-attention、cross-attention、卷积或 token mixing；MLP 权重在 token 间共享，参数量不随 token 数量增长。
- 没有专家、router、相位面、FFT/Fresnel 传播或光路增强。共享训练器里显示的单路 routing 仅是无参数诊断占位，不参与计算。

## 数据与损失

固定 10 类，每类 3 张 gallery、20 张测试，其余图片全部训练，总计 30/2625/200。每个 epoch 使用自然数据量，类别均衡 PK batch 为 10 类乘每类 3 张。

总损失沿用可比较的检索目标：

`1.0 * supervised contrastive + 1.0 * episodic student prototype retrieval CE`

episodic loss 在每个类别的 3 张 batch 图片中随机取 1 张 support、2 张 query，下一批重新选择；support 和 query 均参与反向传播。没有 cosine KD、relational KD 或 teacher-gallery CE。

电子主体、输入输出 adapter、embedding head 的峰值学习率分别为 `1.5e-4`、`1e-4`、`2e-4`，前 5% step 线性 warmup，之后 cosine decay。AdamW weight decay 为 `0.01`，gradient clipping 为 `1.0`，EMA 为 `0.995`，共训练 60 epoch。每个 epoch 都输出测试指标以观察过拟合，但 checkpoint 仍按训练 loss 选择，不使用 test 选模型。

运行方式见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

`configs/release/caltech101_target10_mlp_teacher_kd.yaml` 是严格教师消融：只在上述两个学生 loss 之外增加 `1.0 × cosine embedding KD`，其余设置不变。relational KD 和 teacher-gallery CE 仍保持关闭。
