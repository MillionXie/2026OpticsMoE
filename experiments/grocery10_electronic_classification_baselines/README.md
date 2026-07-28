# Grocery-10 electronic direct-classification baselines

该工程为现有电子 retrieval baselines 提供另一套任务定义：模型不学习
64 维检索向量，也不与 gallery 做余弦相似度匹配，而是直接输出 10 个
类别 logits。

三个模型使用与光学商品实验完全相同的 10 个 SKU，以及官方
`train + val → train`、`test → test` 划分：

- ResNet-18；
- EfficientNet-B0；
- MobileNetV3-Small。

正式配置使用 TorchVision ImageNet-1K 预训练权重，然后端到端微调。
共同的数据流为：

```text
RGB [B,3,224,224]
→ ImageNet-pretrained CNN backbone
→ global feature [B,F]
→ LayerNorm(F)
→ Linear(F,10)
→ raw class logits
→ CrossEntropyLoss
```

这里没有 L2 normalization、cosine similarity、gallery prototype、
supervised contrastive loss 或 embedding retrieval。训练日志明确保存
train/test Top-1，便于判断小数据集上的过拟合。

由于没有 validation split，checkpoint 只按最低训练 cross-entropy
保存；每个 epoch 的 test 仅用于观察，不参与反向传播或 checkpoint
选择。

该结果和 D2NN direct classification 属于同一任务接口，可以直接比较
Top-1/Top-3。它与主 optical student 的 gallery retrieval 指标不是完全
相同的任务，文章中应明确区分。
