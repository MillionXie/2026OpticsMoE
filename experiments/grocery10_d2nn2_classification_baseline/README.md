# Grocery-10 grayscale two-plane D2NN baseline

这是商品光学检索实验的独立纯光分类对照。它使用完全相同的 10 个
SKU 和官方 train/test 图像划分，但任务是固定类别的十分类，不使用
gallery 相似度检索。

## 严格 baseline 定义

```text
RGB 商品文件
→ 转为 448×448 灰度振幅（单一标量光场）
→ 与第一块 448×448 phase-only mask 共面
→ 10 cm 自由空间传播
→ 986×986 global phase-only mask
→ 10 cm 自由空间传播
→ 986×986 有效 CCD
→ 十个固定等面积探测区域（3 / 4 / 3）
→ 十个非负区域能量
→ detector-region cross-entropy
```

第一相位板采用 448×448，是因为四个 224×224 专家具有相同的总像素
面积：`4 × 224² = 448² = 200,704`。它并不模拟四个专家；这只是给
非 MoE D2NN 一个面积尺度更合理的单块输入相位板。

输入图片虽然以 RGB 文件读取，但在进入光路前固定转换为亮度：

```text
gray = 0.2989 R + 0.5870 G + 0.1140 B
```

模型不支持 RGB 四象限编码，也不含可训练颜色 stem。

## 明确不包含

- Qwen、Teacher、蒸馏或 embedding cache；
- MoE、router 或专家选择；
- cosine similarity、prototype 或 64 维检索 embedding；
- detector 后的 Linear/MLP；
- 两块相位板之间的平方探测、LayerNorm、ReLU 或 OEO；
- gallery 图片加入训练；
- detector-plane MSE 优化分支。

推理类别就是能量最大的物理探测区域。训练中的 `log` 只用于计算
cross-entropy，不是可训练电子读出层。

## 参数量

```text
第一块局部相位板：448² = 200,704
第二块 global phase：986² = 972,196
总可训练相位参数：1,172,900
可训练电子参数：0
```

传播 canvas 为 1026×1026，有效范围为 986×986，即四周各保留 20
pixel 的 FFT guard band。波长为 532 nm，pixel pitch 为 8 μm。

## 比较边界

这是 closed-set classification；原实验是 query-to-gallery retrieval。
两者都能反映对 10 个 SKU 的区分能力，但任务和输出接口不同，文章中
应分别标为 “D2NN direct classification” 与 “embedding retrieval”。

测试集可以按 epoch 打印用于观察，但 checkpoint 只按最低训练损失
保存，测试集不参与权重选择。
