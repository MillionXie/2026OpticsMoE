# Grocery-10 two-plane D2NN classification baseline

本实验是商品光学检索实验的独立纯光对照。它使用相同的十个 SKU、官方
train/test 图像划分和主要物理参数，但任务被有意改成普通的十分类。

它不包含：

- Qwen 或其他预训练网络；
- Teacher、embedding cache 或蒸馏；
- MoE、router 或专家选择；
- 64 维 embedding、余弦相似度或对比学习；
- detector 后的 Linear/MLP；
- 两个光学层之间的平方探测、LayerNorm、ReLU 或 OEO 重新加载。

## 光路

```text
RGB 商品图
→ 转换为 224×224 灰度振幅
→ 中央 zero-pad 到 1026×1026
→ 与输入振幅共面的局部 phase-only mask [224,224]
→ 自由空间传播 10 cm
→ global phase-only mask [986,986]
→ 自由空间传播 10 cm
→ 有效 CCD [986,986]
→ 十个固定方形探测区（3/4/3）
→ 十个区域光强
→ CrossEntropy
```

`CrossEntropy` 直接作用于十个探测区域的 log-energy。这里没有可训练的
电子分类头；argmax 对应能量最大的物理探测区域。

输入和第一相位 SLM 默认共面，因此
`input_to_first_phase_distance_m=0.0`。传播使用与实验组一致的：

```text
wavelength = 532 nm
pixel pitch = 8 μm
canvas = 1026×1026
active aperture = 986×986
first phase → global phase = 10 cm
global phase → CCD = 10 cm
```

传播 canvas 比有效口径每边多 20 pixel，用作 FFT guard band。CCD 只读取
986×986 有效范围。

## 参数量

```text
第一局部相位板：224² = 50,176
第二 global phase：986² = 972,196
总可训练相位：1,022,372
可训练电子参数：0
```

## 数据和比较边界

本实验复用 `qwen3_vl_embedding_2b_grocery10_optical_retrieval` 的十个替换后
SKU 和官方划分。正式配置预期为 306 张训练图、260 张测试图和 10 张
iconic gallery 图；gallery 只保留在共同 manifest 中，不参与该分类模型训练。

需要注意，这里是固定十类的 closed-set classification，而原实验是
query-to-gallery retrieval。因此二者的 Top-1 可以用于判断“纯 D2NN 是否能
直接区分这些商品”，但不能被描述成完全相同的检索指标。

## Checkpoint 选择

测试集可按 epoch 打印用于观察，但 checkpoint 只按最低训练
cross-entropy 保存，测试结果不参与选权重。输出包括实时训练 CSV、最终测试
指标、per-SKU 指标、混淆矩阵、两个 phase mask 以及多组中间光场。
