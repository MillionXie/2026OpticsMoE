# Qwen3-VL Vision Optical MoE16 on FSS-1000

本实验把 FSS-1000 的所有类别统一视为前景，完成类别无关的二值显著性目标检测。输入为 `224×224 RGB`，输出为 `224×224` 单通道 mask logits。它是从
`qwen3_vl_embedding_2b_grocery10_optical_retrieval` 独立派生的新实验，不会修改商品检索工程。

## 数据划分

代码使用 FSS-1000 官方 `fss_test_set.txt` 的 240 个测试类别。因为本实验明确不创建 validation，其余官方 train/validation 类别合并为训练集。完整数据通常应得到：

- train：760 个类别、7,600 张图；
- test：240 个类别、2,400 张图；
- train/test 类别交集严格为空。

程序会把实际类别数、图像数和路径写入 `dataset.json`，并持久化
`manifests/class_split.json` 与 `manifests/samples.csv`。若实际数据不符合上述结构，
以 `dataset.json` 的扫描结果为准，不会静默伪造样本。

`prepare_data` 默认先从服务器可访问的公开 Hugging Face 镜像
`nobg/FSS-1000` 下载并验证 10,000 条 image/mask/class_name 记录，再物化为官方目录；
若失败则回退官方 Google Drive 文件 ID。Google 回退需安装 `gdown`。
数据目录也可手动整理为：

```text
data/FSS-1000/fewshot_data/<class>/1.jpg
data/FSS-1000/fewshot_data/<class>/1.png
```

图像和 mask 使用同一套随机裁剪、翻转和旋转参数。RGB 使用 bicubic，mask 始终使用
nearest-neighbor 并在读取后重新二值化。

## 两条模型路径

电子 Teacher 上限：

```text
RGB image
→ frozen Qwen image processor
→ frozen patch/position embedding
→ all frozen native Qwen Vision blocks
→ last pre-merger spatial hidden [ΣT, Dv]
→ runtime image_grid_thw restore [B,Dv,Ht,Wt]
→ lightweight electronic segmentation head
→ [B,1,224,224] logits
```

Optical Student：

```text
RGB image
→ frozen Qwen patch/position embedding
→ native Vision blocks bypassed
→ Linear(Dv,224) + LayerNorm + Softplus
→ existing electronic top-4 router
→ 16 one-layer 224×224 phase-only experts
→ propagation + OEO detector/LN/ReLU/reload
→ 986×986 global phase
→ 10 cm propagation
→ 986×986 CCD ROI, pooled/read to [B,224,224]
→ only valid token rows
→ runtime image_grid_thw restore [B,224,Ht,Wt]
→ lightweight electronic segmentation head
→ [B,1,224,224] logits
```

这里不运行 language model、不创建文本 instruction、不做全局池化，也不使用商品检索的
64 维 readout。原光学 core 中的 `Linear(224,Dv)` output adapter 不参与 segmentation
loss，已冻结且 forward 不调用。

## 空间映射的严格性

Qwen processor 的 `image_grid_thw` 是唯一空间形状来源。实现不会硬编码 14×14：

- `T×H×W` 必须与捕获的 token 数完全相同；
- 当前静态图像要求 `T=1`；
- batch 内网格必须一致；
- token 超过 224、数量不匹配或空间网格不一致时直接报错；
- 不允许 crop、truncate 或 fallback reshape。

在默认 pixel budget 50,176 和正方形 224 输入下，常见网格是
`[1,14,14]`（196 tokens），但这只是运行结果，不是代码假设。

## 训练与 checkpoint

Teacher 和 Student 独立训练，损失均为：

```text
BCEWithLogits + Dice
```

Student 额外保留小权重 router balance/importance。没有 validation；checkpoint 只根据
最低训练损失保存。每 epoch 的 test 指标仅用于观察，不参与 checkpoint、反向传播或调参。

可选 `fss1000_saliency_mask_kd.yaml` 只蒸馏电子 Teacher 的最终 mask logits，不蒸馏
任何 Qwen hidden。缓存 logits 时关闭几何增强，避免像素错位。

主要指标为 mean IoU、mean Dice/F1、MAE 和 pixel accuracy。结果和典型/失败案例分别写入
`metrics/` 与 `figures/`。
