# Grocery-10 商品图像检索：Qwen3-VL-Embedding Teacher + Optical Student

本目录也提供 `grocery31_pretrain.yaml`，用于先在 Grocery Store Dataset
的全部 31 个包装 SKU 上预训练同一个光学 Student，再将其 checkpoint
迁移到筛选后的 10 个 SKU 上微调。31 类预训练不会改变物理网络结构；
它只扩大商品外观覆盖范围。

当 gallery-aligned loss 开启时，`train_log.csv` 额外记录
`train_top1`、`train_top3` 和 `train_mrr`。这些训练指标表示“当前增强后
训练 query 对本 batch 所选 SKU 的 Student gallery prototype”的在线检索
结果；测试指标仍来自完整固定 test/gallery。两者之间的差距可用于判断
过拟合，但训练在线指标通常比完整测试更乐观。

31 类训练使用 `P=10, K=3`。每个 batch 只附加当前 10 个 SKU 的标准
gallery 图，因此一次模型前向仍为 `30 query + 10 gallery = 40`，不会因
总类别从 10 增到 31 而膨胀为 61。

本实验验证一个小规模、部署形式明确的商品图库检索任务：

```text
自然环境商品图 → 64 维归一化向量 → 余弦相似度 → Top-1 / Top-3 SKU
```

它不是十分类实验。模型没有十分类 logits、CrossEntropy 分类头或 reranker；SKU
标签只用于 supervised contrastive loss 和检索指标。

## 数据与 10 个 SKU

代码自动下载官方 Grocery Store Dataset，并严格使用其细粒度 SKU 和官方划分。
默认配置选择以下 10 种包装商品：

1. `Bravo-Apple-Juice`
2. `God-Morgon-Orange-Juice`
3. `Tropicana-Golden-Grapefruit`
4. `Arla-Ecological-Medium-Fat-Milk`
5. `Garant-Ecological-Standard-Milk`
6. `Oatly-Natural-Oatghurt`
7. `Oatly-Oat-Milk`
8. `Alpro-Blueberry-Soyghurt`
9. `Alpro-Shelf-Soy-Milk`
10. `Yoggi-Strawberry-Yoghurt`

水果、蔬菜和散装商品不进入实验。10 个名字仅存在于 YAML 配置中，训练脚本没有
硬编码 SKU。

划分方式：

- train query：官方 `train + val`；
- test query：官方 `test`；
- gallery：每个 SKU 的官方 iconic 商品图（官方数据只有一张）；
- 不创建 validation；
- manifest 固定并保存 SHA256；
- train、test、gallery 的图片绝对路径必须两两不相交。

图片文件不会被复制到新的类别目录。`grocery10_subset.csv` 只记录路径和元数据。

## Frozen Teacher

Teacher 是完全冻结且始终处于 eval 模式的
`Qwen/Qwen3-VL-Embedding-2B`。每张图只使用统一 instruction：

```text
Represent this product image for image-to-image product retrieval.
```

输入不含商品名、SKU、标题或描述。低维输出遵循 Qwen embedding 的 Matryoshka
接口语义：

```text
最后有效 language token hidden
→ 取前 64 个 Matryoshka 维度
→ L2 Normalize
```

没有增加可训练的 `2048→64` MLP。gallery、train 和 test 的所有 Teacher
embedding 会预先缓存；Student 训练期间不会再次运行 Teacher。

## Optical Student

Student 复用仓库中已经验证的 Qwen DeepStack + homogeneous optical MoE16
replacement：

- frozen Qwen tokenizer、token embedding、vision patch embedding、vision merger、
  DeepStack injection 和 final RMSNorm；
- Vision Optical MoE16：**1 层专家相位 + 1 层 global phase**；
- Language Optical MoE16：**1 层专家相位 + 1 层 global phase**；
- 唯一专家层包含 16 个 `224×224` phase-only 专家，由电子输入相关
  Top-4 router 选择；
- 专家层传播后执行一次 square-law detection、逐专家 LayerNorm、ReLU
  和幅度重新加载，再经过 `986×986` global phase；
- `986×986` active footprint，四周各 20 像素传播 guard，FFT canvas `1026×1026`；
- 专家/global/CCD 的配置传播距离均为 10 cm；
- Transformer identity residual 保留；
- native electronic attention 关闭；
- phase dropout 关闭。

这对应仓库的 `...moe16_224_1layer_baseline`，不是四阶段 MoE16
版本。Student 仅保留一个辅助 DeepStack 路径：Vision stage-1 的
辅助输出经第一个 frozen DeepStack merger 注入；Language 的唯一
光学层位于 decoder index 1，使主视觉 embedding 和该辅助注入都先
进入语言光学层。

Language optical 最终 CCD 读出为非负 `[B,224,224]`。读取最后一个有效 language
token 对应的行，得到 detector feature `[B,224]`。检索读出只有：

```text
Detector feature [B,224]
→ LayerNorm(224)
→ Linear(224,64)
→ L2 Normalize
→ Student embedding [B,64]
```

Linear 后没有 ReLU、GELU、Sigmoid、Softmax，因此 64 维向量允许正负值。

## Loss

逻辑 batch 使用 `P` 个 SKU、每个 SKU `K` 张图。默认 `P=10, K=4`：

```text
L_kd  = mean(1 - cosine(student, frozen_teacher))
L_ret = supervised contrastive / supervised InfoNCE
L     = lambda_kd * L_kd + lambda_ret * L_ret
```

默认 `lambda_kd=1`、`lambda_ret=1`、temperature `0.07`。不使用分类 CE、
teacher logits、生成 loss 或 test loss。

## Epoch 50 后的修复性续训

首轮 50 epoch 使用普通 PK-batch supervised contrastive loss：每批
`10 SKU × 4 images`，因此每个 anchor 有 3 个同 SKU 正例和 36 个异
SKU 反例。它并非没有反例，但标准 gallery 图片只参与评测，没有进入
Student 训练；同时未启用 router 均衡损失，容易出现专家塌缩。

`configs/grocery10_continue100.yaml` 用于从已有 epoch-50 权重继续 100
个 epoch，并针对这两个问题进行修复：

- 每批为 30 张自然图（`10×3`）加 10 张固定标准 gallery，总前向仍是
  40 张，不提高原训练的 batch 峰值；
- supervised contrastive loss 使用自然图与 gallery 的联合 embedding；
- 增加 query 对 10 个可微 Student gallery prototype 的交叉熵检索损失，
  每个 query 显式面对 9 个错误商品反例；
- 加入已有 router 的 balance/importance loss，记录 Vision/Language
  路由熵、活跃专家数和最大 importance；
- base learning rate 降为 `2e-4`，router 单独使用 `1e-3`，Teacher KD
  权重提高到 3；
- 只加载 Student 权重并重置 AdamW 状态。旧 epoch 1–50 checkpoint、
  日志和指标会自动归档到
  `checkpoints/pre_resume_epoch_0050/`。

续训命令（从仓库根目录执行）：

```text
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_continue100.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_optical_retrieval/last_checkpoint.pt
```

如果首段恢复训练已经得到 epoch 57 checkpoint，则使用稳定版配置完成
epoch 58–150。该配置将 router LR 从 `1e-3` 降至 `1e-4`、base LR
降至 `5e-5`，并把 KD 权重提高到 5，以避免小数据下的快速路由振荡：

```text
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_continue_epoch57_stable.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_optical_retrieval/best_continuation_from_epoch_0050.pt
```

若 gallery 与 query 两端同时更新出现移动 prototype 振荡，可从稳定的
epoch 60 训练损失 checkpoint 使用固定-gallery-gradient 版本完成
epoch 61–150。这里只在 gallery CE 分支停止 prototype 梯度；10 张
gallery 仍在同一次 Student forward 中，并继续接受 Teacher KD：

```text
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_continue_epoch60_fixed_gallery.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_optical_retrieval/best_continuation_from_epoch_0057.pt
```

若该任务已完成 epoch 118，可在修复一次性 test DataLoader 的
`persistent_workers` 文件描述符泄漏后继续到 epoch 150：

```text
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_continue_epoch118_to150.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_optical_retrieval/last_checkpoint.pt
```

每轮可输出 test 指标供观察，但 checkpoint 只按训练总损失保存：

- `last_checkpoint.pt`：最后一轮；
- `best_train_loss_checkpoint.pt`：训练总损失最低；
- test 从不用于反向传播、调参或 checkpoint 选择。

## 三组评测

1. Frozen Teacher query vs Frozen Teacher gallery；
2. Optical Student query vs Optical Student gallery（主要部署结果）；
3. Optical Student query vs Frozen Teacher gallery（embedding 对齐诊断）。

gallery 支持：

- `mean_prototype`（默认）：同 SKU gallery 的归一化向量均值后再归一化；
- `max_similarity`：对该 SKU 的所有 gallery 图取最大相似度。

指标包括 Top-1、Top-3、MRR、每 SKU Top-1、10×10 confusion matrix、正确
SKU 相似度、最大错误 SKU 相似度和每个 query 的 Top-3 结果。

## 输出

所有运行结果默认保存在本实验目录内部：

```text
experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/
```

配置中的 `output_dir` 相对于 `configs/` 解析，因此使用 `../runs/...`。
数据集仍放在仓库级 `data/GroceryStoreDataset`，不会复制进实验输出目录。

正式 run 至少写出：

```text
config.yaml
environment.json
dataset.json
manifests/grocery10_subset.csv
teacher_cache/teacher_embeddings.pt
train_log.csv
teacher_metrics.json
student_metrics.json
per_sku_metrics.csv
retrieval_results.csv
confusion_matrix.png
teacher_retrieval_examples.png
student_retrieval_examples.png
student_failure_cases.png
last_checkpoint.pt
best_train_loss_checkpoint.pt
model.json
```

`model.json` 会列出每个可训练 tensor 的名称、shape 和参数量。Teacher 原始参数
保持冻结；可训练部分仅为两个 optical surrogate（含既有 adapters/router/OEO）
和新的 detector retrieval readout。

## 数据与模型要求

- 首次运行需要联网下载 Grocery Store Dataset；
- 首次运行需要可访问 `Qwen/Qwen3-VL-Embedding-2B`；
- 运行 full optical MoE16 推荐 CUDA GPU；
- 默认逻辑 batch 40 对显存要求较高，显存不足时同时减小 `P`、`K` 和
  `batch_size`，并保持 `batch_size=P×K`、`K≥2`。

数据或模型不可用时会明确报错，不会回退到合成数据、其他 Grocery 数据集或分类任务。
