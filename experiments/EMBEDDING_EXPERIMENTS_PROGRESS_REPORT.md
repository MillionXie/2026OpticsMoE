# Qwen3-VL-Embedding-2B 光学实验阶段性汇报

更新时间：2026-07-30

## 1. 研究目标与公共架构

这一阶段主要验证冻结的 `Qwen3-VL-Embedding-2B` 能否作为电子教师，
指导可部署的光学网络完成商品检索和类别无关显著性分割。

当前较稳定的单层 Optical MoE16 主干为：

```text
224×224 RGB image
→ frozen Qwen vision patch / position embedding
→ trainable hidden-to-optical input adapter
→ electronic Top-4 router
→ 4×4 Optical MoE16
   └─ 16 个 224×224 phase-only experts，每个专家 1 个相位面
→ OEO detection / normalization / ReLU / routing-weight reload
→ 986×986 global phase
→ free-space propagation
→ CCD intensity
→ 224×224 detector readout
→ task-specific electronic readout/head
```

物理画布为 `1026×1026`，有效范围为 `986×986`。16 个专家按
`4×4` 排列，expert pitch 为 254，专家之间保留 30 像素间隔。商品
检索输出归一化 embedding；分割任务保留 token 的二维空间排列并输出
`224×224` mask。

## 2. Grocery Store Dataset 商品检索

### 2.1 任务

输入自然环境拍摄的商品图片，从固定 gallery 中检索相同 SKU。Teacher
和 Student 都输出 L2-normalized embedding，使用余弦相似度完成检索。

主要流程经历了：

1. 10-SKU 直接训练；
2. 扩展到 31 个包装商品进行预训练；
3. 回到重新筛选的 10-SKU 数据微调；
4. 在微调权重上增加较强但不破坏包装文字的增强，并使用 EMA 参数评测。

### 2.2 结果

| 实验 | 系统 | Top-1 | Top-3 | MRR |
|---|---|---:|---:|---:|
| 初始 10-SKU | Frozen Teacher | 90.59% | 99.22% | 0.9458 |
| 初始 10-SKU | Optical Student | 30.98% | 74.51% | 0.5472 |
| 31-SKU 预训练 | Frozen Teacher | 75.03% | 95.39% | 0.8529 |
| 31-SKU 预训练 | Optical Student | 49.55% | 79.26% | 0.6629 |
| 10-SKU 微调 | Frozen Teacher | 90.77% | 99.23% | 0.9481 |
| 10-SKU 微调 | Optical Student | 71.54% | 91.15% | 0.8219 |
| 10-SKU 强化增强 + EMA | Optical Student | **73.46%** | **91.92%** | **0.8362** |

### 2.3 当前结论

- 31-SKU 预训练再微调明显优于直接在小规模 10-SKU 上训练。
- 当前最佳 Student Top-1 为 73.46%，已经保留了 Teacher 较大一部分
  商品检索能力。
- 主要错误集中在包装非常相似的奶制品、果汁和同品牌近邻 SKU。
- EMA 和适度增强带来小幅、稳定的进一步提升，但没有消除近邻 SKU
  混淆。

## 3. ABO 大图库商品实例检索

### 3.1 任务

ABO 实验将难度提升为同一 `item_id` 的多视角实例检索。Query 与 Gallery
图像严格分离；测试 item 在训练中出现，但测试图像不参与训练。

训练分为两阶段：

1. Stage 1：约 60,000 张商品图像/epoch 的大规模多视角预训练，共
   50 epochs；
2. Stage 2：3,000 个高质量商品微调，共 50 epochs，并加入仅训练期
   使用的 ID head。

部署时移除 ID head，只保留：

```text
Optical core + global phase + CCD readout + embedding projection
```

### 3.2 训练状态

- Stage 1 loss：1.8606 → 0.7376；
- Stage 2 loss：4.7100 → 0.8067；
- Stage 2 ID 训练准确率：0.05% → 86.94%；
- `stage1_best_train_loss.pt`、`stage2_best_train_loss.pt` 和
  `deployment_encoder.pt` 均已生成；
- 正式 3,000-item 评测已经完成。

### 3.3 结果

| 系统 | Top-1 | Recall@5 | Recall@10 | mAP / MRR |
|---|---:|---:|---:|---:|
| Frozen Teacher | 48.65% | 76.46% | 83.40% | 0.6105 |
| Optical Student | 9.66% | 20.02% | 25.57% | 0.1515 |
| Student query / Teacher gallery（诊断） | 2.06% | 8.48% | 14.15% | 0.0632 |

500-item 规模下：

| 系统 | Top-1 | Recall@5 | Recall@10 | mAP / MRR |
|---|---:|---:|---:|---:|
| Frozen Teacher | 65.54% | 90.23% | 92.72% | 0.7651 |
| Optical Student | 13.32% | 31.44% | 42.10% | 0.2284 |

### 3.4 当前结论

- Student 已经能够拟合训练身份，但在大规模实例级 gallery 上泛化明显
  不足。
- Teacher 在 3,000-item 上也只有 48.65% Top-1，说明单张 gallery、
  视角差异和大量相似商品共同造成了很高难度。
- Student-query / Teacher-gallery 更差，说明 Student 与 Teacher 全局
  embedding 坐标系仍未充分对齐；这不仅是 gallery 聚合问题。
- 后续更合理的路线是先缩小 gallery 或增加每个 item 的 gallery 视角，
  再加强多视角一致性学习，而不是单纯延长现有训练。

## 4. FSS-1000 类别无关显著性分割

### 4.1 电子 Teacher 上限

```text
Frozen Qwen Vision spatial hidden
→ lightweight electronic segmentation head
→ 224×224 binary mask
```

Teacher 测试结果：

| mIoU | Dice/F1 | MAE | Pixel Accuracy |
|---:|---:|---:|---:|
| **0.7163** | **0.8225** | **0.0807** | **0.9294** |

这说明冻结 Qwen Vision 的空间特征对 FSS-1000 分割具有较好的可用性。

### 4.2 单层 Optical Student

| 实验 | mIoU | Dice/F1 | MAE | Pixel Accuracy |
|---|---:|---:|---:|---:|
| 初始正式 Student | **0.5570** | **0.6936** | 0.1608 | **0.8656** |
| 单层 + global phase，从零训练 100 epochs | 0.5460 | 0.6832 | **0.1536** | 0.8623 |

从零训练 100 epochs 的过程中，观察到的最高 test mIoU 为 0.5532，
但正式 checkpoint 仍按最低训练损失选择，test 不参与选择。

### 4.3 COCO → DUTS 预训练尝试

为了提高空间泛化能力，另建了三层 Optical MoE16 预训练路线：

```text
COCO train2017
→ Qwen merger 前 spatial hidden
→ fixed PCA 1024→224
→ 三层 Optical MoE16 特征蒸馏
→ DUTS 分割预训练
→ FSS-1000 迁移微调
```

PCA oracle：

- hidden cosine：0.9894；
- token 平均相对重建误差：0.0841；
- PCA 仅生成监督目标，不进入 Student 推理路径。

COCO 特征蒸馏完成 30 epochs，但提升有限：

- train cosine：0.1945 → 0.2088；
- train loss：4.2041 → 4.1195。

DUTS 完成 55 epochs：

| mIoU | Dice/F1 | MAE | Pixel Accuracy |
|---:|---:|---:|---:|
| 0.3360 | 0.4704 | 0.2256 | 0.7956 |

### 4.4 迁移状态

COCO/DUTS checkpoint 已完成，但截至本报告生成时，真正的 FSS 迁移训练
尚未启动。正确的迁移工程是：

```text
experiments/qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16
```

已完成 shape smoke，并确认实际加载：

- `optical_core`；
- `ccd_residual_recombiner`；
- `segmentation_head`；
- DUTS source epoch 55。

验证形状：

```text
visual tokens:       196
CCD:                 [B,224,224]
spatial feature:     [B,224,14,14]
mask logits:         [B,1,224,224]
```

需要注意：此前运行的
`fss1000_saliency_single_layer_from_scratch_100ep.yaml` 是从零训练 baseline，
不是预训练迁移。由于 DUTS test mIoU 目前仅为 0.3360，迁移训练不一定
超过单层 from-scratch 的 0.5570，但仍应通过正式迁移实验验证。

## 5. 阶段性判断

1. **小规模商品检索最成功。** 31-SKU 预训练后回到 10-SKU 微调，
   Student Top-1 从约 31% 提升到 73.46%。
2. **FSS 分割达到中等可用水平。** 单层 Optical Student mIoU 约
   0.55，相对电子 Teacher 的 0.716 仍有明显差距。
3. **当前 COCO/DUTS 预训练目标尚未充分学好。** PCA 本身质量高，但
   Optical Student 对 PCA target 的拟合改善很小，DUTS 跨数据集效果也
   较弱。
4. **ABO 3,000-item 实例检索过难。** Student 在训练身份上可以拟合，
   但对新视角的实例级检索泛化不足；Teacher 本身的上限也明显下降。
5. 后续应优先完成正确的 FSS 预训练迁移对照，并将 ABO 分成 gallery
   规模、gallery 视角数和商品相似度三个维度逐步消融。

## 6. 数据和结果路径约定

- 大型原始数据统一放在仓库根目录 `data/`；
- 可复用特征/PCA/teacher cache 放在仓库根目录 `data/cache/` 或
  `cache/`；
- 每个实验的训练结果统一放在该实验自己的
  `experiments/<experiment>/runs/`；
- 不再把新结果写入仓库根目录 `runs/`；
- 原始数据、cache、runs 和 checkpoint 均不提交 Git。

