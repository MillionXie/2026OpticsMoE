# COCO → DUTS：Qwen Vision PCA224 指导的 Optical MoE16 预训练

本工程为独立实验，不修改现有 FSS-1000、Grocery retrieval 或 SPAQ
工程。目标是先让光学主干在大量自然图像上学习 Qwen 的通用空间特征，
再用 DUTS-TR 学习类别无关显著性分割，最后将所得主干用于 FSS-1000
等下游任务。

## 统一主干

```text
224×224 RGB
→ frozen Qwen patch / position stem
→ Linear(1024,224) + LayerNorm + Softplus
→ one input-dependent electronic Top-4 router
→ Optical MoE16 stage 1 → OEO
→ Optical MoE16 stage 2 → OEO
→ Optical MoE16 stage 3 → OEO
→ global phase
→ 10 cm propagation
→ square-law CCD
→ crop physical 986×986 ROI
→ pool/LN/ReLU to [224,224]
→ Fout = Fccd + alpha·Linear(LayerNorm(Fccd))
→ valid token rows [T,224]
→ downstream task head
```

每个 optical stage 包含 16 个独立的 224×224 phase-only expert，布局为
4×4、pitch 254、间隔 30；每张图由电子 router 选择 4 个专家。三层使用
第一次输入得到的同一组专家和 routing weights，不在中间重新路由。

每层 OEO 执行：

```text
phase modulation
→ 10 cm propagation
→ square detection
→ selected expert 独立 LayerNorm
→ ReLU
→ 重新乘最初 routing weight
→ 未选专家严格置零
→ zero-phase amplitude reload
```

第三次 reload 后与 global phase 共面；global phase 后只传播 10 cm 到
CCD，不再加入 OEO。

共享重组层的 `alpha` 默认是可训练标量，初始化为 0.1。该层与 optical
core 一起参与 COCO 预训练、DUTS 训练和推理，并包含在所有 backbone
checkpoint 中。旧的 `Linear(224,1024)` hidden restore adapter 已从本
Student 中删除。

## 阶段一：COCO 通用特征蒸馏

Teacher 是完全冻结的 `Qwen/Qwen3-VL-Embedding-2B` Vision。只捕获最后
一个原生 Vision block 的 merger 前空间 hidden `[T,1024]`。不使用
语言模型、任务 head、COCO label、caption、mask 或 teacher logits。

PCA 只离线拟合一次：

```text
Qwen final pre-merger spatial hidden [T,1024]
→ fixed PCA
→ teacher target [T,224]
```

PCA：

- 只用于生成和缓存 teacher target；
- mean/components 均不训练；
- 不属于 Student；
- 不保存进 Student checkpoint；
- 不增加下游推理延迟；
- 会记录 explained variance、重建误差、token 数和 seed。

完整 COCO train2017（118,287 图）用于训练；val2017（5,000 图）只观察
拟合效果，绝不选择 checkpoint。checkpoint 按最小 train loss 保存。

```text
L = cosine_loss
  + 0.5 × SmoothL1(beta=0.1)
  + 0.03 × router_balance
```

Teacher PCA target 以 FP16 shard 存在 `data/cache/...`，不放在 runs。
写入过程可恢复：每个 shard 完成后原子更新索引，中断后再次执行会跳过
已经缓存的样本。

## 阶段二：DUTS 显著性预训练

加载 COCO 的 optical core 和共享 224→224 重组层，连接与现有 FSS
工程相同的轻量分割头：

```text
[B,224,14,14]
→ token LayerNorm
→ Linear(224,128)
→ 3×3 Conv/GN/GELU: 128→64→32→16
→ bilinear upsample to 224×224
→ 1×1 Conv
→ binary mask logits
```

训练日程：

1. 前 5 epoch 冻结 optical core 和重组层，只训练 segmentation head；
2. 随后 50 epoch 重建 optimizer 并联合训练：
   - optical core LR = `1e-4`
   - recombiner LR = `2e-4`
   - segmentation head LR = `1e-3`

损失：

```text
BCEWithLogits
+ Dice
+ 0.75 × SoftIoU
+ 0.25 × Boundary
```

DUTS-TE 每个 epoch 的结果仅用于观察；checkpoint 仍按最小 DUTS-TR
train loss 保存。

## 自动下载与磁盘

`prepare_data` 自动下载：

- COCO official `train2017.zip`、`val2017.zip`；
- DUTS 官方 `DUTS-TR.zip`、`DUTS-TE.zip`。

不下载 COCO annotations，因为本阶段不使用 label。解压后会严格验证：

- COCO train：118,287；
- COCO val：5,000；
- DUTS-TR：10,553 对；
- DUTS-TE：5,019 对。

正式实验建议预留至少 70 GB：

- COCO/DUTS 图像及下载解压过程；
- Qwen checkpoint；
- 约 10.4 GB 的完整 COCO FP16 PCA224 teacher target；
- checkpoints、日志和可视化。

默认在成功解压后删除 ZIP，以免同时保留压缩包与目录。
下载中断时会保留 `.part` 并在下次执行时通过 HTTP Range 续传。

## 输出

主要文件：

```text
resolved_config.yaml
dataset.json
metrics/pca_fit.json
metrics/pca_oracle.json
metrics/coco_training_history.csv
metrics/duts_training_history.csv
metrics/duts_test.json
checkpoints/coco_backbone_best_train_loss.pt
checkpoints/coco_backbone_last.pt
checkpoints/duts_student_best_train_loss.pt
checkpoints/duts_student_last.pt
figures/coco_training_curves.png
figures/duts_training_curves.png
figures/duts_examples/
figures/*_phase_masks/
```

## 注意

- 224×224 输入通常产生 `image_grid_thw=[1,14,14]` 和 196 个 pre-merger
  token；代码始终根据运行时 grid 检查，不硬编码 196。
- token 数超过 224 会直接报错，不 crop、truncate 或重排 token。
- mask resize 始终使用 nearest-neighbor。
- COCO val 和 DUTS test 均不参与参数更新或 checkpoint 选择。
- smoke 配置只限制使用的样本数量；若本机没有数据，自动下载仍需先取得
  完整官方 ZIP，因为官方服务器不提供按样本下载。
