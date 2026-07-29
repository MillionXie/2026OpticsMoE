# ABO 两阶段商品图像检索：Qwen3-VL Teacher → Optical MoE16 Student

本工程用于验证：

> 给定商品的新视角 Query 图像，能否从已登记商品 Gallery 中检索出相同 `item_id`。

它是独立实验，不修改 Grocery Store 商品检索及其他已有工程。数据来自
[Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)。
默认自动下载官方 83 MiB 商品元数据包和约 3 GiB 的 256px catalog image 小图包；
224×224 输入无需下载 110 GiB 原图包。

## 最重要的防泄漏规则

`prepare_data` 在整个流程最开始就生成并锁定：

- Stage-2 train images；
- Gallery images；
- Query images。

Gallery/Query 的 `image_id` 随后被硬性排除于：

- Stage-1 大规模预训练；
- Stage-2 商品微调。

代码会检查集合交集；一旦发现泄漏立即终止。固定划分保存在：

```text
cache/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16*/manifests/abo_fixed_split.csv
```

同一 `item_id` 会同时存在于 Stage-2 train、Gallery 和 Query，因此这是
closed-set instance retrieval；但三部分没有重复图像。修改划分配置后，旧 manifest
不会被静默覆盖。确实要建立全新实验时才设置 `dataset.rebuild_manifest: true`。

## 网络

Student 只运行 Qwen Vision，不运行 Language Model：

```text
224×224 RGB image
→ frozen Qwen Vision patch/position embedding
→ Linear(Qwen hidden,224) + LayerNorm + Softplus
→ electronic Top-4 router
→ 16个224×224专家中的Top-4（每专家1层phase）
→ OEO detection / per-expert LN / ReLU / routing weight reload
→ 986×986 global phase
→ 10 cm propagation
→ CCD + pooling: [B,224,224]
→ valid token rows only
→ LayerNorm(224) + Linear(224,224)
→ mean token pooling
→ L2 normalize
→ 224D retrieval embedding
```

不同纵横比的 ABO 商品图会由 Teacher 和 Student 共用同一预处理：保持纵横比缩放，
再以白色背景居中 letterbox 到 224×224。这样不会拉伸包装，同时避免 Qwen grid 对齐
把非方形图像推到 224 个 optical token 以上；代码仍会对运行时 token 数严格检查，
不会裁剪 token 或 hidden。

物理布局保持既有 MoE16：

- 4×4 experts；
- expert size 224；
- pitch 254（30 pixel gap）；
- active footprint 986×986；
- FFT canvas 1026×1026；
- wavelength 532 nm；
- pixel pitch 8 µm；
- propagation distance 10 cm；
- one expert phase + one global phase。

通用 `224→Qwen hidden` restore adapter 没有被调用并保持冻结。阶段二 ID head
`Linear(224, item_count)` 仅帮助训练；`deployment_encoder.pt` 中明确不包含它。

## 两阶段损失

Stage 1：

```text
L = 1.0 * cosine KD(Qwen embedding)
  + 1.0 * supervised contrastive loss
  + 0.03 * router balance
```

Stage 2：

```text
L = 1.0 * supervised contrastive loss
  + 0.5 * item-ID CE
  + 0.2 * cosine KD
  + 0.03 * router balance
```

Teacher 使用固定 instruction：

```text
Represent this product image for image-to-image product retrieval.
```

Teacher 输出采用 Qwen3-VL-Embedding 的 224D Matryoshka 前缀并再次 L2
归一化，没有额外训练 MLP。

## 商品筛选

Stage 2 默认从图像数不少于 4 的商品中选择。筛选采用可复现的 catalog-quality
proxy：原图分辨率、边缘清晰度、对比度及边界背景均匀程度；再在最密集的商品类型中
均衡选择，使同类型相似商品形成更有价值的 hard negatives。这是工程筛选代理，不是
人工质量标注，实际论文实验前应抽样核查 manifest。

对于 5 张图的商品，固定划分严格为 3 train / 1 Gallery / 1 Query；主 catalog
图优先固定为 Gallery，Query 从其余视角中选择。

## Checkpoint 纪律

没有 validation。两个训练阶段均以最低训练 loss 保存 best checkpoint；
Gallery/Query 不用于选 epoch、调参或反向传播。最终 `evaluate` 才运行一次大库检索。

报告：

- Top-1；
- Recall@5；
- Recall@10；
- mAP；
- MRR；
- 每个 item 的指标；
- 每个 Query 的 Top-10 item 和相似度。

在 item prototype 排名中，每个 Query 只有一个相关 item，因此 AP 等于倒数排名，
所以此协议下 mAP 与 MRR 数值相同。

## 推荐顺序

1. `abo_smoke.yaml`：20-item 程序 smoke；
2. `abo500.yaml`：500-item 损失和检索协议验证；
3. `abo3000.yaml`：正式 3,000-item 实验；
4. `abo5000.yaml`：扩展到 5,000-item Gallery。

ABO 的官方网页与 AWS Registry 对许可名称的展示目前并不完全一致；用于论文或共享数据
前，请按官方数据页面、archive 内 license 和单位要求完成归属与合规核查。
