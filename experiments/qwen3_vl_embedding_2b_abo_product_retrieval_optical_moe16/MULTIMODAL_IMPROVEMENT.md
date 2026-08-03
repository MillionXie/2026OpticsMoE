# ABO 多模态光学检索改进版

## 为什么需要改

原 3,000 商品实验的 Teacher/Student Top-1 分别约为 `48.65% / 9.66%`。
日志还显示路由 importance loss 接近 15（16 专家时的极端塌缩量级），而
Stage 2 的 KD loss 反而上升。原版只替换 Qwen Vision，直接从视觉 CCD
池化得到检索向量；这对细粒度实例检索的表达能力不足。

ABO 当前筛选集每个商品平均约 5.7 张真实图片，中位数 5，实际范围 4–10。
全库中拥有 15 张以上图片的商品也很少，因此不能通过配置获得“每商品几十张
独立实拍图”。改进版只对训练图进行轻量随机裁剪、旋转和亮度/对比度变化；
固定 Gallery 与 Query 始终不增强，也不会进入训练。

## 新的数据流

配置：`configs/abo3000_multimodal.yaml`

```text
image + fixed retrieval instruction
→ frozen Qwen image/text embeddings
→ Vision Optical MoE16（1 expert phase + 1 global phase）
→ frozen Qwen vision merger + one native DeepStack injection
→ Language Optical MoE16（1 expert phase + 1 global phase）
→ final valid language-token CCD row
→ LayerNorm(224) + Linear(224,224)
→ L2-normalized 224D retrieval embedding
```

Qwen 原生 Vision/Language Transformer block 仍冻结并由光学 surrogate 替换；
训练的是两套 optical adapters、两个电子 Top-4 routers、专家/global phase、
OEO normalization 和最终 224D readout。默认不加入 attention，也不启用
Transformer residual，以便单独测量 Vision+Language Optical MoE 的能力。

## Teacher 适配

`Qwen3-VL-Embedding-2B` 主干完全冻结。离线缓存最终有效 language token 的
2048D hidden，然后只训练：

```text
LayerNorm(2048) → Linear(2048,224) → L2 normalize
```

该约 46 万参数的 product metric adapter 使用 Stage-2 train 图像的
SupCon + cosine-margin identity loss；Gallery/Query 不参与训练。adapter
只用于生成更适合 ABO 的蒸馏目标，不进入 Student 部署。

## 对比学习与路由

- 物理 batch 为 `8 items × 2 views = 16`；
- 8192 条 detached FIFO embedding memory 提供跨 batch 负例；
- 增加 batch 内 pairwise-similarity 蒸馏；
- router temperature 从 2.0 逐步降到 1.0；
- 同时记录/约束 balance、importance 和 dense routing entropy；
- 每个 epoch 保存 Vision/Language 每位专家的选择率和平均权重；
- phase、router、adapter、training-only identity head 使用独立学习率；
- phase 不使用 weight decay，电子参数使用轻量 weight decay。

旧 `abo500/3000/5000.yaml` 默认仍保持 `vision_only + frozen_mrl` 和原单一
优化器行为，旧 checkpoint 也继续支持加载。改进版使用独立 cache/run 目录，
不会覆盖此前结果。

## 运行顺序

`--phase all` 依次完成数据准备、原始 Teacher hidden 缓存、Teacher metric
adapter、Stage 1、Stage 2、最终 Gallery/Query 评测。正式运行前建议先执行：

```bash
pytest experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/tests -q
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo_multimodal_smoke.yaml --phase all
```

正式 3,000 商品实验：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase all
```
