# Architecture

## 检索任务

    query image -> normalized 64-D embedding
    gallery image -> normalized 64-D embedding
    cosine similarity -> category ranking -> Top-1 / Top-3 / MRR

默认 `mean_prototype`：同一类别的 gallery embedding 先分别 L2 normalize，再求均值并再次 L2 normalize；也支持 `max_similarity`。

## 冻结 Teacher

    RGB image + fixed object-retrieval instruction
    -> frozen Qwen3-VL-Embedding-2B
    -> official 64-D embedding output
    -> L2 normalize

Teacher 始终为 `eval + no_grad + requires_grad=False`，只离线缓存一次，不训练额外 MLP、分类头或 LoRA。

## Optical Student

    224x224 RGB
    -> frozen Qwen processor / patch embedding
    -> Vision input adapter (teacher hidden -> 224)
    -> Vision Optical MoE4 (2x2, Top-2, one phase-only expert plane)
    -> per-expert CCD + LN + ReLU
    -> reapply original routing amplitude; unselected experts stay zero
    -> response-amplitude preservation
    -> Vision global phase -> propagation -> CCD token rows
    -> frozen Qwen multimodal merge/injection
    -> Language input adapter (2048 -> 224)
    -> Language Optical MoE4 with identical physical layout
    -> Language global phase -> propagation -> CCD token rows
    -> last valid token feature
    -> LayerNorm -> Linear(224,64) -> L2 normalize

没有类别分类头；监督和评测均在 64 维检索空间完成。Vision 和 Language 都是光学 MoE，native attention、Transformer residual 与 phase dropout 默认关闭。

## 光路参数

- 4 个 224×224 专家，2×2 排布，pitch 254，active 478，FFT canvas 518；
- Vision/Language 各包含一层 expert phase 和一层 global phase；
- 532 nm，仿真像素 16 μm；
- expert→OEO、OEO→global、global→CCD 均为 5 cm；
- `phase = 2*pi*sigmoid(raw_phase)`，raw phase 零初始化，对应初始相位 π；
- K-space 约束开启，`theta_max_deg=0.65`；
- 2×2 CCD integration；DC loss、phase dropout 关闭。

## Loss

    L = 8.0 * embedding KD
      + 0.10 * relational KD
      + 1.0 * supervised contrastive retrieval
      + 0.25 * student gallery-prototype CE
      + 0.10 * teacher-gallery alignment
      + 0.02 * router balance
      + 0.005 * router importance
      + 0.10 * routing-response consistency

PK batch 默认为 `10 classes × 3 images = 30`，让每个 anchor 在 batch 内至少拥有一个同类正样本。

## 两阶段兼容性

模型没有依赖类别数的分类层。第一阶段 101 类与第二阶段 10 类只改变 manifest、sampler、gallery 和训练样本，因此 optics、router、adapter 和 64-D readout 可完整续训。第二阶段从 epoch-30 EMA 权重开始，重新初始化 AdamW，并使用较小学习率训练到绝对 epoch 50。
