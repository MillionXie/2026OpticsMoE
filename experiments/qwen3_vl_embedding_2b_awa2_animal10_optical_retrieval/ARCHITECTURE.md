# Architecture

## 任务

    query image -> 64-D normalized embedding
    gallery image -> 64-D normalized embedding
    cosine similarity -> class prototype ranking -> Top-1 / Top-3 / MRR

同类的 3 个 gallery embedding 先分别 L2 normalize，再求均值并再次 L2 normalize，形成
`mean_prototype`。配置也支持 `max_similarity`。

## Teacher

    RGB image + fixed animal-retrieval instruction
    -> frozen Qwen3-VL-Embedding-2B
    -> official Matryoshka-compatible first 64 embedding dimensions
    -> L2 normalize

教师始终为 `eval + no_grad + requires_grad=False`，只离线计算一次并缓存；不训练 MLP、
分类头或 LoRA。

## Optical Student

学生沿用 Grocery 检索实验中目前较稳定、可部署的 response-preserving MoE4：

    224x224 RGB
    -> frozen Qwen processor / token embedding / patch embedding
    -> Vision input adapter (teacher hidden -> 224)
    -> Vision Optical MoE4 (2x2, Top-2, one phase-only expert plane)
    -> per-expert CCD + LN + ReLU
    -> reapply the original routing amplitude once; unselected experts remain zero
    -> response-amplitude preservation
    -> Vision global phase -> propagation -> CCD 224-D token rows
    -> frozen Qwen multimodal merge/injection
    -> Language input adapter (2048 -> 224)
    -> Language Optical MoE4 with the same physical definition
    -> Language global phase -> propagation -> CCD token rows
    -> last valid token detector feature
    -> LayerNorm -> Linear(224,64) -> L2 normalize

没有分类 head；监督对比学习直接在 64 维检索空间进行。Vision/Language 都使用光学 MoE，
native attention 和 transformer residual 在本基线中关闭。

## 光路配置

每个 Vision/Language optical stack 均为：

- 2x2 共 4 个专家，Top-2；
- expert phase：`224x224`，每专家 1 层；
- expert pitch：254，active：478，FFT canvas：518；
- 532 nm，仿真像素 16 um；
- expert 到 OEO、OEO 到 global、global 到 CCD 均为 5 cm；
- phase 为 `2*pi*sigmoid(raw_phase)`，raw phase 零初始化，对应初始相位 pi；
- K-space 约束开启，`theta_max_deg=0.65`；
- 2x2 CCD integration；
- DC loss 和 phase dropout 关闭。

## Loss

    L = 8.0 * embedding KD
      + 0.10 * relational KD
      + 1.0 * supervised contrastive retrieval
      + 0.25 * student gallery-prototype CE
      + 0.10 * teacher-gallery alignment
      + 0.02 * router balance
      + 0.005 * router importance
      + 0.10 * routing-response consistency

所有权重均由 YAML 控制。PK batch 默认 `10 classes x 3 images = 30`，确保每个 anchor
始终存在至少两个同类正样本。

## 两阶段 checkpoint 兼容性

模型没有依赖类别数的分类层。阶段一的 50 类和阶段二的 10 类只改变 sampler、manifest、
gallery 与损失样本，因此 Vision/Language optics、routers、adapters 和 64-D readout 可完整续训。
阶段二重置 AdamW 状态，并使用更小的 phase/router/adapter/readout 学习率。

