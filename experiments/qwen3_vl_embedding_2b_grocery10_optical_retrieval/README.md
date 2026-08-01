# Grocery10 optical image retrieval

This experiment retrieves one of ten packaged grocery products from a fixed
gallery. It uses a frozen Qwen3-VL-Embedding-2B teacher and a trainable optical
Student. It is a metric-retrieval task, not a ten-class classifier.

Chinese summary: 本工程将自然拍摄商品图和标准 gallery 图编码为 64 维 L2 归一化
向量，并用余弦相似度完成 Top-1/Top-3 检索。当前维护版本同时包含 Vision 和
Language 两套 Optical MoE，不是 Vision-only。

## Maintained result

The strongest reproducible saved checkpoint is the epoch-159 EMA checkpoint
from the three-stage Grocery31 → replacement Grocery10 → strong-augmentation
continuation. On the fixed 260-query/10-gallery split:

| system | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|
| Frozen Teacher | 90.77% | 99.23% | 94.81% |
| Optical Student | 73.46% | 91.92% | 83.62% |

The training log briefly reached 74.23% at epoch 152, but no checkpoint for
that exact epoch was retained. The canonical claim therefore uses the saved
epoch-159 EMA checkpoint. See `BEST_VERSION.md` for the complete audit.

## Student architecture

Both stacks use the same physical geometry but independent parameters:

```text
hidden tokens
→ Linear(D,224) → LayerNorm(224) → Softplus
→ zero-pad token rows to 224×224
→ electronic Top-4 router
→ directly load weighted copies into a 4×4 expert amplitude mosaic
→ one 224×224 phase-only layer per expert (16 masks)
→ 10 cm angular-spectrum propagation
→ square-law detector
→ per-expert LayerNorm → ReLU
→ reapply the same routing weights and zero unselected experts
→ one 986×986 global phase
→ 10 cm propagation
→ 986×986 CCD ROI
→ adaptive pooling to 224×224 → LayerNorm → ReLU
→ Linear(224,D) and Transformer-style residual
```

Vision uses `D=1024`; Language uses `D=2048`. Frozen Qwen patch/token
embeddings, vision merger, one native DeepStack visual injection, multimodal
token injection, and final RMSNorm remain in the path. The last valid Language
detector row is read by:

```text
LayerNorm(224) → Linear(224,64) → L2 normalization
```

There is no activation after the 64-D linear layer. Total trainable Student
parameters are 4,951,848.

## Physical geometry

* wavelength: 532 nm
* pixel pitch: 8 µm
* expert: 224×224 pixels
* layout: 4×4, pitch 254, gap 30
* active footprint: 986×986
* propagation canvas: 1026×1026 (20-pixel numerical guard each side)
* expert/global propagation distance: 10 cm each
* one expert phase plane + one global phase plane in each stack

The ideal amplitude-to-phase 4f relay is represented as co-planar amplitude and
phase modulation; no additional numerical propagation is inserted for the
relay.

## Training

The frozen Teacher produces official 64-D Matryoshka embeddings. The Student
optimizes cosine embedding KD, supervised contrastive retrieval, optional
gallery alignment, and router regularization. Packaging-safe augmentation never
uses horizontal flips, MixUp, CutMix, strong blur, or random erasing.

Canonical configs:

* `grocery10_best_reproduction.yaml`: complete three-stage pipeline;
* `grocery10_best_reproduction_stage1_grocery31.yaml`;
* `grocery10_best_reproduction_stage2_replaced10.yaml`;
* `grocery10_best_reproduction_stage3_strong_ema.yaml`;
* `grocery10_hardware_deployment.yaml`: real SLM/CCD export and replay;
* `grocery10.yaml` / `grocery10_smoke.yaml`: base and smoke definitions.

Historical exploratory configs and failed run directories were removed to keep
the experiment unambiguous.

## Hardware deployment

The trained network requires four physical exposures per sample: Vision expert,
Vision global, Language expert, and Language global. Phase masks are shared
across all samples, so acquisition is organized plane-first as four folder
playbacks. See `HARDWARE_DEPLOYMENT.md` for:

* exact amplitude/phase BMP dimensions and centering;
* original images, 224×224 token fields, masks, captured CCD intensities,
  electronic reload amplitudes, and simulation complex-field references;
* required CCD file formats and filenames;
* the four electronic processing commands;
* final hardware embedding/retrieval evaluation.

Commands are collected in `RUN_COMMANDS.md`.
