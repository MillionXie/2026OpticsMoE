# Architecture

## Teacher path

```text
RGB product image
  + one fixed retrieval instruction
→ official Qwen chat template
→ frozen Qwen3-VL-Embedding-2B vision-language model
→ final valid language-token hidden (2048)
→ first 64 Matryoshka dimensions
→ L2 normalization
→ z_teacher [B,64]
```

Teacher is under `eval()`, `requires_grad_(False)`, and `torch.no_grad()`.
All embeddings are materialized before Student training.

## Student path

```text
RGB product image + same fixed instruction
→ frozen Qwen processor / patch embedding
→ Vision Optical MoE16: one expert phase stage + one global phase
→ frozen vision merger and one native DeepStack auxiliary injection
→ frozen token embedding + multimodal injection
→ Language Optical MoE16: one expert phase stage + one global phase
→ final optical CCD intensity
→ detector LayerNorm/nonlinearity and pool [B,224,224]
→ select last valid token row [B,224]
→ trainable LayerNorm(224)
→ trainable Linear(224,64)
→ L2 normalization
→ z_student [B,64]
```

The detector tensor and final embedding serve different roles:

- detector intensity/readout is nonnegative;
- the post-detector Linear output is signed;
- only the signed 64-D vector is L2-normalized for retrieval.

No activation is allowed after `Linear(224,64)`.

## Optical geometry per vision/language core

```text
Input adapter: hidden D → 224
LayerNorm + Softplus
token rows zero-padded to 224×224
electronic Top-4 router
4×4 homogeneous expert bank
one bank of 16 experts × 224×224 phase-only masks
10 cm propagation
square-law detection → per-expert LayerNorm → ReLU → routed amplitude reload
986×986 global phase active area
10 cm propagation
1026×1026 propagation canvas
986×986 physical CCD ROI
AdaptiveAvgPool → 224×224 token-row readout
```

There are exactly two trainable phase planes in each optical stack:

1. one heterogeneous-in-position but homogeneous-in-type bank containing
   sixteen independent `224×224` expert masks;
2. one shared `986×986` global phase mask.

This experiment deliberately follows the repository's
`qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline`.
It does **not** use the four-stage MoE16 implementation.

## Single auxiliary DeepStack mapping

The electronic Teacher retains Qwen's three native DeepStack auxiliary paths.
The one-layer Student cannot provide three distinct optical stages, so it keeps
only the first native auxiliary route:

```text
Vision:
  patch hidden
  → one expert stage
  → auxiliary detector/readout → frozen deepstack_merger_list[0]
  → global phase + final detector/readout → frozen final vision merger

Language:
  decoder slot 0 is bypassed and receives the one auxiliary injection
  → the sole optical language layer is installed at decoder index 1
  → all later decoder layers are bypassed
```

No single optical tap is copied three times.

The existing output adapter is retained because it feeds the frozen Qwen merger,
DeepStack injection and language replacement path. Retrieval additionally reads
the graph-carrying final language detector tensor directly.

## Trainability boundary

Trainable:

- Vision optical expert/global phase masks;
- Language optical expert/global phase masks;
- existing vision/language input and output adapters;
- electronic Top-K routers;
- configured optoelectronic interlayer affine parameters, if enabled;
- retrieval `LayerNorm(224)` and `Linear(224,64)`.

Frozen:

- complete Teacher;
- Student Qwen tokenizer and processor;
- token embedding;
- patch embedding;
- vision merger;
- native DeepStack machinery;
- final RMSNorm;
- all unreplaced Qwen parameters.

The authoritative parameter inventory is generated at runtime in `model.json`.

## Retrieval, not classification

There is no ten-way output layer. A query is ranked against SKU gallery embeddings
using cosine similarity. SKU labels only define positive pairs in the supervised
contrastive loss and ground truth during evaluation.

## Leakage and cache guards

The subset builder rejects any shared image path across train, test and gallery.
Teacher-cache identity contains:

- subset manifest SHA256;
- exact selected SKU order;
- model ID;
- instruction;
- processor min/max pixels;
- embedding dimension;
- low-dimensional pooling method.

Any mismatch requires an explicit cache rebuild.
