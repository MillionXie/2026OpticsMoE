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
→ Vision DeepStack Optical MoE16 (4 stages)
→ frozen vision merger and native three-point DeepStack injection
→ frozen token embedding + multimodal injection
→ Language Optical MoE16 (4 stages)
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
16 experts × 224×224 phase per stage
4 optical/OEO stages
986×986 global phase active area
1026×1026 propagation canvas
986×986 physical CCD ROI
AdaptiveAvgPool → 224×224 token-row readout
```

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
