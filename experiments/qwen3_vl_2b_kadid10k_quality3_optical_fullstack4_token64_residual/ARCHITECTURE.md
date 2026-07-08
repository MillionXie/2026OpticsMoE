# Architecture and data flow

## Dataset

The loader detects common KADID image, reference, score, distortion-level, and distortion-type column names. Images are resolved in this order:

```text
data_root / image_dir / filename
data_root / filename
```

Labels are generated globally before reference-disjoint splitting. The split sequence is:

```text
all references -> train-pool references + test references
train-pool references -> student-train references + validation references
```

## Multimodal model

```text
distorted image + IQA prompt
  -> Qwen processor and tokenizer
  -> frozen patch embedding
  -> vision token64 residual optical4 [T_v,1024]
  -> frozen vision merger [visual tokens,2048]
  -> original multimodal injection
  -> language token64 residual optical4 [B,S,2048]
  -> frozen final RMSNorm
  -> answer hidden [B,2048]
  -> trainable MLP
  -> logits [B,3]
```

Both optical surrogates use one input adapter, four optical conversions without intermediate electronic adapters, and one output adapter. A `[T,64]` or `[S,64]` tensor is copied directly into the leading rows of a zero-filled 64-by-64 field.

Independent residual scales implement:

```text
vision:   Y_v = beta_v X_v + alpha_v Delta_v
language: Y_l = beta_l X_l + alpha_l Delta_l
```

The default beta values are fixed 1.0 buffers. The default alpha values are trainable parameters initialized to 0.1.

## Classification head

The default KADID three-class head is `LayerNorm(2048) -> Linear(2048,64) -> GELU -> Dropout -> Linear(64,3)`. The head type and dimensions are configurable, shared by teacher and student, recorded in checkpoints, and itemized in `model.json`. Legacy configs without head keys retain the original MLP behavior.

Validation and inference can write a bounded, read-only diagnostic set. It compares restored student hidden against cached teacher targets and records field/hidden statistics. Detector intensity comes directly from square-law detection, normalization, and ReLU, so a negative detector count is treated as a bug and emits a warning.
