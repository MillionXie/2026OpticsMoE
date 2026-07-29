# Architecture and protocol

## Immutable data protocol

```text
official ABO listing/image metadata
→ remove ambiguous image IDs shared by multiple item IDs
→ choose high-quality Stage-2 item IDs
→ deterministic per-item 60/20/20 image split
→ freeze Gallery and Query image IDs
→ build Stage-1 pool while excluding every frozen Gallery/Query image
→ persist manifest + SHA256
```

The invariant is checked every time the manifest is loaded:

```text
(stage1_train ∪ stage2_train) ∩ (gallery ∪ query) = ∅
```

Identity overlap is intentional:

```text
item_ids(stage2_train) = item_ids(gallery) = item_ids(query)
```

## Stage 1

```text
60k–100k non-held-out ABO views
├─ frozen Qwen3-VL-Embedding-2B → normalized 224D Teacher target (cached)
└─ frozen Qwen patch/position stem
   → Optical MoE16 (one expert stage)
   → global phase + CCD
   → token-wise LN + Linear(224,224)
   → valid-token mean + L2
   → normalized 224D Student
```

PK batches guarantee at least two views of every sampled `item_id`, so every
SupCon anchor has a positive.

## Stage 2

The Stage-1 encoder checkpoint is loaded, the optimizer is reset, and a temporary
item classifier is added:

```text
224D normalized Student embedding
├─ retrieval losses
└─ temporary Linear(224,N_items) → ID cross entropy
```

The classifier is stored only in training checkpoints. The deployment artifact
contains only:

- optical input adapter/norm;
- electronic router;
- expert/global phase parameters;
- OEO normalization parameters;
- CCD token LayerNorm and Linear(224,224).

## Evaluation

Each held-out Gallery/Query image is encoded independently. For
`mean_prototype`, normalized Gallery embeddings from the same item are averaged
and normalized again. Every Query ranks all 3,000–5,000 item prototypes by
cosine similarity.

The following spaces are evaluated:

1. Teacher Query vs Teacher Gallery;
2. Student Query vs Student Gallery (main deployment result);
3. Student Query vs Teacher Gallery (alignment diagnostic).

