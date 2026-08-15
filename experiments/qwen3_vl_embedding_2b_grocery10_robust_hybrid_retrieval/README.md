# Robust hybrid optical retrieval

This is a new experiment derived from
`qwen3_vl_embedding_2b_grocery10_optical_retrieval`. It keeps the same frozen
Qwen3-VL-Embedding-2B teacher, Grocery10 split, MoE4 geometry and retrieval
objective, but moves capacity toward compact electronics and explicitly trains
against optical registration error.

The baseline is not modified. Shared dataset, teacher-cache, evaluation and
training utilities are imported from it so that the new architecture remains
directly comparable without copying data-pipeline code.

## Main changes

- Two learnable residual gates per modality: one after the expert phase/CCD/OEO
  segment and one after the global phase/final CCD segment. Each is a sigmoid
  convex input/optical mixture, initialized to 80% electronic input.
- A depthwise-separable local electronic refiner follows every mixture. Its
  parameter count does not scale with the 224x224 pixel count; no flattened
  50,176-pixel MLP is used.
- Independent amplitude-input, phase-mask and CCD translations are sampled up
  to 12 logical pixels during training. At 16 um simulation sampling and 2x
  hardware export, this covers up to 24 physical 8 um pixels.
- Block phase bypass dropout is enabled at 5%, and the existing 0.65-degree
  k-space constraint remains enabled.
- Phase LR is `1e-4`: 40x below the baseline from-scratch `4e-3` and 5x below
  the released Grocery10 continuation `5e-4`.
- The 64-D head becomes a gated 96-D bottleneck residual followed by the signed
  linear projection and L2 normalization. It remains below 100k parameters.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the baseline audit, equations, loss
definition and server results.

## Commands

Run from the repository root:

```bash
python -m experiments.qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval/configs/release/robust_hybrid_moe4.yaml \
  --phase all
```

Quick checks:

```bash
python -m pytest \
  experiments/qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval/tests -q
```

This configuration starts a new model. A baseline checkpoint cannot be loaded
strictly because the residual refiners and embedding head add new state.
