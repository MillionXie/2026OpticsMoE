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

## Regularized continuation

The from-scratch config deliberately oversamples each logged epoch. Once its
training retrieval approaches saturation, stop it and continue the latest EMA
weights with the natural-length, electronics-only regularization config:

```bash
RUN_ROOT=experiments/qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval/runs
SOURCE="$RUN_ROOT/robust_hybrid_moe4_from_scratch"
TARGET="$RUN_ROOT/robust_hybrid_moe4_regularized_continuation"
mkdir -p "$TARGET/teacher_cache"
cp "$SOURCE/teacher_cache/teacher_embeddings.pt" "$TARGET/teacher_cache/"
cp "$SOURCE/teacher_cache/metadata.json" "$TARGET/teacher_cache/"

python -m experiments.qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval/configs/continuation/regularized_electronic_finetune.yaml \
  --phase train \
  --resume-checkpoint "$SOURCE/ema_last_checkpoint.pt"
```

This continuation freezes phase/router tensors, reduces electronic learning
rates, enables weight decay and stronger image/dropout augmentation, uses only
the natural number of PK batches, and emphasizes frozen-Teacher targets over
the current Student gallery. Use a chronologically chosen `ema_last` checkpoint
rather than a test-selected `ema_best_observed_test` checkpoint for an unbiased
comparison.

## Recommended full retraining

Grocery10 contains only 306 training images. The checked-in full retraining
route first uses all 31 packaged GroceryStore SKUs (964 train images), then
adapts to the target ten SKUs. Both stages use natural-length epochs, fixed
training schedules, nonzero AdamW learning rates and no per-epoch test peeking.

All commands are in one executable file:

```bash
GPU_ID=1 bash \
  experiments/qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval/commands/01_retrain_grocery31_to_grocery10.sh
```

Stage 1 uses `1e-4` for electronics, `5e-5` for router/phase and 60 natural
epochs. Stage 2 uses `5e-5` electronics, `2e-5` router, `1e-5` phase and 20
natural epochs. The script reuses compatible Teacher caches when present,
refuses to overwrite an existing checkpoint, evaluates only fixed final EMA
checkpoints, and writes separate logs/output directories for both stages.
