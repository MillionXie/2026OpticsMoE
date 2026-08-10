# Nonshared token-wise MoE optimization

## Baseline diagnosis

The completed nonshared run is preserved under
`runs/qwen3_vl_embedding_2b_grocery10_tokenwise_vision_language_moe4_nonshared_8um`.

Observed baseline:

- frozen Qwen teacher: Top-1 90.77%, Top-3 99.62%, MRR 95.10%;
- nonshared student: Top-1 30.38%, Top-3 58.85%, MRR 50.01%;
- final train-batch Top-1: 99.75%;
- train-loss minimum at epoch 99 gave only 23.08% test Top-1.

This is severe image-level overfitting rather than insufficient optimization.
The router remained active (last-batch normalized entropy: Vision 0.768,
Language 0.512), so simply increasing the balance coefficient is not the main
remedy.

## Recorded optimization recipe

Stage 1 pretrains the exact same nonshared Vision+Language optical architecture
on 31 packaged GroceryStore SKUs. Stage 2 initializes model weights from the
stage-1 EMA checkpoint, resets the optimizer and epoch counter, and adapts to
the target ten SKUs.

Changes relative to the baseline:

1. 31-SKU packaged-product pretraining without target-test image leakage;
2. P=10, K=2 batches, giving every anchor more negative identities;
3. pointwise cosine KD weight increased from 1 to 8;
4. pairwise relational KD over the batch embedding geometry;
5. cross-entropy against frozen teacher gallery prototypes;
6. EMA model weights with decay 0.99;
7. stronger but packaging-safe crop/brightness/contrast/rotation augmentation;
8. 3% block phase bypass only during target fine-tuning;
9. lower target-stage learning rates and a fresh optimizer.

The student architecture, physical 8-µm geometry, nonshared position experts,
top-2 router, two optical planes, output embedding dimension and deployment
path are unchanged.

For efficiency, the 31-SKU pretraining test set is evaluated only after the
stage completes because transfer always uses EMA-last. During target-10
fine-tuning, test is still observed each epoch per the existing project
convention. EMA best-observed-test is explicitly marked selection-biased; the
EMA-last checkpoint is also retained for a non-test-selected reference.
