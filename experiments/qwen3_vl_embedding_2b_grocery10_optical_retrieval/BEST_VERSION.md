# Saved-version audit

## Recommended deployable MoE4

- Architecture: 2×2, four experts, Top‑2; one expert phase and one global phase per Vision/Language stack.
- Checkpoint: `runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_replay/ema_last_checkpoint.pt`.
- Fixed checkpoint epoch: absolute epoch 40 EMA.
- Full evaluation (260 test query + 10 gallery): Top‑1 0.676923, Top‑3 0.873077, MRR 0.791590.
- Training: `configs/optimization/grocery31_moe4_pretrain.yaml` followed by `configs/optimization/grocery10_moe4_from31_strong_ema.yaml`.
- Hardware: `configs/grocery10_moe4_from31_hardware.yaml`.

The original live log once showed Top‑1 0.696154 at epoch 40, but that live parameter snapshot was not retained. It must not replace the reproducible 0.676923 result from the saved EMA checkpoint.

## Historical MoE16 upper result

- Architecture: 4×4, 16 experts; not compatible with MoE4 masks/checkpoints.
- Checkpoint: `runs/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/ema_best_train_loss_checkpoint.pt`.
- Result: Top‑1 0.734615, Top‑3 0.919231, MRR 0.836157.

## MoE4 from-scratch baseline

- Checkpoint: `runs/qwen3_vl_embedding_2b_grocery10_moe4_hardware_robust/best_train_loss_checkpoint.pt`.
- Result: Top‑1 0.542308, Top‑3 0.861538, MRR 0.710127.

CCD captures produced with one checkpoint's phase masks cannot be reused as though they came from another checkpoint's masks.
