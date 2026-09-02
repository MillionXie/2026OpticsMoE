# Commands

Run from the repository root. Four training commands can use four different
GPUs only after the common initialization command has completed.

## 1. Prepare and audit the split

```bash
python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk2.yaml \
  --phase prepare_data
```

Expected counts are `train=10428`, `validation=0`, `periodic_test=1000`.

## 2. Generate the common untrained body/head exactly once

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk2.yaml \
  --phase materialize_initialization
```

This writes `runs/shared_untrained_initialization.pt`. It contains no Router
weights. Do not regenerate it between variants.

## 3. Train the predeclared single-seed matrix

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk1.yaml --phase train
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk2.yaml --phase train
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk4.yaml --phase train
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/optical_power_topk2.yaml --phase train
```

The log evaluates test at epoch 1, every 5 epochs, and epoch 100. Other epochs
print `test=SKIPPED(interval=5)`. Each run writes its best EMA checkpoint to:

```text
runs/<variant>_periodic_test5/checkpoints/ema_best_periodic_test_pck.pt
```

Selection is maximum test PCK@0.2, then lower test NME, lower test loss, and
earlier epoch. These output directories differ from the older development-set
runs, so the two protocols cannot overwrite one another.

## 4. Optional detailed evaluation of the selected checkpoint

Example for E1:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk1.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/runs/electronic_power_topk1_periodic_test5/checkpoints/ema_best_periodic_test_pck.pt
```

Replace the config and run directory with the matching E2, E4 or O2 variant.
Training already used periodic test for checkpoint selection; this command is
only needed when per-sample predictions/figures from the selected checkpoint
are required.
