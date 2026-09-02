# Commands

Run from the repository root. Four training commands can use four different
GPUs only after the common initialization command has completed.

## 1. Prepare and audit the split

```bash
python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk2.yaml \
  --phase prepare_data
```

Expected counts are `train=10228`, `dev=200`, `sealed_test=1000`.

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

The log must print `test=SEALED` on every epoch. Training must not create
`sealed_test_evaluation.json` or `metrics/sealed_test.json`.

## 4. Explicit final test once per completed variant

Example for E1:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/configs/release/electronic_power_topk1.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/runs/electronic_power_topk1/checkpoints/ema_best_development_pck.pt
```

Replace both `electronic_power_topk1` occurrences with the matching E2, E4 or
O2 run. Never use a test result to choose another epoch or edit a hyperparameter.
