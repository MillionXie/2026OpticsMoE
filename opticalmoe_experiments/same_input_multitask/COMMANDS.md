# Same-Input Multitask Commands

Run commands from `opticalmoe_experiments/`.

## Runtime Notes

- Formal configs default to `num_workers=16`, `pin_memory=auto`, `persistent_workers=true`, and `prefetch_factor=4`.
- On Linux servers this is the recommended starting point. If CPU or RAM is saturated, reduce `num_workers` to `8` or `4`.
- On Windows or while debugging, set `num_workers=0`.
- `--smoke_test` automatically forces `num_workers=0`, `persistent_workers=false`, and `prefetch_factor=null`.
- The optical canvas is `1000 x 1000`, but the default trainable global FC phase window is center `600 x 600`; the padding is transparent and not trainable.
- dSprites configs default to `sampling_protocol.enabled=true` and `total_size=12000`.
- With `train_test_ratio=[4,1]` and `val_split=0.1`, this gives `train=8640`, `val=960`, `test=2400`.
- Use `max_train_samples`, `max_val_samples`, and `max_test_samples` for exact split caps.
- `batch_mode=paired_same_input` means the same image batch is reused for all tasks in one optimizer update.

## Audit Dataset Config Fields

```bash
python scripts/audit_dataset_config_fields.py
```

Audit task-specific head fields:

```bash
python scripts/audit_head_config_fields.py
```

## Stage 1: Shape + Scale

Smoke test:

```bash
python same_input_multitask/scripts/train_same_input_multitask.py \
  --config same_input_multitask/configs/dsprites_shape_scale_learnable_moe_E9_complex.yaml \
  --run_name simt_dsprites_shape_scale_learnable_smoke \
  --epochs 1 \
  --smoke_test \
  --device cuda
```

Formal learnable MoE:

```bash
python same_input_multitask/scripts/train_same_input_multitask.py \
  --config same_input_multitask/configs/dsprites_shape_scale_learnable_moe_E9_complex.yaml \
  --run_name simt_dsprites_shape_scale_learnable_E9_seed7 \
  --device cuda
```

Fixed uniform MoE:

```bash
python same_input_multitask/scripts/train_same_input_multitask.py \
  --config same_input_multitask/configs/dsprites_shape_scale_fixed_uniform_moe_E9_complex.yaml \
  --run_name simt_dsprites_shape_scale_fixed_uniform_E9_seed7 \
  --device cuda
```

Shared D2NN:

```bash
python same_input_multitask/scripts/train_same_input_multitask.py \
  --config same_input_multitask/configs/dsprites_shape_scale_shared_d2nn.yaml \
  --run_name simt_dsprites_shape_scale_shared_d2nn_seed7 \
  --device cuda
```

## Stage 2: Add x Position 4-Bin

```bash
python same_input_multitask/scripts/train_same_input_multitask.py \
  --config same_input_multitask/configs/dsprites_shape_scale_xpos4_learnable_moe_E9_complex.yaml \
  --run_name simt_dsprites_shape_scale_xpos4_learnable_E9_seed7 \
  --device cuda
```

## Stage 3: Add y Position 4-Bin

```bash
python same_input_multitask/scripts/train_same_input_multitask.py \
  --config same_input_multitask/configs/dsprites_shape_scale_xpos4_ypos4_learnable_moe_E9_complex.yaml \
  --run_name simt_dsprites_shape_scale_xpos4_ypos4_learnable_E9_seed7 \
  --device cuda
```

## Evaluation

Prompt swap only:

```bash
python same_input_multitask/scripts/run_prompt_swap_eval.py \
  --run_dir same_input_multitask/runs/simt_dsprites_shape_scale_learnable_E9_seed7 \
  --checkpoint best.pt \
  --device cuda
```

Full evaluation:

```bash
python same_input_multitask/scripts/evaluate_same_input_multitask.py \
  --run_dir same_input_multitask/runs/simt_dsprites_shape_scale_learnable_E9_seed7 \
  --checkpoint best.pt \
  --device cuda
```

## Tables and Figures

Rebuild master tables:

```bash
python same_input_multitask/scripts/build_same_input_multitask_tables.py \
  --runs_dir same_input_multitask/runs \
  --out_dir same_input_multitask/results
```

Plot prompt swap:

```bash
python same_input_multitask/visualization/plot_prompt_swap_matrix.py \
  --master_dir same_input_multitask/results \
  --run_id simt_dsprites_shape_scale_learnable_E9_seed7 \
  --out_dir same_input_multitask/figures/reports/shape_scale
```
