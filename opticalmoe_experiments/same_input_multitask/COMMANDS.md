# Same-Input Multitask Commands

Run commands from `opticalmoe_experiments/`.

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
