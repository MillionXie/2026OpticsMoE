# Run commands

All commands are single-line commands.

## Prepare and validate KADID metadata

```bash
python -m experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual/configs/kadid10k_quality3_smoke.json --phase prepare_data
```

## Precompute teacher outputs

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual/configs/kadid10k_quality3_smoke.json --phase teacher_precompute
```

## Train student

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual/configs/kadid10k_quality3_smoke.json --phase student_train
```

## Full experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual/configs/kadid10k_quality3.json --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual/tests -q
```
