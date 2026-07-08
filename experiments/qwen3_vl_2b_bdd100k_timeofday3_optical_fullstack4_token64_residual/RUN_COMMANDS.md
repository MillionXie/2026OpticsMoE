# Run commands

All commands are single-line commands.

## Prepare data

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/configs/bdd100k_timeofday3.json --phase prepare_data
```

## Smoke phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/configs/bdd100k_timeofday3_smoke.json --phase teacher_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/configs/bdd100k_timeofday3_smoke.json --phase student_train
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/configs/bdd100k_timeofday3.json --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/tests -q
```

## Head ablations

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/configs/bdd100k_timeofday3_bottleneck64.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/configs/bdd100k_timeofday3_linear.json --phase all
```
