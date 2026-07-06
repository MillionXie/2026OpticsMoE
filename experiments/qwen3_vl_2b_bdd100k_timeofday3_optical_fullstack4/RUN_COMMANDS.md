# Run commands

以下命令均为单行，可直接复制，不包含续行反斜杠。

## Prepare data

```bash
python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4/configs/bdd100k_timeofday3.json --phase prepare_data
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4/configs/bdd100k_timeofday3_smoke.json --phase all
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4/configs/bdd100k_timeofday3.json --phase all
```

## Teacher precompute

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4/configs/bdd100k_timeofday3.json --phase teacher_precompute
```

## Student training

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4/configs/bdd100k_timeofday3.json --phase student_train
```
