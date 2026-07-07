# Run commands

All commands are single-line commands without continuation backslashes.

## Prepare Scene-4 ImageFolder and reports

```bash
python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4.json --phase prepare_data
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4_smoke.json --phase all
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4.json --phase all
```

## Teacher precompute

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4.json --phase teacher_precompute
```

## Teacher MLP training

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4.json --phase teacher_train
```

## Student training

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4.json --phase student_train
```

## Student inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.run --config experiments/qwen3_vl_2b_bdd100k_scene4_optical_fullstack4/configs/bdd100k_scene4.json --phase student_inference
```
