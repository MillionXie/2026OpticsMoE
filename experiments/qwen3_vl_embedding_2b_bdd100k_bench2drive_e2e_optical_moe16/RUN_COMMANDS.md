# Commands

All commands are run from the `2026OpticsMoE` repository root.

## Dependency-free smoke

```bash
python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive_smoke.yaml --phase smoke
```

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/tests -q
```

## Data validation

```bash
python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase prepare_data
```

## BDD100K Optical Backbone pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase fit_pca
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase bdd_pretrain
```

## Bench2Drive behavior cloning

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase bc_all
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase bc_evaluate
```

## Closed-loop SAC

Install CARLA 0.9.15 and Bench2Drive, set `sac.env_factory`, and then run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 --config experiments/qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16/configs/bench2drive.yaml --phase sac_train
```

CARLA rendering GPU is selected by CARLA's `-graphicsadapter` argument; `CUDA_VISIBLE_DEVICES` only controls PyTorch.
