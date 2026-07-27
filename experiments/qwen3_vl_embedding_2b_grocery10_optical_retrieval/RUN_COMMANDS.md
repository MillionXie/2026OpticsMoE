# Commands

All commands below are run from the `2026OpticsMoE` repository root. They are intentionally one line each.

## Smoke

```bash
python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_smoke.yaml --phase all
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml --phase all
```

## 31 packaged-SKU pretraining

The command keeps the same one-expert-phase plus one-global-phase Student and
pretrains it on all packaged SKUs. It uses a separate output directory and
teacher cache.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery31_pretrain.yaml --phase all
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery31_pretrain_smoke.yaml --phase all
```

## Phases

Prepare/download the official Grocery Store Dataset and write the fixed subset manifest:

```bash
python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml
```

Cache frozen 64-D Teacher embeddings:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml
```

Train the optical retrieval Student:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml
```

Evaluate Teacher, Student, and cross-space diagnostic retrieval:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.evaluate_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml
```

Regenerate result figures from the saved prediction CSV:

```bash
python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.visualize_retrieval_results --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml
```

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
