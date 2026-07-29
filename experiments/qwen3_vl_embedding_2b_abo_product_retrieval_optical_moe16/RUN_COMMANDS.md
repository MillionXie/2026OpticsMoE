# Commands

All commands are intended to run from the repository root. They contain no shell line-continuation backslashes.

## Unit smoke

```bash
pytest experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/tests -q
```

## Download and freeze the split

```bash
python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo500.yaml --phase prepare_data
```

## 500-item validation experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo500.yaml --phase all
```

## 3,000-item formal experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase all
```

## 5,000-item experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo5000.yaml --phase all
```

## Individual phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase cache_teacher
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase train_stage1
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase train_stage2
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase evaluate
```

```bash
python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase visualize
```

