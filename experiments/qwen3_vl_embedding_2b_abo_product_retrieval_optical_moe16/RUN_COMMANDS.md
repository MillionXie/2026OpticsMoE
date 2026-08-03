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
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo500.yaml --phase all
```

## 3,000-item formal experiment

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase all
```

## 5,000-item experiment

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo5000.yaml --phase all
```

## Individual phases

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase cache_teacher
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase train_stage1
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase train_stage2
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase evaluate
```

```bash
python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000.yaml --phase visualize
```

## Improved Vision + Language Optical MoE

The improved configuration uses an offline product-adapted frozen-Qwen target,
Vision and Language Optical MoE16, cross-batch contrastive memory, and explicit
router-collapse penalties. It writes to independent cache/run directories.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo_multimodal_smoke.yaml --phase all
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase all
```

Individual phases, when resuming manually:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase cache_teacher
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase train_teacher_adapter
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase train_stage1
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase train_stage2
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo3000_multimodal.yaml --phase evaluate
```
