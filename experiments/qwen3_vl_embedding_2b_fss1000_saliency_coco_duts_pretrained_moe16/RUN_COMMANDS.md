# Commands

Run from the repository root.

Prepare/verify FSS-1000:

```bash
python -m experiments.qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16 \
  --config experiments/qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16/configs/fss1000_finetune.yaml \
  --phase prepare_data
```

Verify the transferred checkpoint and tensor shapes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16 \
  --config experiments/qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16/configs/fss1000_finetune.yaml \
  --phase shape_smoke
```

Complete fine-tuning and final evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16 \
  --config experiments/qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16/configs/fss1000_finetune.yaml \
  --phase all
```

Re-evaluate the train-loss-selected checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16 \
  --config experiments/qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16/configs/fss1000_finetune.yaml \
  --phase test
```

Tests:

```bash
pytest experiments/qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16/tests -q
```

