# Run commands

Run all commands from the repository root.

## Download and validate the official dataset

```bash
python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch.yaml \
  --phase prepare_data
```

## Shape smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch_smoke.yaml \
  --phase shape_smoke
```

## End-to-end from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch.yaml \
  --phase all
```

## COCO -> DUTS pretrained transfer

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_coco_duts_pretrained.yaml \
  --phase all
```

## Fastest recovery from the completed head-only run

This keeps the already ISIC-adapted segmentation head, resets the optimizer,
and genuinely unfreezes the optical core, router and recombiner:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_resume_previous_pretrained_unfreeze.yaml \
  --phase all
```

## Re-evaluate the selected checkpoint

Replace the config with either formal config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch.yaml \
  --phase test
```

## Unit tests

```bash
pytest experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/tests -q
```
