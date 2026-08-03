# Run commands

All commands are run from the `2026OpticsMoE` repository root. Commands are
single-line commands and do not use backslash continuation.

## Fine-tune on 100 fixed seen categories

This protocol selects 100 eligible classes from the original official test
pool, uses 8 images per class for adaptation, and reserves 2 different images
per class for testing. The command reuses the preserved one-layer Student and
Teacher checkpoints, builds a split-specific mask cache, fine-tunes all
optical/router/head parameters for 50 epochs, and writes results to a new run.

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_seen100_finetune.yaml --phase all
```

Prepare and inspect the exact 800/200 image split without loading Qwen:

```bash
python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_seen100_finetune.yaml --phase prepare_data
```

## Final reproducible one-layer Student run

This command starts a new Optical Student from random/zero initialization,
trains for 100 epochs, evaluates the minimum-training-loss checkpoint, and
saves prediction and optical-path visualizations automatically.

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_single_layer_from_scratch_100ep.yaml --phase student_train
```

## Data preparation

```bash
python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase prepare_data
```

## Shape smoke

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_smoke.yaml --phase shape_smoke
```

## Rebuild the reusable Teacher mask cache

The preserved formal run already contains this cache. Run these commands only
if that cache was deliberately removed.

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase teacher_train
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_mask_kd.yaml --phase cache_teacher_masks
```

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/tests -q
```
