# Run commands

All commands are run from the `2026OpticsMoE` repository root. Commands are
single-line commands and do not use backslash continuation.

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
