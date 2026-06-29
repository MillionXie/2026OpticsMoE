# Commands

Run these commands from `opticalmoe_experiments/`.

## CIFAR10

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py \
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_moe.yaml \
  --device cuda
```

```bash
python foundation_distillation/scripts/train_feature_distilled_moe.py \
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_moe.yaml \
  --run_name cifar10_gray_clip_feature_distill_smoke \
  --epochs 1 \
  --smoke_test \
  --device cuda
```

```bash
python foundation_distillation/scripts/train_feature_distilled_moe.py \
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_moe.yaml \
  --run_name cifar10_gray_clip_vitb32_feature_distill_seed7 \
  --device cuda
```

## Imagenette

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py \
  --config foundation_distillation/configs/imagenette_gray_clip_vitb32_feature_distill_moe.yaml \
  --device cuda
```

```bash
python foundation_distillation/scripts/train_feature_distilled_moe.py \
  --config foundation_distillation/configs/imagenette_gray_clip_vitb32_feature_distill_moe.yaml \
  --run_name imagenette_gray_clip_vitb32_feature_distill_seed7 \
  --device cuda
```

## Evaluation And Tables

```bash
python foundation_distillation/scripts/evaluate_feature_distilled_moe.py \
  --run_dir foundation_distillation/runs/cifar10_gray_clip_vitb32_feature_distill_seed7 \
  --checkpoint best.pt \
  --device cuda
```

```bash
python foundation_distillation/scripts/build_distillation_tables.py \
  --runs_dir foundation_distillation/runs \
  --out_dir foundation_distillation/results
```

`--smoke_test` limits training and evaluation to one batch without changing the cached split definition. It also uses `dataset.smoke_batch_size` (default 1) and forces `num_workers=0`.
