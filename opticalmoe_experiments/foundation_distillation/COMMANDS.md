# Commands

Run these commands from `opticalmoe_experiments/`.

The student configs use `fast120_520` (`520` canvas, `120` input/expert,
`450` prompt/global-FC window). Teacher preprocessing is backend-specific and
does not change student geometry.

## CIFAR10

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py 
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_moe.yaml 
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

## End-to-end MoE baseline

CIFAR10 smoke test (no teacher cache required):

```bash
python foundation_distillation/scripts/train_end_to_end_moe.py \
  --config foundation_distillation/configs/cifar10_gray_end_to_end_moe.yaml \
  --run_name cifar10_gray_end_to_end_moe_smoke \
  --epochs 1 \
  --smoke_test \
  --device cuda
```

## DINOv2 Feature Distillation

Install the optional backend:

```bash
pip install transformers
```

Build the CIFAR10 DINOv2 cache:

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py 
  --config foundation_distillation/configs/cifar10_gray_dinov2_vits14_feature_distill_moe.yaml 
  --device cuda
```

Train CIFAR10 with cached DINOv2 features:

```bash
python foundation_distillation/scripts/train_feature_distilled_moe.py 
  --config foundation_distillation/configs/cifar10_gray_dinov2_vits14_feature_distill_moe.yaml 
  --run_name cifar10_gray_dinov2_vits14_feature_distill_seed7 
  --device cuda
```

Build the Imagenette DINOv2 cache:

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py \
  --config foundation_distillation/configs/imagenette_gray_dinov2_vits14_feature_distill_moe.yaml \
  --device cuda
```

Train Imagenette with cached DINOv2 features:

```bash
python foundation_distillation/scripts/train_feature_distilled_moe.py \
  --config foundation_distillation/configs/imagenette_gray_dinov2_vits14_feature_distill_moe.yaml \
  --run_name imagenette_gray_dinov2_vits14_feature_distill_seed7 \
  --device cuda
```

Build and train the DINOv2-base CIFAR10 variant:

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py \
  --config foundation_distillation/configs/cifar10_gray_dinov2_vitb14_feature_distill_moe.yaml \
  --device cuda

python foundation_distillation/scripts/train_feature_distilled_moe.py \
  --config foundation_distillation/configs/cifar10_gray_dinov2_vitb14_feature_distill_moe.yaml \
  --run_name cifar10_gray_dinov2_vitb14_feature_distill_seed7 \
  --device cuda
```

Build and train the DINOv2-base Imagenette variant:

```bash
python foundation_distillation/scripts/build_teacher_feature_cache.py \
  --config foundation_distillation/configs/imagenette_gray_dinov2_vitb14_feature_distill_moe.yaml \
  --device cuda

python foundation_distillation/scripts/train_feature_distilled_moe.py \
  --config foundation_distillation/configs/imagenette_gray_dinov2_vitb14_feature_distill_moe.yaml \
  --run_name imagenette_gray_dinov2_vitb14_feature_distill_seed7 \
  --device cuda
```

## Teacher Feature Probe

The matched MLP probe reads the existing cache and does not load the teacher encoder:

```bash
python foundation_distillation/scripts/train_teacher_feature_probe.py 
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_moe.yaml 
  --probe_type matched_mlp 
  --run_name cifar10_clip_vitb32_teacher_matched_mlp_probe 
  --device cuda
```

## LeNet Feature-Distillation Diagnostic

This baseline reuses the existing CIFAR10-gray CLIP cache. Formal training:

```bash
python foundation_distillation/scripts/train_lenet_feature_distilled.py 
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_lenet.yaml 
  --run_name cifar10_gray_clip_vitb32_feature_distill_lenet_seed7 
  --device cuda
```

Smoke test:

```bash
python foundation_distillation/scripts/train_lenet_feature_distilled.py \
  --config foundation_distillation/configs/cifar10_gray_clip_vitb32_feature_distill_lenet.yaml \
  --run_name cifar10_gray_clip_vitb32_feature_distill_lenet_smoke \
  --epochs 1 \
  --smoke_test \
  --device cuda
```

CIFAR10 formal baseline:

```bash
python foundation_distillation/scripts/train_end_to_end_moe.py \
  --config foundation_distillation/configs/cifar10_gray_end_to_end_moe.yaml \
  --run_name cifar10_gray_end_to_end_moe_seed7 \
  --device cuda
```

Imagenette formal baseline:

```bash
python foundation_distillation/scripts/train_end_to_end_moe.py \
  --config foundation_distillation/configs/imagenette_gray_end_to_end_moe.yaml \
  --run_name imagenette_gray_end_to_end_moe_seed7 \
  --device cuda
```

Evaluate a baseline checkpoint:

```bash
python foundation_distillation/scripts/evaluate_end_to_end_moe.py \
  --run_dir foundation_distillation/runs/cifar10_gray_end_to_end_moe_seed7 \
  --checkpoint best.pt \
  --device cuda
```

Plot CIFAR10 distilled-versus-baseline accuracy after rebuilding the master tables:

```bash
python foundation_distillation/visualization/plot_distillation_vs_baseline.py \
  --master_csv foundation_distillation/results/master_distillation_final_metrics.csv \
  --dataset cifar10 \
  --out foundation_distillation/figures/cifar10_distillation_vs_baseline.png
```
