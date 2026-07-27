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

After the 31-SKU Teacher cache exists, screen two replacements on the official
validation-source images only (never on test):

```bash
python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.select_replacement_skus --all-sku-config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery31_pretrain.yaml --target-config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10.yaml --drop-sku Garant-Ecological-Standard-Milk --drop-sku Bravo-Apple-Juice
```

Fine-tune the replacement 10-SKU subset from the 31-SKU checkpoint. The new
output directory receives an independent manifest and Teacher cache:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_finetune.yaml --phase all --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery31_optical_pretrain/best_train_loss_checkpoint.pt
```

Generalization continuation from the replacement-10 epoch-141 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_continue_epoch141_augmented_kd.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/best_train_loss_checkpoint.pt
```

Diagnose one-view versus three-view gallery coverage without changing weights:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.analyze_gallery_coverage --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_finetune.yaml --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/best_train_loss_checkpoint.pt --additional-gallery-per-sku 2
```

Run the same continuation with a ten-times smaller optical-phase learning rate:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_continue_epoch141_augmented_kd_phase_slow.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/best_train_loss_checkpoint.pt
```

Run the pairwise Teacher-geometry distillation continuation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_continue_epoch141_augmented_relational_kd.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/best_train_loss_checkpoint.pt
```

Run the frozen Teacher-gallery anchor continuation:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_continue_epoch141_teacher_gallery_anchor.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/best_train_loss_checkpoint.pt
```

Run the stronger packaging-safe augmentation continuation:

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_replaced_continue_epoch141_stronger_augmentation.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_finetune/best_train_loss_checkpoint.pt
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
