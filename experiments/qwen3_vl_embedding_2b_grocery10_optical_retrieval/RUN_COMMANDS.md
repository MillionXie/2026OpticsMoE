# Grocery10 commands

All commands run from the `2026OpticsMoE` repository root. Exploratory configs
were removed; the commands below are the maintained entry points.

## Reproduce the strongest saved recipe from scratch

This runs Grocery31 pretraining, replacement-Grocery10 fine-tuning, and the
40-epoch strong-augmentation/EMA continuation, then evaluates and exports the
best optical checkpoint.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.reproduce_best --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction.yaml
```

Use `--dry-run` to print the complete chain without starting it. See
`BEST_VERSION.md` for the audited metrics and checkpoint selection rule.

## Train the three canonical stages manually

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction_stage1_grocery31.yaml --phase all
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction_stage2_replaced10.yaml --phase all --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/grocery10_best_reproduction/01_grocery31_pretrain/best_train_loss_checkpoint.pt
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction_stage3_strong_ema.yaml --phase all --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/grocery10_best_reproduction/02_replaced10_finetune/best_train_loss_checkpoint.pt --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/grocery10_best_reproduction/03_strong_augmentation_ema/ema_best_train_loss_checkpoint.pt
```

## Prepare the real SLM/CCD experiment

Generate original/processor images, token fields, four shared phase masks,
first-plane amplitude BMPs, and all four simulated detector references:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml --phase prepare
```

Process the four physical CCD folders in order:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml --phase process_vision_expert
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml --phase process_vision_global
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml --phase process_language_expert
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml --phase process_language_global
```

For an end-to-end dry hardware rehearsal using simulated CCD intensity:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_hardware_deployment.yaml --phase all_simulation
```

See `HARDWARE_DEPLOYMENT.md` before collecting any physical CCD files.

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
