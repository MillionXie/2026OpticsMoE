# Grocery10 commands

All commands run from the `2026OpticsMoE` repository root. Exploratory configs
were removed; the commands below are the maintained entry points.

## Train the phase-engaged Grocery10 variant

This variant is intended to train physically meaningful masks rather than let
the residual/adapters/readout absorb nearly all optimization. It starts from
zero raw phase and writes phase motion/gradient diagnostics every epoch:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_phase_engaged.yaml --phase all
```

Small three-epoch validation, including one phase-only focus epoch:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_phase_engaged_smoke.yaml --phase all
```

Monitor `phase_delta`, `phase_std`, `phase_grad` and per-stack unselected expert
counts in stdout. Detailed values are in `metrics/phase_training_latest.json`;
raw phase tensors and fixed-scale previews are under `phase_training/`. See
`PHASE_TRAINING_FIX.md` for the root-cause analysis.

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

### Hardware-robust MoE4 with 2x2 superpixels

This is a separate architecture, not a reinterpretation of a MoE16
checkpoint. It uses four experts in a 2x2 grid, routes each sample to top-2,
simulates 224x224 logical expert pixels at 16 um, and repeats every logical
SLM pixel into an exact 2x2 block on the physical 8 um device. The logical
active footprint is 478x478; the exported physical footprint is 956x956.
Captured 956x956 CCD intensity is reduced to 478x478 by exact non-overlapping
2x2 mean binning, never interpolation.

Train the new model from scratch:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_phase_dc_kspace_moe4_superpixel2.yaml --phase all
```

Run its one-epoch smoke configuration:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_phase_dc_kspace_moe4_superpixel2_smoke.yaml --phase all
```

After training, export the MoE4/superpixel2 hardware package:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_phase_dc_kspace_moe4_superpixel2_hardware.yaml --phase prepare
```

The MoE4 checkpoint is shape-incompatible with every MoE16 checkpoint by
design. Do not use `--resume-checkpoint` from the old 4x4 expert bank.

There are two deliberately separate deployment configs:

- `grocery10_hardware_deployment.yaml` exports the historical epoch-159 EMA
  checkpoint used by the strongest reproduction pipeline.
- `grocery10_phase_dc_kspace_hardware.yaml` exports the newly trained
  phase-DC + k-space checkpoint. Use this one for the zero-order/DC experiment.

The phase-DC deployment uses `positive_percentile=98.0` amplitude encoding.
The normalization divisor is computed only from strictly positive pixels, so
hard-zeroed experts, expert gaps, and token padding remain black without making
the useful pixels artificially dark. The upper 2% of positive amplitudes is
clipped, then the remaining values are mapped linearly to 8-bit. Every file's
divisor, clipped ratio, encoded mean, and positive-pixel median are recorded in
its `amplitude_metadata/*.json` file.

Export the new phase-DC + k-space checkpoint:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_phase_dc_kspace_hardware.yaml --phase prepare
```

This command does not train or alter weights. It loads the exact checkpoint
named in the hardware YAML and copies both the resolved model config and
checkpoint into `00_manifest/` for auditability.

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
