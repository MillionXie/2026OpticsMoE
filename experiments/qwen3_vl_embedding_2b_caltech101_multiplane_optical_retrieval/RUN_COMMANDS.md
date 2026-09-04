# Run commands

Run from the repository root.

## Smoke tests

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval.run_all_variants --smoke --phase train
```

## Four primary controlled experiments

Continuous five-plane D2NN:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/configs/release/d2nn_continuous.yaml --phase all
```

D2NN with four intermediate full-aperture CCD/normalization/sigmoid reloads:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/configs/release/d2nn_oeo_sigmoid.yaml --phase all
```

Continuous MoE with one fixed input router:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/configs/release/moe_continuous_fixed_router.yaml --phase all
```

MoE with per-expert sigmoid OEO and independent router per expert plane:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/configs/release/moe_oeo_dynamic_router.yaml --phase all
```

## Supplemental fixed-router OEO MoE

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/configs/release/moe_oeo_fixed_router.yaml --phase all
```

Run the four primary groups and then the supplement sequentially:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval.run_all_variants --phase all
```

Resume from a run checkpoint by using the corresponding config:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval --config CONFIG_PATH --phase train --resume-checkpoint CHECKPOINT_PATH
```

## Build plot-ready tables

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval.compare_results
```

Generated tables:

```text
experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/comparison/comparison.csv
experiments/qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval/comparison/comparison.json
```
