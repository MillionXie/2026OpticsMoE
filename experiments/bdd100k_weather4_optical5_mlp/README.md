# BDD100K Weather-4 Optical5 + MLP

This experiment is a no-Qwen optical baseline. It converts BDD100K images to grayscale amplitude fields and trains a five-layer differentiable optical propagation network plus an MLP readout end-to-end for Weather-4 classification. It does not use a tokenizer, language model, teacher model, transformer block replacement, distillation, or MoE routing.

## Pipeline

```text
RGB image
 -> 224 x 224 grayscale in [0, 1]
 -> RMS-normalized amplitude
 -> 256 x 256 optical field
 -> five optical propagation / mask / detection conversions
 -> 16 x 16 adaptive detector pooling
 -> MLP readout
 -> [clear, rainy, snowy, foggy] logits
```

Each optical conversion applies a trainable phase mask and optional amplitude mask, angular-spectrum free-space propagation, square-law detection, intensity normalization, a ReLU-like nonlinearity, and square-root amplitude re-encoding. There is no residual electronic bypass. The phase mask is applied before propagation because phase applied directly at a square-law detector would have no effect on intensity.

## Dataset

Prepare this ImageFolder layout at the configured `data_root` (by default `<repository>/data/bdd100k_weather4`):

```text
data/bdd100k_weather4/
  train/{clear,rainy,snowy,foggy}/
  test/{clear,rainy,snowy,foggy}/
```

The training directory is split per class into training and validation subsets. The test directory is never used during training. Because `foggy` is rare, the run records macro-F1, balanced accuracy, per-class metrics, and the confusion matrix in addition to overall top-1 accuracy.

`--phase prepare_data` validates and summarizes an already organized ImageFolder dataset; it does not download BDD100K because redistribution and access depend on the dataset provider.

## Outputs

The run directory contains resolved configuration, environment, dataset and model manifests; training and test metrics; best and last checkpoints; training curves; confusion matrix; wrapped phase masks; per-layer light fields; and detector outputs. Phase dropout is used only in training at or after its configured start epoch and is always disabled in evaluation.

See [RUN_COMMANDS.md](RUN_COMMANDS.md) for commands.
