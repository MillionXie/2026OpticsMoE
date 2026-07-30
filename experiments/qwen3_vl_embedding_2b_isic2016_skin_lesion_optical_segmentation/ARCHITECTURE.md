# Architecture

## Frozen electronic front end

The original Qwen3-VL Vision patch and position embedding is retained at a
fixed 224x224 processor budget. Native Vision Transformer blocks, the
language model and text processing are not executed by the student.

## Optical backbone

The implementation reuses the exact validated COCO/DUTS backbone:

- 16 phase-only experts in a 4x4 grid;
- each expert is 224x224 pixels;
- 254-pixel pitch, 986x986 active footprint and 1026x1026 FFT canvas;
- electronic Top-4 input-dependent router;
- three independent expert phase stages;
- per-selected-expert OEO normalization/ReLU/reload between stages;
- original routing weights reapplied after each OEO step;
- one 986x986 global phase plane;
- 10 cm expert-stage, global and CCD propagation distances;
- CCD readout pooled to 224x224.

The detector readout is recombined as:

```text
Fout = Fccd + alpha * Linear(LayerNorm(Fccd))
```

with trainable `alpha`, initialized to 0.1. Packed token rows are restored
using runtime `image_grid_thw`; no token cropping, truncation or hard-coded
14x14 reshape is permitted.

## Segmentation decoder

The shared lightweight head projects each 224D token to 128 channels and
uses a small convolutional decoder (`128 -> 64 -> 32 -> 16 -> 1`) to output
224x224 raw mask logits. Sigmoid is only applied for metrics and display.

## Initialization comparison

| Run | Optical core | Recombiner | Segmentation head | Qwen |
|---|---|---|---|---|
| scratch | random | random | random | frozen |
| COCO/DUTS pretrained | transferred | transferred | transferred | frozen |

The pretrained optimizer state is deliberately not restored.
