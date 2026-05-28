# OpticalMoE

Initial PyTorch framework for a diffractive optical neural network classifier.

This first version intentionally implements a single optical classifier only:

- angular spectrum propagation with `torch.fft`
- five trainable phase-only diffractive layers by default
- a reserved physical prompt plane using `IdentityPrompt`
- fixed detector arrays
- configurable electronic readout
- MNIST/FashionMNIST/KMNIST dataloaders
- CSV logging, checkpoints, and visualizations

It does **not** implement MoE, expert banks, trainable optical routing, trainable prompts, or black-box optimization yet.

## Quick Start

```bash
cd opticalmoe
python scripts/train.py --config configs/mnist_donn.yaml --run_name smoke_mnist
```

Run outputs are written to:

```text
runs/smoke_mnist/
  config.yaml
  metrics.csv
  summary.json
  best.pt
  last.pt
  detector_layout.png
  confusion_matrix.png
  phases/
  light_fields/
  sample_outputs/
```

## Physical Defaults

- wavelength: 532 nm
- pixel size: 8 um
- input image size: 200 x 200
- padding: 200 pixels on each side
- simulation grid: 600 x 600
- phase layers: 5
- propagation distance: 5 cm for every segment

For 5 phase layers, the optical path has 7 propagation segments:

```text
input -> prompt
prompt -> phase layer 1
phase layer 1 -> phase layer 2
phase layer 2 -> phase layer 3
phase layer 3 -> phase layer 4
phase layer 4 -> phase layer 5
phase layer 5 -> detector
```

The prompt plane always exists. No-prompt experiments use `IdentityPrompt`.

python scripts/train.py --config configs/mnist_donn.yaml --run_name mnist_001
