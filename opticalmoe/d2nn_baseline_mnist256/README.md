# MNIST 256-to-400 D2NN Baseline

This folder is a standalone MNIST D2NN baseline for comparison or external use.

It is intentionally separate from the current OpticalMoE, `opticalmoe_experiments/single_task`, and multitask code paths. It does not use MoE, prompt routing, expert banks, or the 9-expert AS global router.

## Experiment

- Dataset: MNIST
- Train split: full official MNIST train split
- Test split: full official MNIST test split
- Input image: resized to `256 x 256`
- Optical canvas: centered padding to `400 x 400`
- Optical layers: 5 full-canvas phase-only masks
- Phase mask size: `400 x 400`
- Pixel size: `8 um`
- Wavelength: `532 nm`
- Propagation: Angular Spectrum propagation
- Inter-layer distance: `5 cm`
- Last-layer-to-detector distance: `5 cm`
- Detector: 10 detector regions, default `32 x 32`, grid layout
- Readout: configurable `optical_only`, `linear`, or `mlp`; default `mlp`
- Optimizer: AdamW, `lr=0.001`, `weight_decay=0.0005`
- Batch size: 128

## Phase Dropout

`regularization.phase_dropout` is optical phase dropout. During training only, it randomly bypasses phase modulation at phase pixels or blocks:

```text
keep * exp(i phase) + (1 - keep) * 1
```

It does not modify stored phase parameters and is disabled automatically in `model.eval()`.

`readout.dropout` is separate electronic dropout inside the optional electronic readout head.

## Output Structure

Each run is saved to:

```text
runs/<run_name>/
  config.yaml
  config_resolved.json
  command.txt
  environment.json
  git_info.json
  architecture_report.json
  summary.json
  checkpoints/
    best.pt
    last.pt
  metrics/
    epoch_metrics.csv
    final_metrics.json
    confusion_matrix.csv
  figures/
    training_curves.png
    confusion_matrix.png
    light_fields/
    phase_masks/
    detector_outputs/
    samples/
```

Every saved visualization epoch includes light-field snapshots, phase masks, detector outputs, and sample predictions.

