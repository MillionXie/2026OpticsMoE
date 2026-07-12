# MNIST-4 400x400 Amplitude D2NN Baseline

This experiment is a standalone four-class MNIST D2NN baseline.

It is created from `d2nn_baseline_mnist256`, but the optical architecture is intentionally simplified:

```text
MNIST digit image, digits 0/1/2/3 only
-> resize to 400 x 400 grayscale amplitude
-> angular spectrum propagation, 3 cm
-> one trainable 400 x 400 phase-only modulation layer
-> angular spectrum propagation to detector, 20 cm
-> four detector energies
-> logits for four classes
```

There is no electronic Linear, MLP, CNN, normalization head, or trainable readout after the detector. The detector output is the final classifier output.

## Classes

The default four classes are:

| Model label | MNIST digit |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |

## Optical geometry

| Item | Value |
|---|---:|
| input amplitude field | 400 x 400 |
| trainable phase area | 400 x 400 |
| padding | none |
| number of phase layers | 1 |
| wavelength | 532 nm |
| pixel size | 16 um |
| input-to-layer distance | 3 cm |
| layer-to-detector distance | 20 cm |
| inter-layer distance | 3 cm, kept only for config consistency |

The phase is constrained by:

```python
phase = 2.0 * pi * torch.sigmoid(raw_phase)
```

So the effective phase is always in `[0, 2π]`.

## Detector layout

The four output detectors use the requested 2 x 2 layout:

```text
det_size = 50
start_pos_x = 75
start_pos_y = 75
N_det_sets = [2, 2]
det_steps_x = [150, 150]
det_steps_y = 150
```

Detector top-left coordinates are:

```text
class 0: y[75:125],  x[75:125]
class 1: y[75:125],  x[225:275]
class 2: y[225:275], x[75:125]
class 3: y[225:275], x[225:275]
```

## Configs

Three phase initializations are provided:

| Config | Effective initialization intent |
|---|---|
| `config_phase_zero.yaml` | near-zero effective phase |
| `config_phase_uniform.yaml` | uniform effective phase over `[0, 2π]` |
| `config_phase_gaussian.yaml` | Gaussian effective phase around `π`, std from `init_std` |

`config.yaml` is the same as `config_phase_zero.yaml`.

## Outputs

Runs are saved under:

```text
runs/<run_name>/
```

Each run contains checkpoints, metrics, confusion matrix, phase masks, detector outputs, and light-field visualizations.
