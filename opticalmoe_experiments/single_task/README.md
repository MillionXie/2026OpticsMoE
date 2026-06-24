# Single-Task Experiments

This experiment family compares four single-task classifiers under the same
dataset preprocessing:

- `general_d2nn`: a non-MoE diffractive optical neural network baseline.
- `fixed_route_moe`: AS global-router OpticalMoE with fixed prompt amplitudes.
- `learnable_route_moe`: AS global-router OpticalMoE with learnable prompt
  amplitudes and phase biases.
- `lenet5`: electronic neural-network baseline.

The default OpticalMoE uses the successful Angular-Spectrum global-router path:

```text
input
-> AngularSpectrumPropagator input_to_prompt
-> prompt-plane complex-amplitude global router
-> AngularSpectrumPropagator prompt_to_expert
-> expert entrance aperture
-> expert phase layers
-> global FC phase
-> detector/readout
```

It does not use the old spatial partition prompt and does not use FFT
convolution to create the expert entrance field.

## Datasets

Supported datasets:

- MNIST
- Fashion-MNIST
- KMNIST
- EMNIST, default split `letters`
- CIFAR10 grayscale

All optical configs default to `input_size=134`. CIFAR10 is converted to
single-channel grayscale before optical propagation.

## Config Matrix

The `configs/` folder contains a complete single-task matrix:

| Dataset | Learnable MoE | Fixed MoE | D2NN | LeNet-5 |
| --- | --- | --- | --- | --- |
| MNIST | yes | yes | yes | yes |
| Fashion-MNIST | yes | yes | yes | yes |
| KMNIST | yes | yes | yes | yes |
| EMNIST letters | yes | yes | yes | yes |
| CIFAR10 grayscale | yes | yes | yes | yes |

The YAML files intentionally expand nested fields instead of using compact
one-line maps. Commonly tuned parameters are visible under:

- `dataset`
- `model`
- `optics`
- `prompt`
- `readout`
- `regularization.phase_dropout`
- `optimizer`
- `training`
- `visualization`

## Default MoE Geometry

The default 9-expert setup is fair134:

- `canvas_size=1000`
- `input_size=134`
- `expert_size=134`
- `expert_pitch=200`
- `padding=200`
- `prompt_aperture_size=600`
- expert centers: `[300, 500, 700] x [300, 500, 700]`

The MoE code also supports `num_experts=4` with a 2x2 global-router layout.

## Saved Outputs

Each run is saved under:

```text
single_task/runs/<run_id>/
```

Important outputs:

- `config.yaml`, `config_resolved.json`, `command.txt`
- `git_info.json`, `environment.json`
- `architecture_report.json`
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics/epoch_metrics.csv`
- `metrics/final_metrics.json`
- `metrics/confusion_matrix.csv`
- `figures/training_curves.png`
- `figures/confusion_matrix.png`
- every 10 epochs: light fields, prompt maps, phase masks, detector outputs,
  and sample predictions
- `summary_for_master/` rows for rebuilding master tables

## Dropout Clarification

`readout.dropout` is electronic dropout inside the detector readout head.

`regularization.phase_dropout` is optical phase-layer dropout. It randomly
bypasses phase pixels or phase blocks during training and is automatically
disabled during evaluation.

## Recommended Run Order

1. Run a MNIST smoke test.
2. Run MNIST baselines: learnable MoE, fixed MoE, D2NN, LeNet-5.
3. Extend to Fashion-MNIST, KMNIST, EMNIST letters, and CIFAR10 grayscale.
4. Rebuild master tables from all completed run folders.
