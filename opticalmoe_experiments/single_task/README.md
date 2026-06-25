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

`canvas_size=1000` is the propagation window. The trainable active optical
window is the center `600 x 600` region, matching the prompt aperture
`y=200:800, x=200:800`. For fair134, the 9 expert apertures occupy an expert
union size of `534 x 534`, so the center `600 x 600` window covers the expert
bank while leaving propagation padding outside. The padding region is
transparent and is not trainable in the global FC phase mask.

The MoE code also supports `num_experts=4` with a 2x2 global-router layout.

## General D2NN Accounting

The `general_d2nn` baseline is defined as:

```text
input
-> first AngularSpectrumPropagator
-> 5 center-window D2NN phase masks
-> layer5_to_fc propagation
-> one center-window global FC phase mask
-> fc_to_detector propagation
-> detector/readout
```

The `target_local_phase_param_count` field in D2NN configs refers only to the
5 local center-window phase masks. The actual optical parameter count also
includes the center-window `global_fc` phase mask. For the default D2NN config:

- local D2NN phase params: `5 * 402 * 402 = 808020`
- center-window global FC params: `600 * 600 = 360000`
- total optical params: `1168020`

D2NN phase masks are saved under `figures/phase_masks/<epoch>/` as
`d2nn_phase_layer_*.png`, `d2nn_all_phase_layers.png`, and
`global_fc_phase_window.png`, `global_fc_phase_region_on_canvas.png`, and the
compatibility file `global_fc_phase.png`.

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

`summary_for_master/expert_usage_rows.json` stores prompt amplitudes,
normalized prompt powers, and fixed-validation-batch expert energy ratios for
MoE runs. `summary_for_master/optical_energy_rows.json` stores per-stage optical
energy diagnostics and is used to rebuild `master_optical_energy.csv`.

## Dropout Clarification

`readout.dropout` is electronic dropout inside the detector readout head.

`regularization.phase_dropout` is optical phase-layer dropout. It randomly
bypasses phase pixels or phase blocks during training and is automatically
disabled during evaluation.

LeNet-5 is an electronic baseline. It does not save optical phase masks or
optical energy rows, and it now adapts to the configured dataset input size
instead of assuming `134 x 134`.

## DataLoader Workers

All configs expose `num_workers`, `pin_memory`, `persistent_workers`, and
`prefetch_factor`. Linux servers should usually start with `num_workers=16`,
`pin_memory=auto`, `persistent_workers=true`, and `prefetch_factor=4`. If CPU or
RAM pressure is high, try `num_workers=8` or `4`. On Windows or during
debugging, use `num_workers=0`. The `--smoke_test` flag automatically forces
`num_workers=0`, `persistent_workers=false`, and `prefetch_factor=null`.

## Recommended Run Order

1. Run a MNIST smoke test.
2. Run MNIST baselines: learnable MoE, fixed MoE, D2NN, LeNet-5.
3. Extend to Fashion-MNIST, KMNIST, EMNIST letters, and CIFAR10 grayscale.
4. Rebuild master tables from all completed run folders.
