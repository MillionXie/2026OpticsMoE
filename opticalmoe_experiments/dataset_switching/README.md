# Dataset Switching Experiments

This experiment family studies dataset switching, not same-input multitask
classification. Different datasets use different optical prompts and readout
heads while sharing one 9-expert AS global-router optical backbone.

First-stage tasks:

- `mnist`: MNIST digits, 10 classes.
- `fashionmnist`: Fashion-MNIST, 10 classes.
- `emnist_letters`: EMNIST letters, 26 classes. Labels are mapped from
  `1..26` to `0..25`.

Stage three adds `kmnist`. CIFAR10-gray and USPS are intentionally not included
in this experiment family yet.

## Main Method

```text
MNIST input -> prompt_mnist -> shared optical backbone -> MNIST readout
Fashion input -> prompt_fashionmnist -> shared optical backbone -> Fashion readout
EMNIST letters input -> prompt_emnist_letters -> shared optical backbone -> EMNIST readout
```

The shared backbone is the successful 9-expert fair134 Angular-Spectrum global
router:

- `canvas_size=1000`
- `input_size=134`
- `expert_size=134`
- `expert_pitch=200`
- `padding=200`
- `prompt_aperture_size=600`
- expert centers `[300, 500, 700] x [300, 500, 700]`

The `1000 x 1000` canvas is the propagation window. The active trainable
optical window is the center `600 x 600` region (`y=200:800, x=200:800`), used
by the prompt aperture and the default global FC phase mask. The 9 fair134
expert apertures have an expert union size of `534 x 534`, so this active
window covers the expert bank. The surrounding padding is transparent and not
trainable in the global FC phase mask.

The expert entrance field is produced by AngularSpectrumPropagator from the
prompt plane. The implementation does not use FFT convolution to synthesize the
expert entrance field, does not use the old spatial partition prompt, and does
not gate amplitudes at the expert entrance plane.

## Baselines

- `learnable_route_moe`: task-specific learnable prompt amplitudes/phase biases.
- `fixed_route_moe`: uniform fixed prompt amplitudes and no trainable prompt
  phase biases. It does not manually assign datasets to experts.
- `shared_d2nn`: shared non-MoE optical D2NN backbone with task-specific heads.
- `independent_d2nn`: one separate D2NN per dataset. This is not an upper bound;
  it is a parameter/accounting baseline for separate networks.

## Core Diagnostics

- Prompt swap evaluation is the central evidence for dataset-specific optical
  prompts. It is saved to `metrics/prompt_swap_matrix.csv`.
- Expert usage is saved to `diagnostics/expert_usage.csv` and
  `summary_for_master/expert_usage_rows.json`.
- Prompt similarity is saved to `diagnostics/prompt_similarity.csv` and
  `summary_for_master/prompt_similarity_rows.json`.
- Master tables are rebuilt under `dataset_switching/results/`.

## Output Layout

Runs are saved under:

```text
dataset_switching/runs/<run_id>/
```

Important files:

- `architecture_report.json`
- `metrics/epoch_metrics.csv`
- `metrics/task_metrics.csv`
- `metrics/prompt_swap_matrix.csv`
- `metrics/final_test_metrics.json`
- `diagnostics/expert_usage.csv`
- `diagnostics/prompt_similarity.csv`
- `diagnostics/optical_energy_by_stage.csv`
- `summary_for_master/*.json`

## DataLoader Workers

Each task dataset config exposes `num_workers`, `pin_memory`,
`persistent_workers`, and `prefetch_factor`. On Linux servers, start with
`num_workers=16`, `pin_memory=auto`, `persistent_workers=true`,
`prefetch_factor=4`. If CPU or memory pressure is high, use `8` or `4`. On
Windows or during debugging, use `num_workers=0`. Smoke tests force
`num_workers=0`, disable persistent workers, and omit prefetching.
