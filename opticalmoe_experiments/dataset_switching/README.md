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

Each dataset has its own detector/readout head module. These modules are
separate `nn.Module` instances and do not share parameters. The global
`detector:` and `readout:` sections in the YAML files are defaults; each
`training.multitask.tasks[*].head` can override them. If a task does not define
`head`, it still receives an independent head using the global defaults.

EMNIST-letters has 26 output classes and may benefit from a larger
`hidden_dim` than MNIST/Fashion-MNIST. To tune only EMNIST:

```yaml
training:
  multitask:
    tasks:
      - name: emnist_letters
        head:
          hidden_dim: 96
          activation: gelu
```

The resolved per-task head configs are saved in `config_resolved.json`,
`architecture_report.json`, and `summary.json`.

The shared backbone now defaults to the `fast120_520` 9-expert
Angular-Spectrum global router: canvas `520`, input/expert `120`, pitch `150`,
centers `[110, 260, 410] x [110, 260, 410]`, and outer padding `35`. The expert
union is `[50:470, 50:470]` (size `420`). Prompt and global FC use the center
`[35:485, 35:485]` active window (size `450`), with transparent non-trainable
padding outside. `fair134_1000` remains an explicit legacy profile.

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

### Independent D2NN parameter budget

The independent baseline is compared as a **group of separate networks**:

```text
MNIST D2NN + Fashion-MNIST D2NN + EMNIST-letters D2NN
```

The relevant parameter comparison is therefore the sum of the three
independent optical networks versus one shared three-task MoE. Current configs
use:

```text
propagation canvas: 520 x 520
five local phase windows: 360 x 360
one global-FC phase window: 450 x 450
optical parameters per task: 5 * 360^2 + 450^2 = 850500
```

The `360 x 360` local grid matches one complete MoE expert layer's local phase
count (`9 * 120 * 120 = 129600`). Each independent network and the sum of all
independent networks are reported from actual parameters; this is not claimed
to be an exact group-level parameter match.

Three standalone configs are provided so the networks can be trained
independently or in parallel:

- `mnist_independent_d2nn_canvas400_grid220.yaml`
- `fashionmnist_independent_d2nn_canvas400_grid220.yaml`
- `emnist_letters_independent_d2nn_canvas400_grid220.yaml`

These historical filenames are retained for command compatibility. Their
resolved contents now use canvas `520`, local grid `360`, and global FC `450`.

The combined `mnist_fashion_emnist_letters_independent_d2nn.yaml` remains
available. It runs three fully separate models sequentially, and `--task`
can select one task from that combined file. The script reports both the
parameters actually executed in the current run and the planned three-network
parameter total.

## Core Diagnostics

- Prompt swap evaluation is the central evidence for dataset-specific optical
  prompts. It is saved to `metrics/prompt_swap_matrix.csv`.
- Expert usage is saved to `diagnostics/expert_usage.csv` and
  `summary_for_master/expert_usage_rows.json`.
- Prompt similarity is saved to `diagnostics/prompt_similarity.csv` and
  `summary_for_master/prompt_similarity_rows.json`.
- Master tables are rebuilt under `dataset_switching/results/`.
- Epoch logs now print every task's train/validation loss and accuracy in
  addition to joint/macro metrics. Full per-task history is saved in
  `metrics/task_metrics.csv`.

## Dataset Size Controls

Each task has its own `training.multitask.tasks[*].dataset` block, and every
task dataset exposes:

```yaml
sampling_protocol:
  enabled: false
  total_size: null
  train_test_ratio: [4, 1]
  class_balanced: true
  seed_offset: 0
max_train_samples: null
max_val_samples: null
max_test_samples: null
```

`enabled: false` uses the official full train/test split for that dataset, then
uses `val_split` to carve validation out of the train split. `enabled: true`
makes `total_size` mean train+val+test for that task. For example, with
`total_size=10000`, `[4,1]`, and `val_split=0.1`, the effective split is about
`train=7200`, `val=800`, `test=2000`.

Use `max_train_samples`, `max_val_samples`, and `max_test_samples` for direct
per-split caps. The training script prints the effective split sizes for every
task and writes them to `loader_summary.json`.

## Multitask Training Fields

- `sequential_backward: true`: each update loops through tasks and performs
  task-by-task forward/backward before one shared `optimizer.step()`. This is
  the memory-safe mode for the shared optical backbone.
- `balanced_sampling: true`: task loaders are cycled so MNIST/Fashion/EMNIST
  get balanced update opportunities even if their dataset sizes differ.
- `loss_reduction: mean`: task losses are averaged after applying weights, so
  the joint loss scale does not grow just because more tasks are added.

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
