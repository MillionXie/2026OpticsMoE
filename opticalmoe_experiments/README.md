# OpticalMoE Experiments

This directory is a clean experiment workspace placed beside the legacy
`opticalmoe/` project. It is intended for reusable experiments built around the
validated Angular-Spectrum global-router OpticalMoE path:

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

It intentionally does not use the older spatially partitioned prompt or FFT
convolution shortcut for expert entrance generation.

Implemented first:

- `single_task/`: single-dataset classification baselines and MoE variants.

Placeholders for future work:

- `dataset_switching/`
- `same_input_multitask/`
- `expert_task_ablation/`
- `prompt_ablation/`

## Dataset Size Control

All training YAML files expose the same dataset-size controls.

```yaml
sampling_protocol:
  enabled: false
  total_size: null
  train_test_ratio: [4, 1]
  class_balanced: true
  seed_offset: 0
```

`enabled: false` means use the official full dataset split. For torchvision
datasets such as MNIST, the official train split is used as a train pool, then
`val_split` is carved out of that train pool, and the official test split stays
as test.

`enabled: true` means `total_size` is the approximate experiment-wide total:

```text
total_size = train + val + test
train_test_ratio = [4, 1] -> 80% train_pool, 20% test
val_split cuts validation from train_pool
```

For example, `total_size=10000`, `train_test_ratio=[4,1]`, `val_split=0.1`
gives approximately:

```text
train_pool = 8000
test = 2000
val = 800
train = 7200
```

You can also use direct split caps:

```yaml
max_train_samples: null
max_val_samples: null
max_test_samples: null
```

These are useful when you want exact values such as train=5000, val=1000,
test=1000. Smoke tests have highest priority and force their own small split
sizes.

## DataLoader Fields

All dataset configs support:

```yaml
num_workers: 16
pin_memory: auto
persistent_workers: true
prefetch_factor: 4
```

- `num_workers`: DataLoader worker process count. Linux servers can start at
  `16`; reduce to `8` or `4` if CPU/RAM is saturated. Windows/debug runs are
  usually safest with `0`.
- `pin_memory: auto`: enables pinned CPU memory when CUDA is available.
- `persistent_workers: true`: keeps workers alive between epochs when
  `num_workers > 0`.
- `prefetch_factor: 4`: each worker prepares 4 batches ahead; with 16 workers
  this can stage about 64 batches.

`--smoke_test` forces:

```yaml
num_workers: 0
persistent_workers: false
prefetch_factor: null
```

Each run writes `loader_summary.json`, and `summary.json` / master-row JSONs
also include the effective train/val/test sizes and loader settings.

## Common Multitask Fields

- `sequential_backward: true`: in dataset switching, each update forwards and
  backwards one task at a time, then calls one shared `optimizer.step()`. This
  lowers GPU memory pressure.
- `balanced_sampling: true`: in dataset switching, each task is sampled more
  evenly even if datasets have different sizes.
- `loss_reduction: mean`: average task losses so adding tasks does not scale
  the joint loss upward.
- `batch_mode: paired_same_input`: same-input multitask only. The same dSprites
  image batch is reused for all tasks in the update.

## Task-Specific Readout Heads

Multitask experiments use independent electronic detector/readout heads per
task. The global `detector:` and `readout:` sections are defaults only.

In `dataset_switching`, each task can override the default under:

```yaml
training:
  multitask:
    tasks:
      - name: mnist
        head:
          hidden_dim: 64
```

In `same_input_multitask`, overrides live under:

```yaml
training:
  task_heads:
    shape:
      hidden_dim: 32
```

If a task-specific head is missing, the task still gets its own independent
head module using the global defaults. Unknown task names in head configs raise
an error. Architecture reports include `task_head_configs`,
`task_readout_parameter_counts`, and `task_readout_modules_are_independent`.

## Grayscale Behavior

MNIST, Fashion-MNIST, KMNIST, and EMNIST are already grayscale; setting
`grayscale: false` does not save meaningful compute because the transform still
outputs one channel. CIFAR10 configs are intentionally grayscale for the current
single-channel optical amplitude pipeline. Do not use `grayscale: false` as a
speed optimization; RGB optics would need a separate pipeline.

## Config Audit

Run this after editing YAML files:

```powershell
python opticalmoe_experiments/scripts/audit_dataset_config_fields.py
```

It fails if any training config misses the required dataset-size or DataLoader
fields.
