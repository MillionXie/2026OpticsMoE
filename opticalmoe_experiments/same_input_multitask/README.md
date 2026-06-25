# Same-Input Multitask

This experiment studies task switching on the same dSprites input image. A
single batch of images is reused for every task in the update. The model changes
only the task-specific optical prompt and the task-specific readout head while
sharing the 9-expert optical backbone.

Core training step:

```text
same images
  -> prompt_shape + shape readout -> shape loss
  -> prompt_scale + scale readout -> scale loss
  -> optional x/y position readouts -> position losses
weighted mean loss -> backward -> optimizer.step
```

Each task has its own detector/readout head module. For example, `shape`,
`scale`, `x_position_4bin`, and `y_position_4bin` use separate `nn.Module`
instances and do not share electronic readout parameters. The global `readout:`
section is only the default architecture. Per-task overrides live in
`training.task_heads`:

```yaml
training:
  task_heads:
    shape:
      hidden_dim: 32
    scale:
      hidden_dim: 64
```

If a task is omitted from `task_heads`, it still gets an independent readout
head using the global defaults. Unknown task names in `task_heads` raise an
error. Resolved head configs and readout parameter counts are saved in
`config_resolved.json`, `architecture_report.json`, and `summary.json`.

## Stages

Stage 1:

- `shape`
- `scale`

Stage 2:

- `shape`
- `scale`
- `x_position_4bin`

Stage 3:

- `shape`
- `scale`
- `x_position_4bin`
- `y_position_4bin`

Orientation and color are intentionally excluded. All targets are classification
labels.

## dSprites Labels

dSprites latent class columns are used directly:

- `shape = latents_classes[:, 1]`, 3 classes.
- `scale = latents_classes[:, 2]`, 6 classes.
- `x_position_4bin = latents_classes[:, 4] // 8`, 4 classes.
- `y_position_4bin = latents_classes[:, 5] // 8`, 4 classes.

The x/y labels are deterministic coarse bins of the official latent classes;
they are not generated pseudo-labels.

## Optical Model

The default MoE reuses the successful fair134 AS global-router geometry:

- canvas: `1000 x 1000`
- input: `134 x 134`
- experts: `9 x 134 x 134`
- expert centers: `[300, 500, 700] x [300, 500, 700]`
- prompt aperture: center `600 x 600`
- propagation: Angular Spectrum only

The `1000 x 1000` canvas is only the propagation window. The active trainable
optical window is the center `600 x 600` region (`y=200:800, x=200:800`). The
global FC phase mask is trainable only in that window; outside it, propagation
padding is transparent and not trainable. The prompt is also aperture-limited
to the center `600 x 600` and its trainable parameters are channel amplitudes
and phase biases, not a pixel-wise `1000 x 1000` prompt.

The expert entrance field is produced by
`AngularSpectrumPropagator(prompt_to_expert)`. The code does not use FFT
convolution to synthesize the expert plane, does not split the input into
patches, and does not apply a gate at the expert entrance as the main routing
mechanism.

## Outputs

Each run is saved under:

```text
same_input_multitask/runs/<run_id>/
```

Important files:

- `metrics/task_metrics.csv`
- `metrics/same_input_task_switching.csv`
- `metrics/prompt_swap_matrix.csv`
- `diagnostics/task_expert_energy_history.csv`
- `diagnostics/prompt_similarity.csv`
- `summary_for_master/*.json`

Master tables are rebuilt under:

```text
same_input_multitask/results/
```

Prompt swap is the main evidence that the optical prompt affects task behavior:
for each readout task, the evaluation keeps the readout fixed and swaps the
prompt task.

Epoch logs now show each task's train/validation loss and accuracy in addition
to macro/joint metrics. Full per-task history is saved in
`metrics/task_metrics.csv`.

## Dataset Size Controls

same-input multitask has one top-level `dataset` block because every task uses
the same dSprites image batch.

```yaml
sampling_protocol:
  enabled: true
  total_size: 12000
  train_test_ratio: [4, 1]
  class_balanced: false
  seed_offset: 0
max_train_samples: null
max_val_samples: null
max_test_samples: null
```

For dSprites, `enabled: true` now means: shuffle all official dSprites indices,
take `total_size`, split that selected pool into train_pool/test by
`train_test_ratio`, then split validation from train_pool by `val_split`.

With the default `total_size=12000`, `[4,1]`, and `val_split=0.1`, the split is:

```text
train = 8640
val = 960
test = 2400
```

The train/val/test indices are deterministic and non-overlapping. Multi-label
class balancing is not forced for dSprites because simultaneously balancing
shape, scale, x bin, and y bin is a different sampling problem; the configs
therefore set `class_balanced: false`.

Use `max_train_samples`, `max_val_samples`, and `max_test_samples` for direct
per-split caps. Each run prints and saves `loader_summary.json`.

## Paired Batch Field

`batch_mode: paired_same_input` means each update uses the same dSprites image
batch for every task. The model runs that same `images` tensor through
`prompt_shape`, `prompt_scale`, and optional x/y prompts, computes each task
loss, averages the weighted losses, then performs one optimizer step. This
experiment does not use independent per-task loaders.

## DataLoader Workers

The dSprites dataset config exposes `num_workers`, `pin_memory`,
`persistent_workers`, and `prefetch_factor`. On Linux servers, start with
`num_workers=16`, `pin_memory=auto`, `persistent_workers=true`,
`prefetch_factor=4`. If CPU or memory pressure is high, reduce workers to `8`
or `4`. On Windows or while debugging, set `num_workers=0`. Smoke tests force
`num_workers=0`, disable persistent workers, and omit prefetching.
