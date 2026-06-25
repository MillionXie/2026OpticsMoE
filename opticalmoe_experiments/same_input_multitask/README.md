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
