# Configuration reference

The JSON files use a grouped schema instead of a flat list. `config_resolved.json` is written with the same grouping, so the submitted configuration can be inspected directly after a run.

## Experiment and data

- `experiment.output_dir`: run output directory, resolved relative to the config file.
- `dataset.train_limit` / `test_limit`: permanent whole-dataset limits, mainly for smoke tests.
- `dataset.train_limit_per_class` / `test_limit_per_class`: permanent per-class subset limits.
- `dataset.train_samples_per_class_per_epoch`: rotating per-class epoch window. It limits work in one student epoch without discarding the rest of the cached training set. Successive epochs move through each class, and batches are class-mixed.
- `dataset.validation_fraction`: fraction of cached training features reserved for teacher-head validation.

## Qwen and batching

- `qwen.processor.min_pixels` / `max_pixels`: Qwen image-processor pixel budget. The resulting pre-merger token count must not exceed 120.
- `qwen.runtime`: backbone dtype, attention implementation and device.
- `batching`: independent batch sizes for teacher-cache extraction, student training, inference and teacher-head training.
- `teacher_cache.log_interval_batches`: command-line refresh interval while teacher features are cached.

## Homogeneous optical MoE

- `vision_adapter`: token-channel projection and maximum visual token count.
- `moe.geometry`: verified 480 canvas / 450 active area / 3x3 experts / 120 expert size / 150 pitch geometry.
- `moe.router`: input-dependent top-k routing controls.
- `moe.optics`: wavelength, pixel pitch, distances, phase initialization and optional k-space constraint.
- `moe.optoelectronic_interlayers`: square-law detection, LayerNorm, nonlinearity and optical reload between phase stages. `per_expert_enabled=true` keeps expert normalizations independent.
- `moe.final_detector_readout`: full 480x480 detector readout. The default LayerNorm is deliberately non-affine.

## Optimization, logging and saving

- `optimizer.type`: `adam` or `adamw`.
- `optimizer.scheduler`: `cosine` or `none`.
- `training.logging.interval_batches`: print after this many student batches.
- `training.logging.interval_seconds`: also print when this wall-clock interval is reached. The first of the batch/time limits triggers a refresh, so a slow batch cannot leave the terminal silent for too long.
- `training.checkpoint_interval_epochs`: save numbered epoch snapshots at this interval. `last` and a newly improved `best` are still saved immediately.
- `visualization.interval_epochs`: phase-mask visualization interval.
- `regularization.phase_dropout`: optional training-only phase bypass. It is disabled in the main config.

## Useful CLI overrides

The following values can be changed without editing JSON:

```text
--epochs
--batch-size / --student-batch-size
--train-samples-per-class-per-epoch
--log-interval-batches
--log-interval-seconds
--checkpoint-interval-epochs
--visualization-interval-epochs
--disable-visualization
```

Legacy flat JSON keys remain accepted for backward compatibility, but all supplied configs and resolved outputs use the grouped schema.
