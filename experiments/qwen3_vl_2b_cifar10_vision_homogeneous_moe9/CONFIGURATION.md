# Configuration reference

The JSON files use a grouped schema instead of a flat list. `config_resolved.json` is written with the same grouping, so the submitted configuration can be inspected directly after a run.

## Experiment and data

- `experiment.output_dir`: run output directory, resolved relative to the config file.
- `dataset.train_limit` / `test_limit`: permanent whole-dataset limits, mainly for smoke tests.
- `dataset.train_limit_per_class` / `test_limit_per_class`: permanent per-class subset limits.
- `dataset.train_samples_per_class_per_epoch`: rotating per-class epoch window. It limits work in one student epoch without discarding the rest of the cached training set. Successive epochs move through each class, and batches are class-mixed.
- `dataset.validation_fraction`: used only to select the frozen teacher classification head. Student optical training uses the complete retained CIFAR-10 training split.

## Qwen and batching

- `qwen.processor.min_pixels` / `max_pixels`: Qwen image-processor pixel budget. The resulting pre-merger token count must not exceed 120.
- `qwen.runtime`: backbone dtype, attention implementation and device.
- `batching`: independent batch sizes for teacher-cache extraction, student training, inference and teacher-head training.
- `teacher_cache.log_interval_batches`: command-line refresh interval while teacher features are cached.

## Homogeneous optical MoE

- `vision_adapter`: token-channel projection and maximum visual token count.
- `moe.geometry`: verified 480 canvas / 450 active area / 3x3 experts / 120 expert size / 150 pitch geometry.
- `moe.router`: input-dependent top-k routing controls. With `input_layernorm_enabled=true`, the pooled 10x10 router feature receives a non-affine per-sample LayerNorm before the linear gate. This removes shared DC/scale variation and exposes sample-dependent spatial differences without adding trainable normalization parameters or routing noise.
- `moe.optics`: wavelength, pixel pitch, distances, phase initialization and optional k-space constraint.
- `moe.optoelectronic_interlayers`: square-law detection, LayerNorm, nonlinearity and optical reload between phase stages. `per_expert_enabled=true` keeps expert normalizations independent. With `reapply_routing_weights=true`, each selected expert is multiplied by its sample-dependent sparse router coefficient after LayerNorm and activation, restoring the amplitude relationship that normalization removes. With `hard_route_mask=true`, unselected expert regions are then forced to exact zero before zero-phase reload. The complete order is `detection -> per-expert LN -> activation -> amplitude weight -> hard zero -> reload`.
- `moe.final_detector_readout`: full 480x480 detector readout. The default LayerNorm is deliberately non-affine.

## Optimization, logging and saving

- `optimizer.type`: `adam` or `adamw`.
- `optimizer.scheduler`: `cosine` or `none`.
- `training.logging.interval_batches`: print after this many student batches.
- Logging is batch-triggered only. Each update includes the cumulative selection rate and mean selected routing weight for all nine experts.
- `training.student_selection_split`: fixed to `test`. The complete CIFAR-10 test split is evaluated after every student epoch and selects `best`; this intentionally matches the legacy homogeneous-MoE protocol but makes the reported best-test number selection-biased.
- `training.checkpoint_interval_epochs`: save numbered epoch snapshots at this interval. `last` and a newly improved `best` are still saved immediately.
- `visualization.interval_epochs`: phase-mask and debug-example visualization interval.
- `visualization.sample_count`: random test examples saved at each visualization epoch.
- `visualization.save_intermediate_fields`: saves RGB input, optical input, prompt expert amplitudes, routing weights, fan-out field, five stage fields, detector intensity, and teacher/student hidden comparisons.
- `regularization.phase_dropout`: optional training-only phase bypass. It is disabled in the main config. Dropout is sampled again on every training forward/batch, not once per epoch. `phase_bypass` independently bypasses phase pixels; `block_phase_bypass` bypasses square blocks. A bypassed point uses unit complex modulation (zero added phase), not zero optical amplitude. `p` is the expected bypass fraction per phase plane. `batch_shared=true` uses the same spatial dropout mask for all samples in a mini-batch; `false` samples an independent mask per image. `start_epoch` delays activation. A mild starting point is `mode=block_phase_bypass`, `p=0.02`, `block_size=8`, `batch_shared=true`, `start_epoch=10`.

## Useful CLI overrides

The following values can be changed without editing JSON:

```text
--epochs
--batch-size / --student-batch-size
--train-samples-per-class-per-epoch
--log-interval-batches
--checkpoint-interval-epochs
--visualization-interval-epochs
--disable-visualization
```

Legacy flat JSON keys remain accepted for backward compatibility, but all supplied configs and resolved outputs use the grouped schema.
