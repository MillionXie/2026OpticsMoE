# Configuration guide

- `spaq_brightness.json`: full Brightness single-task run.
- `spaq_colorfulness.json`: full Colorfulness single-task run.
- `spaq_contrast.json`: full Contrast single-task run.
- `*_smoke.json`: 24 train images, 12 test images, one epoch, task-local caches.

The `task_name` value must be exactly `Brightness`, `Colorfulness`, or `Contrast`. Each full config uses a different `experiment.output_dir`.

`teacher_cache.source_cache_run_dir` may point at the existing MOS output directory. Only task-independent frozen vision/processor arrays are reused. Set it to `null` to build an independent cache.
