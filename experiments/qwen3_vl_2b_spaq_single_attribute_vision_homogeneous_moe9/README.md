# Qwen3-VL-2B SPAQ Single-Attribute Vision Optical MoE

This independent experiment evaluates three SPAQ attributes separately:

- `Brightness`
- `Colorfulness`
- `Contrast`

每个配置都是一个独立的单任务回归实验。三项任务不会放进同一个 batch、不会共享回归头，也不是多任务学习。MOS 原实验不受修改。

## Model

Teacher:

```text
RGB image
-> frozen Qwen3-VL-2B patch embedding
-> complete frozen electronic vision transformer stack
-> valid-token mean pooling
-> LayerNorm(1024) -> Linear(1024,1)
-> selected SPAQ attribute score
```

Student:

```text
RGB image
-> frozen Qwen3-VL-2B patch embedding
-> homogeneous optical MoE9x5 full vision-stack replacement
-> valid-token mean pooling
-> an independently initialized head with the same structure as the teacher
-> selected SPAQ attribute score
```

The Qwen language model, tokenizer, prompt, and chat template are not used. Labels are divided by 100 for SmoothL1 training and restored to the original 0-100 scale for MAE/RMSE. SRCC and PLCC are also reported.

## Safe visual-cache reuse

The image split, frozen Qwen visual hidden, and processor tensors do not depend on which scalar attribute is predicted. The full configs therefore point `teacher_cache.source_cache_run_dir` at the completed MOS run and reuse only:

- frozen electronic visual hidden states;
- Qwen processor `pixel_values` and `image_grid_thw`.

Current attribute targets always come from the selected SPAQ column. MOS targets stored in the legacy cache are deliberately ignored. Each task has its own output directory, teacher head, teacher predictions, optical weights, checkpoints, and metrics. Cache reuse is rejected if the split digest, model, pixel budget, sample count, or input geometry differs.

Set `teacher_cache.source_cache_run_dir` to `null` if the MOS cache is unavailable. Then run `--phase all` or `teacher_precompute`; the experiment builds its own task-local visual cache.

## Interpretation

Compare `metrics/teacher_inference.json` with `metrics/student_inference.json` inside each task run. The primary ranking metric is SRCC; MAE is on the original 0-100 score scale. The configured best student checkpoint is selected on the test SRCC for compatibility with the source experiment, so it is explicitly marked selection-biased. The final/last checkpoint remains available for a stricter fixed-epoch report.
