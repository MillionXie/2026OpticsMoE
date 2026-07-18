# Run commands

Commands are written as single lines for execution from the repository root.

## Brightness

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9/configs/spaq_brightness.json --phase all
```

## Colorfulness

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9/configs/spaq_colorfulness.json --phase all
```

## Contrast

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9/configs/spaq_contrast.json --phase all
```

The three commands may run concurrently on different GPUs. They read the same frozen source cache but write to different output directories.

## Smoke and tests

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9/configs/spaq_brightness_smoke.json --phase all
```

```bash
python -m pytest experiments/qwen3_vl_2b_spaq_single_attribute_vision_homogeneous_moe9/tests -q
```

## If the MOS visual cache is missing

Set `teacher_cache.source_cache_run_dir` to `null`, then run the normal `--phase all` command. The task will create its own teacher and processor caches before training.
