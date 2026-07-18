# Commands

Run these from the repository root. Commands are deliberately single-line and contain no shell continuation backslashes.

## Prepare SPAQ

```bash
python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase prepare_data
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos_smoke.json --phase all
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase all
```

## Separate phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase teacher_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase teacher_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase teacher_predictions
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase student_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json --phase student_inference
```

## Tests

```bash
python -m pytest experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/tests -q
```
