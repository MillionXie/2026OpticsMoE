# Commands

所有命令均从仓库根目录 `2026OpticsMoE` 执行，命令中不使用反斜杠续行。

## Install the dataset downloader

```bash
python -m pip install gdown
```

## Prepare FSS-1000

```bash
python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase prepare_data
```

## Strict shape smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_smoke.yaml --phase shape_smoke
```

## Electronic Teacher

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase teacher_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase teacher_test
```

## Optical Student

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase student_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase student_test
```

## Optional mask-logit KD

先使用普通配置训练 Teacher，然后构建无几何增强的最终 mask logit cache：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_mask_kd.yaml --phase cache_teacher_masks
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_mask_kd.yaml --phase student_train
```

KD smoke（复用普通 smoke 已训练的 Teacher）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_mask_kd_smoke.yaml --phase all
```

## One-command smoke / full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency_smoke.yaml --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/configs/fss1000_saliency.yaml --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/tests -q
```
