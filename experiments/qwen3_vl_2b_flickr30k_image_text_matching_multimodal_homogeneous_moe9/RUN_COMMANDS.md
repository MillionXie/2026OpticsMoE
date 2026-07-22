# Commands

Run commands from the repository root. They are intentionally one line each for direct terminal copy/paste.

## Main vision + language optical MoE9 run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase all
```

## Diagnostic vision optical + electronic language run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_electronic_language.json --phase all
```

## Smoke runs

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical_smoke.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_electronic_language_smoke.json --phase all
```

## Separate phases

Override the rotating per-class epoch size without editing JSON:

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase student_train --train-samples-per-class-per-epoch 2000
```

```bash
python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase prepare_data
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase teacher_precompute
```

```bash
python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase teacher_train
```

```bash
python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase teacher_logits
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase student_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/configs/flickr30k_itm_vision_language_optical.json --phase student_inference
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9/tests -q
```
