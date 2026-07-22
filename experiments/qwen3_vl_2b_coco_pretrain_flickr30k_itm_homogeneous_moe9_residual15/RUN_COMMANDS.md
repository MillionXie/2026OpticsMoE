# Commands

所有命令均从仓库根目录执行，单行复制，不含续行反斜杠。

## 检查数据与 manifest

```bash
python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase prepare_data
```

## 通用 COCO 蒸馏

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase generic_input_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase generic_teacher_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase generic_pretrain
```

## Flickr30k 微调

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase flickr_teacher_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase flickr_teacher_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase flickr_teacher_logits
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase flickr_finetune
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase flickr_inference
```

## 全流程

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k.json --phase all
```

## SAM 消融

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --config experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/configs/coco_pretrain_flickr30k_sam.json --phase all
```

## Smoke 与测试

```bash
python -m experiments.qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15 --help
```

```bash
pytest experiments/qwen3_vl_2b_coco_pretrain_flickr30k_itm_homogeneous_moe9_residual15/tests -q
```
