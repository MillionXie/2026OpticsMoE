# Commands

所有命令均从仓库根目录 `2026OpticsMoE` 执行；命令中不含续行反斜杠。

## 正式流程

自动下载并核验 COCO 与 DUTS：

```bash
python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase prepare_data
```

拟合离线 PCA：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase fit_pca
```

检查 PCA oracle：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase pca_oracle_check
```

缓存完整 COCO teacher PCA224 target（可断点续作）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase precompute_teacher
```

训练 COCO 通用 optical backbone：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase coco_pretrain
```

加载 COCO backbone，完成 DUTS 5 epoch head warmup + 50 epoch joint fine-tune：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase duts_train
```

单独测试 train-loss-selected DUTS checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase duts_test
```

完整一键流程：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain.yaml --phase all
```

## Smoke

如果官方数据已经下载，可运行小样本完整 smoke。若数据尚不存在，此命令
也会自动下载完整 ZIP，再只取少量样本：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain --config experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/configs/coco_duts_pretrain_smoke.yaml --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/tests -q
```
