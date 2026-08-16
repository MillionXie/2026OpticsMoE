# 运行命令

以下命令都在仓库根目录执行。正式配置直接在 Caltech101 目标 10 类上随机初始化电子学生；不要传入 101 类或光学模型的 `--resume-checkpoint`。

## 1. 检查数据划分

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase prepare_data
```

预期为 300 张训练图、200 张测试图、30 张 gallery 图。

## 2. 检查或生成教师 cache

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase cache_teacher_embeddings
```

配置复用 `qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval/cache/caltech101_target10_seed42_g3_train30_test20`。cache 是冻结 Qwen 教师对这 530 张图的 64 维监督，不是学生预训练权重；有效时不会重新计算。

## 3. 从零训练电子学生

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase train
```

训练每个 epoch 都打印 `train_top1`、`test_top1` 和 `ema_test_top1`。若中断后确实要续训，才额外加入本工程 checkpoint 的 `--resume-checkpoint <path>`。

## 4. 固定 checkpoint 评测

推荐先评测 EMA 的最低训练损失 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/runs/caltech101_target10_electronic_direct/ema_best_train_loss_checkpoint.pt
```

## 5. 生成检索可视化

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase visualize
```

## 6. 一步冒烟测试

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/smoke/electronic_target10_smoke.yaml \
  --phase all
```
