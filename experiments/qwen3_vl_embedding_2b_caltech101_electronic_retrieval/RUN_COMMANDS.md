# 运行命令

以下命令都在仓库根目录执行。正式配置使用约 2625 张训练图直接随机初始化纯 MLP 电子学生；训练不需要教师 cache，不要传入 101 类或光学模型的 `--resume-checkpoint`。

## 1. 检查数据划分

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase prepare_data
```

预期为 2625 张训练图、200 张测试图、30 张 gallery 图。

## 2. 从零训练电子学生

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase train
```

训练每个 epoch 都打印 `train_top1`、`test_top1` 和 `ema_test_top1`。若中断后确实要续训，才额外加入本工程 checkpoint 的 `--resume-checkpoint <path>`。

## 3. 固定 checkpoint 评测

推荐先评测 EMA 的最低训练损失 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/runs/caltech101_target10_mlp_all_data/ema_best_train_loss_checkpoint.pt
```

## 4. 生成检索可视化

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic.yaml \
  --phase visualize
```

## 5. 一步冒烟测试

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/smoke/electronic_target10_smoke.yaml \
  --phase all
```
