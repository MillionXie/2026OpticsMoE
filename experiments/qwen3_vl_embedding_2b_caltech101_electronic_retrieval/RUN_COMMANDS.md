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

## 6. 有教师 cosine KD 的严格消融

这一组与无教师版本使用完全相同的网络、2625/200/30 数据、采样器、学习率和 60 epoch；唯一新增项是 `1.0 × cosine embedding KD`。首次运行先生成一次全量教师 cache：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_mlp_teacher_kd.yaml \
  --phase cache_teacher_embeddings
```

cache 会稳定保存在本工程的 `cache/caltech101_target10_all_data_seed42_g3_test20/teacher_embeddings.pt`；后续重新训练无需再次生成。

训练：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_mlp_teacher_kd.yaml \
  --phase train
```

固定 EMA checkpoint 评测：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_mlp_teacher_kd.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/runs/caltech101_target10_mlp_all_data_teacher_kd/ema_best_train_loss_checkpoint.pt
```

## 7. 无教师 Electronic V2：2 层 1D token mixer + mean/max pooling

这一组在 Vision 和 Language 中分别使用 2 个 mixer block，不读取 teacher cache，不使用任何教师 loss、attention、MoE 或多层 Qwen 特征。它从零训练 10 类学生，并保留原始 MLP 配置作为对照。最初的每模态 3-block 运行仍由 `caltech101_target10_electronic_token_mixer_3layer.yaml` 保存，不要用 2-block checkpoint 与其交叉续训。

先做一轮冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/smoke/electronic_target10_token_mixer_smoke.yaml \
  --phase all
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic_token_mixer.yaml \
  --phase train
```

固定 EMA checkpoint 评测：

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/configs/release/caltech101_target10_electronic_token_mixer.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/runs/caltech101_target10_electronic_token_mixer_2layer/ema_best_train_loss_checkpoint.pt
```
