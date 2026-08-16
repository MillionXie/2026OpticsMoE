# 运行命令

以下命令均在仓库根目录执行。

## 1. 下载、导出并检查数据划分

```bash
python -m experiments.qwen3_vl_embedding_2b_cifar10_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_cifar10_electronic_retrieval/configs/release/cifar10_electronic_vision2d.yaml \
  --phase prepare_data
```

预期为 49,970 张训练图、10,000 张测试图和 30 张 gallery 图。

## 2. 冒烟测试

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.qwen3_vl_embedding_2b_cifar10_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_cifar10_electronic_retrieval/configs/smoke/cifar10_electronic_vision2d_smoke.yaml \
  --phase all
```

## 3. 正式训练

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_cifar10_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_cifar10_electronic_retrieval/configs/release/cifar10_electronic_vision2d.yaml \
  --phase train
```

## 4. 固定 EMA checkpoint 评测

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_cifar10_electronic_retrieval \
  --config experiments/qwen3_vl_embedding_2b_cifar10_electronic_retrieval/configs/release/cifar10_electronic_vision2d.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_embedding_2b_cifar10_electronic_retrieval/runs/cifar10_electronic_vision2d_no_deepstack/ema_best_train_loss_checkpoint.pt
```
