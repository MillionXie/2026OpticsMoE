# 运行命令

以下命令均从仓库根目录执行。

## 当前 V2 结果的一键验证

服务器已存在完整 V2 checkpoint 时，运行：

```bash
bash FixedFeedbackSFT/commands/01_verify_v2_results.sh
```

该脚本只运行测试、重新聚合和校验结果文件，不会重新训练或删除 checkpoint。

## 性能优先新实验接口

新 backbone 尚未实现。计划中的模块名、CLI 和必须提供的 `commands/` 文件见：

```text
FixedFeedbackSFT/commands/02_performance_first_runbook.md
```

在代码实现前，不要把 runbook 中的接口当作可运行结果。

## V1：已完成分类版

单元测试：

```bash
python -m pytest experiments/d2nn_cifar100c10_fixed_feedback_20stage400/tests -q
```

重新聚合已有结果：

```bash
python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 --config experiments/d2nn_cifar100c10_fixed_feedback_20stage400/configs/main.yaml --phase compare
```

完整分组命令见：

`experiments/d2nn_cifar100c10_fixed_feedback_20stage400/commands/COMMANDS.md`

## V2：CIFAR-100 → CIFAR-10 对比迁移

以下正式训练已在服务器完成。默认命令会复用 complete/checkpoint；只有明确需要
重跑时才应使用代码提供的 force 选项。

单元测试：

```bash
python -m pytest experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/tests -q
```

轻量 smoke：

```bash
python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --phase smoke
```

正式尺寸单 batch CUDA 检查：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase formal_smoke
```

准备数据：

```bash
python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase prepare_data
```

共享 CIFAR-100 对比预训练：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase pretrain
```

NoFT：

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase no_finetune
```

BP：

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase finetune --method bp
```

FA-pretrained：

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase finetune --method fa_pretrained
```

FA-random：

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase finetune --method fa_random
```

聚合比较：

```bash
python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase compare
```

单个诊断 seed 可在微调命令末尾追加：

```text
--seed 1234
```
