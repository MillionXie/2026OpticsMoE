# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 直接运行。

四分类 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config.yaml --smoke_test
```

路由物理验证：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/validate_routing.py --config configs/config.yaml --device cuda --smoke-test
```

四分类完整训练：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config.yaml
```

十分类训练：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config_cifar10_10class.yaml
```

十分类并将 batch 临时改为 32：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config_cifar10_10class.yaml --batch-size 32
```

每轮每类只取 500 张，但在多轮间轮转覆盖完整训练集：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config_cifar10_10class.yaml --train-samples-per-class-per-epoch 500
```

`train_samples_per_class` 是底层数据集永久上限；`train_samples_per_class_per_epoch` 是轮转的单轮预算；`batch_size` 是一次 optimizer step 的样本数。

四分类 AdamW + importance 均衡约束实验（独立 run，不覆盖原结果）：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config_importance_adamw.yaml
```

十分类 AdamW + importance 均衡约束实验：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config_cifar10_10class_importance_adamw.yaml
```

四分类、专家层间20 cm光电转换实验：

```bash
CUDA_VISIBLE_DEVICES=1 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config configs/config_optoelectronic_interlayers_20cm.yaml
```
