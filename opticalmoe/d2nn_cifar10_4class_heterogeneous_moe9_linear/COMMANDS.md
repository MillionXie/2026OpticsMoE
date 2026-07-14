# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 运行，不需要先进入实验目录。

## Help

```bash
python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_linear/train.py --help
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_linear/train.py --config configs/config.yaml --smoke-test --run-name heterogeneous_moe9_linear_smoke
```

## Full training

```bash
CUDA_VISIBLE_DEVICES=2 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_linear/train.py --config configs/config.yaml
```

## Full dataset retained, rotating 1000 samples per class per epoch

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_linear/train.py --config configs/config.yaml --train-samples-per-class-per-epoch 1000
```

## Tests

```bash
python -m pytest opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_linear/tests -q
```

