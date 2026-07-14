# Commands

所有命令从仓库根目录运行。

## Help

```bash
python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_linear/train.py --help
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_linear/train.py --config configs/config.yaml --smoke-test --run-name heterogeneous_moe9_deep_linear_smoke
```

## Full training

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_linear/train.py --config configs/config.yaml
```

## Rotating 1000 samples per class per epoch

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_linear/train.py --config configs/config.yaml --train-samples-per-class-per-epoch 1000
```

## Tests

```bash
python -m pytest opticalmoe/d2nn_cifar10_4class_heterogeneous_moe9_deep_linear/tests -q
```

