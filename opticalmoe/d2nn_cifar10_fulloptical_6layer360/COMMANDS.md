# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 直接运行。

四分类 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_4class_smoke.yaml
```

十分类 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_10class_smoke.yaml
```

四分类完整训练：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_4class.yaml
```

四分类、层间20 cm光电转换对照：

```bash
CUDA_VISIBLE_DEVICES=2 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_4class_optoelectronic_interlayers_20cm.yaml
```

十分类完整训练：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_10class.yaml
```

十分类并将 batch 临时改为 32：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_10class.yaml --batch-size 32
```

保留完整训练集，但每轮每类仅取 500 张：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config configs/cifar10_10class.yaml --train-samples-per-class-per-epoch 500
```
