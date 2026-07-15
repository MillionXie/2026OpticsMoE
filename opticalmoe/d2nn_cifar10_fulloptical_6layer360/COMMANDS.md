# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 运行，不使用续行反斜杠。

## 四分类连续全光

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config opticalmoe/d2nn_cifar10_fulloptical_6layer360/configs/cifar10_4class.yaml --device cuda
```

## 十分类连续全光

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config opticalmoe/d2nn_cifar10_fulloptical_6layer360/configs/cifar10_10class.yaml --device cuda
```

## 四分类：逐层独立 affine LayerNorm + ReLU

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config opticalmoe/d2nn_cifar10_fulloptical_6layer360/configs/cifar10_4class_optoelectronic_interlayers_20cm.yaml --device cuda
```

## 十分类：逐层独立 affine LayerNorm + ReLU

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_cifar10_fulloptical_6layer360/train.py --config opticalmoe/d2nn_cifar10_fulloptical_6layer360/configs/cifar10_10class_optoelectronic_interlayers_20cm.yaml --device cuda
```

## 单元测试

```bash
python -m pytest opticalmoe/d2nn_cifar10_fulloptical_6layer360/tests -q
```
