# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 运行，不使用续行反斜杠。

## 四分类连续光学 MoE

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config opticalmoe/d2nn_cifar10_4class_moe9_5layer480/configs/config.yaml --device cuda
```

## 十分类连续光学 MoE

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config opticalmoe/d2nn_cifar10_4class_moe9_5layer480/configs/config_cifar10_10class.yaml --device cuda
```

## 四分类：独立专家 affine LayerNorm + ReLU

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config opticalmoe/d2nn_cifar10_4class_moe9_5layer480/configs/config_optoelectronic_interlayers_20cm.yaml --device cuda
```

## 十分类：独立专家 affine LayerNorm + ReLU

```bash
CUDA_VISIBLE_DEVICES=1 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/train.py --config opticalmoe/d2nn_cifar10_4class_moe9_5layer480/configs/config_cifar10_10class_optoelectronic_interlayers_20cm.yaml --device cuda
```

## 从已训练 run 导出 one-shot 实验 BMP

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_cifar10_4class_moe9_5layer480/export_oneshot_last_plane.py --run-dir opticalmoe/d2nn_cifar10_4class_moe9_5layer480/runs/cifar10_4class_moe9x5_optoelectronic_interlayers_20cm_seed7 --split test --samples-per-class 50 --batch-size 16 --num-workers 8 --device cuda
```

## 单元测试

```bash
python -m pytest opticalmoe/d2nn_cifar10_4class_moe9_5layer480/tests -q
```
