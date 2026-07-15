# CIFAR-10 6-layer full-field optical baseline

该工程支持 CIFAR-10 四分类和十分类，不使用 router、prompt、专家、CNN 或电子分类头。

## 连续全光版本

```text
grayscale image
-> resize 300x300
-> centered zero padding to 360x360
-> six 360x360 phase-only planes
-> final square-law detector regions
```

连续版本配置为 `cifar10_4class.yaml` 和 `cifar10_10class.yaml`，不包含层间探测或非线性。

## 逐层 affine OEO 版本

- `configs/cifar10_4class_optoelectronic_interlayers_20cm.yaml`
- `configs/cifar10_10class_optoelectronic_interlayers_20cm.yaml`

前五次层间传播后执行：

```text
complex field
-> square-law intensity
-> per-sample spatial LayerNorm over 360x360
-> stage-specific trainable gamma/beta
-> ReLU
-> zero-phase amplitude reload
```

五个 OEO stage 拥有相互独立的 360x360 gamma/beta：

```text
5 stages x 2 affine maps x 360 x 360 = 1,296,000 parameters
```

该数量与同构/异构 MoE 的 `5 x 9 x 2 x 120 x 120` 完全一致。全光模型没有 routing amplitude，因此不存在重新乘回 routing weight 的步骤。第六张相位板之后只传播到最终平方律探测器，不再执行 OEO。

## Physical settings

- wavelength: 532 nm
- simulation pixel size: 16 um
- input-to-first-layer distance: 0 m
- OEO configs inter-layer distance: 20 cm
- OEO configs final detector distance: 20 cm
- phase masks: `6 x 360 x 360 = 777,600` trainable optical parameters

运行命令见 `COMMANDS.md`。
