# CIFAR-10 homogeneous optical MoE

该工程使用 3×3 同构 D2NN 专家、输入相关 top-k router、五层专家相位板、共享 global phase 和 detector-region readout，支持 CIFAR-10 四分类和十分类。

## Optical layout

- canvas：480×480
- active region：450×450
- experts：3×3
- each expert：120×120
- expert pitch：150 pixels
- expert phase stages：5
- global phase：450×450
- wavelength：532 nm
- simulation pixel size：16 μm

连续相干传播配置不包含层间探测、LayerNorm 或 ReLU。

## Unified affine OEO

四分类和十分类分别使用：

- `configs/config_optoelectronic_interlayers_20cm.yaml`
- `configs/config_cifar10_10class_optoelectronic_interlayers_20cm.yaml`

每个 expert stage 后执行：

```text
complex expert fields
-> 20 cm propagation
-> square-law intensity
-> independent LayerNorm statistics for each sample/expert
-> independent gamma/beta for each stage/expert
-> ReLU
-> zero-phase amplitude reload
```

归一化只处理九个 120×120 专家区域，不把整张 480×480 canvas 混合计算统计量；归一化后不重新乘回 routing amplitude。

Affine 参数量：

```text
5 stages × 9 experts × 2 affine maps × 120 × 120 = 1,296,000
```

## Parameter groups

```text
expert phase masks: 9 × 5 × 120 × 120 = 648,000
global phase: 450 × 450 = 202,500
optical phase total: 850,500
router: 909 electronic parameters
affine OEO configs: 1,296,000 electronic gamma/beta parameters
```

连续传播配置的 OEO affine 参数为 0。

## One-shot final-plane export

`export_oneshot_last_plane.py` 用于真实光路的最后一级验证。它从已训练 checkpoint 精确恢复对应旧/新 OEO 结构，按类别筛选 detector margin 较高的正确样本，并导出：

```text
第 5 个专家相位面
-> 20 cm propagation
-> square detection / LayerNorm / ReLU
-> zero-phase amplitude reload (1920×1080 BMP)
-> co-planar global phase (shared 1920×1200 BMP)
-> 20 cm propagation
-> four-region detector
```

振幅和 global phase 在模型中直接相乘，两者之间没有传播。导出器还会用 8-bit 振幅和相位重新仿真，只保留量化后仍正确分类的样本。具体命令见 `COMMANDS.md`。
