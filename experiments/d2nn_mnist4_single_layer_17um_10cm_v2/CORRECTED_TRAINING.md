# MNIST-4 v2 修正版训练说明

## 为什么旧版会长期停在约 43%

旧版分别在 59×59 正确区域和其余 225003 个背景像素上各自求均值，再以
`1.0:0.5` 相加。这样虽然文字上有“背景损失”，但背景面积信息被第二次均值消掉，
其真实贡献不到 1%。旧版 best checkpoint 的验证混淆矩阵中，类别 0 只识别对
`6/592`，因此不是健康的四分类结果。

## 修正版目标

修正版直接复现 notebook 的目标形式，只把坐标按 `478/400` 映射：

```text
target[y, x] = 1  （正确的 59×59 探测区）
target[y, x] = 0  （其余全部像素）
loss = 100 × mean((raw_ccd_intensity - target)^2)
```

这个 mean 同时覆盖 batch 和完整 478×478 平面，因此背景自然按真实像素数参与。
`target_region_mse` 与 `background_mse` 仍写入日志，但只作诊断，不再独立加权。
Notebook 原实验用 20 cm；本工程为新光路固定使用 10 cm，这是一项有意的物理差异。

## 探测区域

Notebook 在 400×400 上使用 `[75,125)` 和 `[275,325)`。逐边按 478/400
比例、round-half-up 映射后为：

```text
class 0: x=[90,149),  y=[90,149)    top-left
class 1: x=[329,388), y=[90,149)    top-right
class 2: x=[90,149),  y=[329,388)   bottom-left
class 3: x=[329,388), y=[329,388)   bottom-right
```

每个区域都是 59×59；中心为 119.5 或 358.5。类别顺序、x/y 轴及裁剪链均已与
notebook 逐项核对，没有 0/3 互换或半像素偏移。CCD 必须提供严格 478×478 的
同一坐标画面，评估器不会 resize。

## 鲁棒性

- 推荐 k 空间截止角为 1.10°，在 17 μm/532 nm/1024 网格上保留 96.29%
  频谱，只滤除方形采样频谱的高频角落；0.80°（保留 62.54%）仅保留为强滤波对照；
- 前 8 epoch 不注入随机错位；
- 第 9 epoch 起，输入、相位、CCD 前复光场分别以 50% 概率做最多 1 像素的
  上/下/左/右零填充平移；
- 验证、测试和 BMP 导出始终关闭随机错位。

CCD 光电探测 `|E|²` 之后没有归一化、激活、log、截断、背景扣除或 resize。
背景抑制完全来自训练目标，而不是实测后的电子补偿。

正式配置：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_corner_kspace.yaml
```
