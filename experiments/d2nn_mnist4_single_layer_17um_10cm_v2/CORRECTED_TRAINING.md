# MNIST-4 v2 修正版训练说明

## 为什么旧版会长期停在约 43%

旧版分别在 59×59 正确区域和其余 225003 个背景像素上各自求均值，再以
`1.0:0.5` 相加。这样虽然文字上有“背景损失”，但背景面积信息被第二次均值消掉，
其真实贡献不到 1%。旧版 best checkpoint 的验证混淆矩阵中，类别 0 只识别对
`6/592`，因此不是健康的四分类结果。

## 修正版目标

修正版直接复现 notebook 的目标形式：

```text
target[y, x] = 1  （正确的 59×59 探测区）
target[y, x] = 0  （其余全部像素）
loss = 100 × mean((raw_ccd_intensity - target)^2)
```

这个 mean 同时覆盖 batch 和完整 478×478 平面，因此背景自然按真实像素数参与。
`target_region_mse` 与 `background_mse` 仍写入日志，但只作诊断，不再独立加权。
Notebook 原实验用 20 cm；本工程为新光路固定使用 10 cm。因此不能只按
`478/400` 缩放位置，还必须补偿传播距离和像素间距造成的偏转角变化。

## 探测区域

Notebook 在 400×400、16 μm、20 cm 上使用 `[75,125)` 和 `[275,325)`。
直接按 `478/400` 映射得到的旧位置是 `[90,149)` / `[329,388)`；在新的
17 μm、10 cm 系统中，四个中心需要约 1.65° 的径向偏转，超过当前 1.10°
k 空间通带，也超出 17 μm 采样在单轴上的可用偏转范围。这正是旧运行长期只有
约 70%–77%、背景亮而目标区不明显的主要原因。

正式版保留当前 478 平面所需的 59×59 区域大小，但按 notebook 中心的传播角度
重新映射：

```text
offset_new_px = offset_notebook_px
                × 16 μm × 0.10 m / (17 μm × 0.20 m)
              = offset_notebook_px × 0.470588...
```

最终区域为：

```text
class 0: x=[162,221), y=[162,221)   top-left
class 1: x=[257,316), y=[162,221)   top-right
class 2: x=[162,221), y=[257,316)   bottom-left
class 3: x=[257,316), y=[257,316)   bottom-right
```

每个区域都是 59×59；中心为 191.5 或 286.5。最外角只需约 1.06° 的径向偏转，
位于 1.10° 通带内。类别顺序、x/y 轴及裁剪链均已与 notebook 逐项核对，没有
0/3 互换。CCD 必须提供严格 478×478 的同一坐标画面，评估器不会 resize。

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
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_angle_roi.yaml
```

旧的比例位置与 CE 配置只保留为诊断消融，不再作为硬件默认。CE=1.0 虽可提高约
2.5 个百分点，却显著降低目标/全平面能量占比并抬高背景；正式模型不采用。

旧诊断配置：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_mse_ce_corner_kspace.yaml
```

它只增加 `0.25 × CE(log(raw four ROI sums), class)`。CE 仅用于训练目标；不会
写入硬件推理图，也不会改变 raw CCD、四区求和或 argmax 判别。
