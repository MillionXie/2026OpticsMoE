# MNIST-4 单层 400×400 D2NN

本实验复现 `github_D2NN_mnist4.ipynb` 的四分类主路径：

```text
MNIST 0/1/2/3 灰度图
-> bicubic resize 336×336
-> 四周 zero padding 32 像素，得到 400×400 振幅场
-> 在同一平面加载一层 400×400 可训练相位
-> 自由空间传播 20 cm
-> 400×400 探测强度
-> 四个 50×50 方形区域积分
-> 四分类输出
```

没有电子 Linear、MLP、CNN 或可训练 readout。可训练参数只有一块 `400×400` 相位板，共 160,000 个参数。

## 输入调制

模板和本工程输入都是振幅调制，不是输入相位调制：

```text
E_input(x,y) = grayscale(x,y), phase_input(x,y) = 0
```

输入振幅场与第一块相位板位于同一平面，距离为 0。相位板随后执行：

```python
phase = 2.0 * pi * sigmoid(raw_phase)
E_after_mask = E_input * exp(1j * phase)
```

## 可配置输入与样本数

模板式输入由配置控制：

```yaml
dataset:
  input_size: 400
  preprocess_mode: resize_then_pad
  resize_size: 336
  interpolation: bicubic
```

如需旧版直接拉伸，可改为：

```yaml
dataset:
  input_size: 400
  preprocess_mode: direct_resize
  resize_size: 400
```

每类样本数也可独立设置：

```yaml
dataset:
  train_samples_per_class: 3000
  test_samples_per_class: 600
  use_full_dataset: false
```

将 `use_full_dataset` 设为 `true` 时忽略两个 per-class 限制，使用 MNIST 中 0/1/2/3 的全部样本。

## 损失与方形 detector target

模板实际训练使用整张探测面监督：

```yaml
loss:
  type: detector_plane_mse
  scale: 100.0
```

即：

```text
loss = 100 × MSE(detector_intensity, class_square_mask)
```

这会直接约束目标类别的 `50×50` 方形区域变亮。旧版仅使用区域积分交叉熵时，只约束区域总能量，不约束光斑必须呈方形，因此出现散斑并不代表 detector 方框定义错误。

`scale=100` 不等于把 Adam 学习率乘 100。Adam 同时维护梯度一阶矩和平方梯度二阶矩；梯度整体乘 100 后，分子约乘 100、分母约乘 100，更新量大体抵消。它主要保持与模板一致并让记录的 loss 数值更明显。对于普通 SGD 且只有这一项 loss、没有裁剪或正则化时，loss 乘 100 才近似等价于学习率乘 100。

如需旧版损失，可配置：

```yaml
loss:
  type: cross_entropy
  scale: 1.0
```

## 物理参数

| 参数 | 数值 |
|---|---:|
| wavelength | 532 nm |
| pixel size | 16 μm |
| canvas | 400×400 |
| trainable phase | 400×400 |
| input-to-mask distance | 0 cm |
| inter-layer distance | 3 cm（单层时不使用） |
| mask-to-detector distance | 20 cm |

## 可选 k 空间传播角约束

角谱传播可以配置圆形传播角截止：

```yaml
optics:
  k_space_constraint_enabled: true
  theta_max_deg: 0.5
```

其条件为：

```text
sqrt(kx² + ky²) <= k × sin(theta_max)
```

超过截止角的频率分量在传播前被置零。配置为 `false` 时不施加该人工角度截止，只保留原有传播波/倏逝波判断。当前 `532 nm + 16 μm` 采样的对角最大离散传播角约为 `1.35°`，因此 5°、10° 等阈值不会产生实际滤波。

三组直接对比配置为：

```text
configs/config_kspace_off.yaml
configs/config_kspace_theta0p5deg.yaml
configs/config_kspace_theta1p0deg.yaml
```

运行报告会保存 `theta_max_deg`、网格最大角、频率通过比例，并输出 `k_space_constraint_mask.png`。

`input_to_layer_distance_m` 可以正常控制输入到第一块相位板的传播：设为 0 时输入直接加载到相位板；改成 `0.03` 时会先执行 3 cm 角谱传播，再加载相位。如果角度约束开启，该输入传播段也会应用同样的 k 空间约束。单层结构中的 `inter_layer_distance_m` 仍然不会被使用。

四个 detector 区域是：

```text
class 0: y[75:125],  x[75:125]
class 1: y[75:125],  x[275:325]
class 2: y[275:325], x[75:125]
class 3: y[275:325], x[275:325]
```

这里 `det_steps_x=150` 和 `det_steps_y=150` 表示相邻 `50×50` 方块之间的空隙，所以下一个方块起点是 `75 + 50 + 150 = 275`，不是 225。

## 相位初始化

相位参数使用 Adam 时默认 `weight_decay=0`。不要把普通电子网络常用的 L2 weight decay 直接施加到 `raw_phase`：它会持续把 sigmoid 参数拉回 0，使有效相位重新接近空间常数 `π`，表现为准确率和相位图长期不动。训练日志会逐 epoch 打印并保存 `phase_std`、`phase_delta_mean` 和首批次 `grad_mean`，可直接判断相位是否真正更新。

三份配置中的初始化均指 `raw_phase`：

| 配置 | raw_phase 初始化 |
|---|---|
| `configs/config_phase_zero.yaml` | 全 0 |
| `configs/config_phase_uniform.yaml` | `Uniform(0, 2π)` |
| `configs/config_phase_gaussian.yaml` | `Normal(0, init_std)` |

有效相位始终为 `2π × sigmoid(raw_phase)`。因此 raw zero 对应空间常数相位 π，不会造成 sigmoid 梯度消失。

## 可视化输出

训练结束后保存：

```text
figures/training_curves.png
figures/confusion_matrix.png
figures/phase_masks/*/phase_mask_overlay.png
figures/detector_outputs/*/ideal_detector_mask_overlay.png
figures/light_fields/final_epoch/sample_*/sample_summary.png
```

每个最终测试样例都包含输入振幅、相位 overlay、探测面光强、方形 detector 边界以及四个区域能量柱状图。所有标量场图均有坐标轴和右侧 colorbar，并使用防裁切布局。

## 1920×1200 SLM BMP 导出

训练结束会加载准确率最高的 `best.pt`，保存：

```text
runs/<run_name>/slm_bmp_best/input_amplitude.bmp
runs/<run_name>/slm_bmp_best/phase_layer_01.bmp
runs/<run_name>/slm_bmp_best/manifest.json
```

当前 400×400、16 μm 平面以 nearest-neighbor 复制为 800×800、8 μm，再居中 zero pad 到宽 1920、高 1200 的灰度 BMP。相位按 `[0,2π)` 线性映射到 `[0,255]`，振幅按 `[0,1]` 映射到 `[0,255]`。
