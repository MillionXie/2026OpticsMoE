# 与 `github_D2NN_mnist4.ipynb` 的逐项核对

## 结论

当前默认配置已对齐模板的输入尺寸、数据抽样、优化器、损失、相位约束、传播距离和 detector 区域。传播公式采用数学等价的未 shift 频谱写法，测试会直接比较两种写法的数值输出。

| 项目 | Notebook 模板 | 当前默认配置 |
|---|---|---|
| classes | MNIST 0/1/2/3 | 相同 |
| resize | 336×336 bicubic | 相同，可配置 |
| padding | zero pad to 400×400 | 相同，可配置 |
| input encoding | 灰度值作为实数振幅 | 相同 |
| input phase | 0 | 相同 |
| input-to-mask | 0 cm，直接调制 | 相同 |
| phase layers | 1 | 相同 |
| raw phase | 可训练 400×400 | 相同 |
| effective phase | `2π sigmoid(raw_phase)` | 相同 |
| phase-to-detector | 20 cm | 相同 |
| pixel pitch | 16 μm | 相同 |
| wavelength | 532 nm | 相同 |
| detector | 四个 50×50 区域 | 相同 |
| train/class | 3000 | 相同，可配置 |
| test/class | 600 | 相同，可配置 |
| optimizer | Adam, lr=0.01 | 相同 |
| active loss | `100 × full-plane MSE` | 相同，可配置 |
| electronic readout | 无 | 无 |

## 输入到相位板的距离及调制类型

模板的第一行光学运算是：

```python
E = E * exp(1j * 2π * sigmoid(raw_phase))
```

因此模板没有在输入与相位板之间传播。输入图像是非负实数灰度值，对应振幅调制；随后相位板进行相位调制。本工程路径完全相同。零距离传播模块现在直接返回输入，不再执行多余的 FFT/IFFT。

## 角谱传播写法

模板使用：

```text
fft2 -> fftshift -> shifted transfer function -> ifftshift -> ifft2
```

本工程使用：

```text
fft2 -> unshifted transfer function -> ifft2
```

两者只是频率数组排列不同，数学等价。测试 `test_unshifted_propagation_matches_notebook_shifted_fft_convention` 对两种实现做数值比较。

模板对 evanescent component 使用指数衰减，本工程配置为清零。但在 532 nm 波长与 16 μm 像素间距下，离散频率范围远小于 `1/λ`，当前 400×400 网格没有 evanescent frequency，因此该分支不影响本实验。

当前工程已像 Notebook/NumPy 一样用 float64 计算频率坐标和大相位，再将传播核保存为 complex64 执行 FFT。这样避免 float32 构造传播相位造成的明显误差，同时把显存控制在 complex64 水平。Notebook 的传播核最终可能保持 complex128，因此两者仍不是逐 bit 相同，但传播核相位与输出已在单元测试中按容差核对。

新增的 k 空间角度约束默认关闭，因此 `config_phase_zero/uniform/gaussian.yaml` 仍保持模板传播。只有使用 `config_kspace_theta*.yaml` 或手工开启 `k_space_constraint_enabled` 时，才会额外抑制超过 `theta_max_deg` 的角谱分量；这是新增对照实验，不是原 Notebook 的默认行为。

## 探测与损失

两者都计算：

```text
I = |E|²
region_score_c = sum(I inside region c) / sum(I over full detector plane)
prediction = argmax(region_score)
```

模板的 `det_steps_x/y=150` 是方块之间的空隙，因此四个左上角为 `(75,75)`、`(75,275)`、`(275,75)`、`(275,275)`。此前将 150 误当成左上角 stride，得到 225；现已修正。

此前工程与模板最大的训练差异不是探测器，而是损失：此前只对四个 `region_score` 做交叉熵；模板实际把完整 `400×400` intensity 与类别方形 mask 做 MSE。现在默认已改成模板的 full-plane MSE。

这也解释了此前图片中的现象：

- `ideal_detector_mask_overlay.png` 中的四个区域一定是规则方形；
- 实际 detector intensity 是否形成方块由损失和训练效果决定；
- 仅用区域交叉熵时，方框内部出现不规则散斑完全允许；
- full-plane MSE 才会显式推动输出接近方形目标。

## 可视化核对

现在分别保存：

```text
ideal_detector_mask_overlay.png       # detector 的几何定义
detector_intensity_with_regions.png   # 真实光强并叠加方框
detector_region_energy_bar.png        # 四个区域的积分结果
phase_mask_overlay.png                # 输入振幅与相位分布叠加
```

这样不会再把“理想探测区域”与“传播后的实际光斑形状”混为一谈。
