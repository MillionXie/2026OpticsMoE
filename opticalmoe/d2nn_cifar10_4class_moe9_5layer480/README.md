# CIFAR-10 四分类纯光 OpticalMoE（9 experts × 5 layers）

本工程以 `d2nn_mnist4_amp_1phase400` 的角谱传播、相位约束、探测积分、整面 MSE 和可视化为物理基线，只参考 `00-template_single_task` 的 MoE 结构组织。没有复用旧工程的像素尺寸或旧传播核。

## 光路

```text
CIFAR-10 airplane/automobile/bird/cat
→ RGB 转灰度振幅
→ nearest-neighbor resize 100×100
→ zero pad 10，得到 120×120
→ 放置到 480×480 中央
→ 自由传播 5 cm
→ 输入相关的 top-k 全局 prompt，450×450 有效、外圈 zero pad 15
→ 自由传播 15 cm 到专家阵列
→ 3×3 共 9 个专家，每个 120×120，pitch=150，间隙=30
→ 每个专家共 5 层，相邻层传播 5 cm
→ 第五层传播 5 cm 到 450×450 global FC
→ global FC 传播 10 cm 到 detector
→ 四个 50×50 方形区域积分
→ 纯光四分类输出
```

没有电子分类 readout。专家相位参数量：

```text
9 × 5 × 120 × 120 = 648,000
```

global FC 相位参数为 `450×450 = 202,500`，因此可训练光学相位总量为 `850,500`。

所有相位参数默认使用 `weight_decay=0`。对 `raw_phase` 使用普通 L2 weight decay 会把相位板压回空间常数并显著抑制训练。epoch 日志和 CSV 会记录 `phase_std`、`phase_delta_mean` 与 `grad_mean`，用于确认专家相位和 global FC 是否在更新。

## 输入相关 Top-k routing

router 使用标准稀疏 MoE 门控：

```text
输入灰度图
→ AdaptiveAvgPool2d(10×10)
→ Linear(100→9)
→ softmax
→ top-k（默认 k=3）
→ 只保留被选权重并重新归一化
```

router 是唯一的电子模块，共 `100×9+9=909` 个参数；它只负责选择专家，不参与最终分类 readout。未选择专家的 prompt grating complex amplitude 严格设为 0，等价于该光栅通道不参与复振幅叠加。prompt 不加入随机相位，phase biases 固定为 0。

`02_prompt_amplitude_and_routing_weights.png` 左侧出现细密干涉纹是正常现象：这里显示的是选中光栅通道在同一全局 prompt 平面上的复振幅叠加，并不是把九个专家画成九个方块。右侧柱状图才直接表示本样本的稀疏路由系数，恰有 `top_k` 根非零柱。专家相位板可以作为整面 mosaic 常驻加载；未选专家对应的 prompt 光栅系数为零，因此没有主动路由到该专家的入射光束，无需逐样本卸载专家相位板。

训练损失额外包含标准 MoE balance loss：

```text
L = L_detector + router_balance_weight × 9 × sum(importance × load)
```

epoch CSV 会记录每个专家的选择率和平均 routing weight；prompt amplitude 图片右侧也会显示九个专家的柱状权重。

在 16 μm 采样下，把光路由到 ±150 像素需要至少约 14.4 cm。默认 `prompt_to_expert=15 cm`，避免原先 5 cm 导致的光栅频率越过 Nyquist；输入到 prompt 和专家层间仍是 5 cm。

## 几何

| 项目 | 数值 |
|---|---:|
| simulation canvas | 480×480 |
| active prompt / expert layout | 450×450 |
| outer padding | 15 |
| expert size | 120×120 |
| expert pitch | 150 |
| expert gap | 30 |
| expert centers | 90/240/390 |
| wavelength | 532 nm |
| simulation pixel pitch | 16 μm |

专家在 480 画布中的起点为：

```text
y/x = 30, 180, 330
```

因此每个 pitch 单元两侧各有 15 像素空隙，相邻专家有效相位之间总间隙为 30。

## Detector

四个 detector 以 480×480 中心 `(240,240)` 对称排布：

```text
class 0: y[115:165], x[115:165]
class 1: y[115:165], x[315:365]
class 2: y[315:365], x[115:165]
class 3: y[315:365], x[315:365]
```

输出直接为四个区域相对全探测面的积分能量，不经过电子层。默认损失保持 D2NN 模板形式：

```text
100 × MSE(detector intensity, class square mask)
```

## 配置

工程只保留 `configs/config.yaml`。默认 `raw_phase` 全 0，有效相位始终为：

```python
2.0 * pi * sigmoid(raw_phase)
```

K 空间约束默认关闭。配置字段旁有注释；需要时将 `k_space_constraint_enabled` 改为 `true` 并设置 `theta_max_deg`。

## 1920×1200 SLM BMP

训练结束会加载 `best.pt`，从测试集中选择“正确分类且目标 detector 能量最高”的样例，保存到：

```text
runs/<run_name>/slm_bmp_best/
```

保存内容包括：

```text
input_amplitude_active450.bmp
prompt_amplitude_active450.bmp
prompt_phase_active450.bmp
expert_layer_01_mosaic_active450.bmp ... expert_layer_05_mosaic_active450.bmp
global_fc_phase_active450.bmp
raw_optical_planes.pt
manifest.json
```

每个物理平面只生成一张 BMP，不再为九个专家分别生成文件。仿真像素 16 μm、SLM 像素 8 μm，严格使用 nearest-neighbor 2× 复制，不使用 bilinear/bicubic，也不会产生原图中不存在的中间像素值：

```text
450×450 → 900×900
```

900×900 居中放入宽 1920、高 1200 的灰度 BMP：

```text
left/right padding = 510
top/bottom padding = 150
```

相位编码为 `uint8(round((phase mod 2π) × 255 / 2π))`，振幅编码为 `uint8(round(amplitude × 255))`，外部 padding 均为 0。

## 输出

每次运行保存配置、环境、数据集统计、架构和参数统计、epoch CSV、专家选择率、best/last checkpoint、训练曲线、混淆矩阵、prompt 权重、五层专家 phase mosaic、global FC、逐层光场、detector 区域和 SLM BMP 包。
