# 三类别 detector 区域设计

## 目标

旧版本让完整末端强度图直接进入电子卷积 readout，光场本身没有人为定义的类别语义。新版在最终 256×256 探测面上增加三个固定、互不重叠的类别方格：

```text
左侧方格   -> daytime
中间方格   -> night
右侧方格   -> dawn_dusk
```

模型不仅要让电子 MLP 正确分类，还要把能量分配到真实类别对应的物理 detector 区域。

## 固定布局

主配置使用 48×48 方格，三个中心横坐标约为 64、128、192，纵坐标均为 128：

```text
256 × 256 detector plane

       daytime             night             dawn_dusk
     +----------+       +----------+       +----------+
     |  48×48   |       |  48×48   |       |  48×48   |
     +----------+       +----------+       +----------+
```

准确边界保存到 `model.json -> class_region_detector -> boxes`，布局图保存到：

```text
figures/detector_regions/layout.png
```

区域 mask 是 fixed buffer，不可训练，不增加模型参数。

## 能量读出

设最终非负强度为 `I[b,y,x]`，第 k 类的固定二值区域 mask 为 `M[k,y,x]`：

```text
E_k = sum(I * M_k)
E_total = sum(I)
region_fraction_k = E_k / E_total
detector_fraction = sum_k(E_k) / E_total
region_distribution_k = E_k / sum_j(E_j)
```

其中：

- `region_fraction_k`：全光场能量中落入第 k 个格子的比例。
- `detector_fraction`：全光场能量中落入三个 detector 格子的总比例。
- `region_distribution_k`：只看三个格子时，第 k 个格子的相对能量比例。

## 三项训练目标

### 1. 电子分类损失

```text
L_classification = CE(electronic_logits, label)
```

电子 readout 继续处理完整末端强度图。

### 2. 类别区域损失

```text
region_logits = log(region_distribution + eps) / temperature
L_region = CE(region_logits, label)
```

它要求目标类别格子的相对能量最大。默认温度为 1.0，权重为 1.0。

### 3. detector 能量集中损失

仅比较三个格子的相对能量不能防止光全部落在格子外，因此增加：

```text
L_concentration = -log(detector_fraction + eps)
```

该项鼓励光进入三个有意义的 detector 区域。默认权重为 0.1，避免它压过分类目标。

总损失：

```text
L_total = L_classification + 1.0 * L_region + 0.1 * L_concentration
```

## 电子 readout 如何使用区域信息

完整末端强度图仍然执行：

```text
log1p
 -> Conv(1 -> 16)
 -> GroupNorm
 -> GELU
 -> Conv(16 -> 32, stride=2)
 -> GroupNorm
 -> GELU
 -> AdaptiveAvgPool(8,8)
 -> flatten
```

随后拼接三个 `region_distribution` 标量：

```text
[32 * 8 * 8] + [3 region features]
 -> LayerNorm
 -> Linear(2051 -> 256)
 -> GELU
 -> Dropout(0.2)
 -> Linear(256 -> 3)
```

这样既保留完整光场中的细节，也让电子分类器明确读取三个物理区域。电子网络没有新增卷积层，只增加三个标量输入。

## 新增配置

```json
{
  "detector_region_size": 48,
  "detector_region_temperature": 1.0,
  "detector_region_loss_weight": 1.0,
  "detector_concentration_loss_weight": 0.1
}
```

改变 `detector_region_size` 会改变物理读出面积，属于实验变量。三个区域必须保持不重叠。

## 新增训练与测试记录

训练 history 增加：

- classification、detector region、detector concentration 三项 loss；
- detector region 单独分类准确率；
- 三个 detector 区域的总能量比例；
- 真实类别区域的平均能量比例。

测试预测 CSV 增加：

- detector 独立预测类别；
- 三个类别区域各自的全场能量比例；
- 三个 detector 区域合计占全场的能量比例。

`figures/detector_outputs/epoch_XXXX.png` 同时显示带方框的末端光场和三个区域内部的能量柱状图。

## 解释边界

区域监督让末端光场具有可解释的类别读出，但最终报告的主分类结果仍是电子 readout logits。应同时报告：

1. 最终电子 top-1 / macro-F1；
2. detector region 独立准确率；
3. 平均 detector energy fraction；
4. 每类目标区域的能量比例。

如果电子准确率高而 detector region 准确率低，说明电子 readout 仍主要依赖格子外或区域内部的复杂纹理；如果两者都高，才能说明类别区域语义确实建立起来。
