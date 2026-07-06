# 三类别 detector 区域与 loss

最终 256×256 探测面设置三个固定、互不重叠的 48×48 方格：

```text
左：daytime    中：night    右：dawn_dusk
```

固定区域不包含可训练参数。准确坐标保存在 `model.json`，布局图保存在 `figures/detector_regions/layout.png`。

设最终强度为 `I`，类别区域 mask 为 `M_k`：

```text
E_k = sum(I * M_k)
E_total = sum(I)
region_fraction_k = E_k / E_total
detector_fraction = sum_k(E_k) / E_total
region_distribution_k = E_k / sum_j(E_j)
```

训练 loss：

```text
L_classification = CE(electronic_logits, label)
L_region = CE(log(region_distribution + eps) / temperature, label)
L_concentration = -log(detector_fraction + eps)

L_total = L_classification
        + detector_region_loss_weight * L_region
        + detector_concentration_loss_weight * L_concentration
```

默认权重：

```json
{
  "detector_region_loss_weight": 1.0,
  "detector_concentration_loss_weight": 0.1,
  "detector_region_temperature": 1.0
}
```

其中 `L_concentration` 约束的是三个方格占总光场的能量比例，不是绝对光强。否则模型可以通过整体放大强度降低 loss，而没有真正形成空间聚焦。

电子 readout 仍处理完整末端强度图，并将三个 `region_distribution` 作为额外语义特征拼接到池化特征后。最终论文结果应同时报告电子分类准确率、detector region 独立准确率和 detector energy fraction。
