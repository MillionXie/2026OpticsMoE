# KADID-10k 纯光 6 层图像质量回归

这个工程是一个不接 Qwen、不接 CNN/MLP、也不把质量离散成三类的纯光回归 baseline。输入只使用失真图像本身，参考图像仅用于 reference-disjoint 数据划分，不会进入模型。

## 任务与光路

```text
KADID-10k distorted RGB image
-> grayscale
-> bicubic resize to 300 x 300
-> centered zero padding to 360 x 360
-> phase-only plane 1
-> 5 cm propagation
-> phase-only plane 2
-> ...
-> phase-only plane 6
-> 10 cm propagation
-> 10 fixed detector regions representing quality anchors 0.00 ... 1.00
-> normalized detector-region energies
-> fixed weighted expectation q_hat in [0, 1]
```

这里 `0` 表示最低质量，`1` 表示最高质量。最后的连续预测为：

```text
p_k = detector_energy_k / sum_j detector_energy_j
q_hat = sum_k p_k * quality_anchor_k
```

因此 detector 之后没有可训练电子层。整个模型唯一可训练的参数是六张相位板：

```text
6 x 360 x 360 = 777,600 optical phase parameters
electronic trainable parameters = 0
```

默认相位采用 `2*pi*sigmoid(raw_phase)` 约束，`raw_phase` 从 0 初始化；波长 532 nm，像素尺寸 16 um，输入直接加载到第一张相位板，层间距离 5 cm，到 detector 的距离为 10 cm。

## 数据与分数

数据加载器自动识别常见字段，包括当前数据中的 `dist_img`、`ref_img`、`dmos` 和 `var`。找不到数据时，`download: true` 会复用现有 KADID-10k 下载/定位程序。

划分按 reference image 完成，train、validation、test 不共享参考图像，避免同一内容的不同失真版本泄漏到多个 split。归一化上下界只用 train split 估计。

默认配置明确设置：

```yaml
quality_score_higher_is_better: true
```

即原始分数越大表示质量越高。如果换用“数值越大质量越差”的 DMOS 来源，必须改为 `false`。无论原始方向如何，模型内部统一为 `0=worst, 1=best`。

## Loss

```text
L = 1.0 * MSE(q_hat, q)
  + 0.2 * MSE(detector_distribution, soft_quality_target)
  + 0.05 * detector_concentration_loss
```

soft target 是以真实连续质量分数为中心、覆盖相邻质量锚点的高斯分布。它用于约束光能落到具有连续质量含义的 detector 区域，而不是训练电子分类器。

## 结果与可视化

每次运行保存在本工程的 `runs/<run_name>/`，包括：

- `dataset.json`：reference-disjoint 划分及分数范围；
- `model.json`：光路、detector 边界和光/电参数量；
- `metrics/training_history.csv`：每轮 loss、MAE、RMSE、PLCC、SROCC；
- `metrics/test_regression.json`：归一化质量和原始分数两套测试指标；
- `metrics/per_distortion_type.json`、`per_distortion_level.json`；
- `metrics/test_predictions.csv`；
- `figures/training_curves.png`、`predicted_vs_target.png`；
- 每层 phase mask、光场强度以及 detector 质量锚点柱状图。

这是 no-reference IQA baseline。灰度输入会丢失色彩失真信息，因此它适合验证纯光结构的基本回归能力，但并不等价于完整的彩色 IQA 系统。
