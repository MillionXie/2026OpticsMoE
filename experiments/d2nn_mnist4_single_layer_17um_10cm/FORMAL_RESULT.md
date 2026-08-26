# MNIST-4 10 cm 正式仿真结果与交付记录

本文只记录本次服务器正式训练、official test 仿真评估结果及实验室交付文件。
服务器仓库根目录为：

```text
/DATA/DATA1/guest3/2026OpticsMoE
```

## 1. 数据划分

| 划分 | 样本数 |
|---|---:|
| Train | 22,279 |
| Validation | 2,475 |
| Official test | 4,157 |

## 2. 最佳验证结果

| 指标 | 结果 |
|---|---:|
| 最佳 epoch（从 1 开始计数） | 29 |
| Validation accuracy | 0.7361616162 |

## 3. Official test 仿真结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.7541496271 |
| Loss（template MSE） | 1.5981779455 |
| Target fraction | 0.06823754 |
| Capture fraction | 0.14086348 |

混淆矩阵的行是真实类别、列是预测类别，类别顺序均为 `0, 1, 2, 3`：

```text
[
  [867,  13,  54,  46],
  [ 16, 888, 123, 108],
  [ 50,  90, 746, 146],
  [ 76, 277,  23, 634]
]
```

该混淆矩阵共包含 4,157 个 official test 样本，其中对角线正确样本数为
3,135，对应 accuracy 为 0.7541496271。

## 4. 相位与可训练参数

| 项目 | 结果 |
|---|---:|
| Phase mean | 3.225398 rad |
| Phase std | 1.496451 rad |
| Phase min | 0.000559 rad |
| Phase max | 6.282399 rad |
| 可训练光学参数 | 228,484 |
| 可训练电子参数 | 0 |

## 5. 交付文件与完整性

以下路径均相对于服务器仓库根目录。

运行目录：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/
```

正式仿真指标：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/metrics/test_metrics.json
```

最佳 checkpoint：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/checkpoints/best.pt
SHA-256: a60456390ee339d6d7f9a7b241bc9df5874b576f90c4767813c99da40d588b884
```

硬件导出根目录：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/hardware_export_10cm_normal_polarity/
```

正式相位 BMP：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/hardware_export_10cm_normal_polarity/phase_to_play/mnist4_single_layer_17um_10cm.bmp
SHA-256: 6670fd88de9dd0bcc6ce05d049f923487431584159c22f6cddf22c59ae58914a
```

实验室交付 ZIP：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/mnist4_single_layer_17um_10cm_lab_bundle.zip
大小: 52,121,990 bytes
SHA-256: 09e26cf5f9fca9abb72d0fe1687cc5b84c561c40bc86da63671eb78145ea9625
```

正式硬件 400 张测试 stage 位于交付包中的：

```text
payload/formal_fixed_random_100_per_class/
```

有偏演示 40 张 stage 位于：

```text
payload/demo_topk/
```

## 6. 结果口径

- `0.7541496271` 是 MNIST official test 数据上的正式仿真精度，不是实际光路精度。
- 实际硬件精度必须完成 `formal_fixed_random_100_per_class` 的 400 张采集与评估后才能报告。
- `demo_topk` 的 40 张样本经过仿真正确性与边界筛选，存在选择偏差，只能用于对齐和演示，不得作为准确率报告。
