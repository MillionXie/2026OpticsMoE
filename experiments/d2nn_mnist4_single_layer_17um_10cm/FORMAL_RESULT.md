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
| 最佳 epoch（从 1 开始计数） | 44 |
| Validation accuracy | 0.7434343434 |
| Validation loss（template MSE） | 1.5794771768 |

正式配置仅将原30轮训练延长到60轮，结构、数据、ROI、loss、batch和学习率均未
同时改变。旧30轮最佳验证准确率为0.7361616162；本次提高0.7273个百分点。

## 3. Official test 仿真结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.7604041376（3161/4157） |
| Loss（template MSE） | 1.5779869163 |
| Detector CE（仅诊断，loss权重为0） | 0.7932167653 |
| Target fraction | 0.0678723387 |
| Capture fraction | 0.1401467001 |

混淆矩阵的行是真实类别、列是预测类别，类别顺序均为 `0, 1, 2, 3`：

```text
[
  [874,  13,  44,  49],
  [ 11, 886, 124, 114],
  [ 48,  85, 745, 154],
  [ 72, 257,  25, 656]
]
```

各类准确率依次约为89.18%、78.06%、72.19%和64.95%。该混淆矩阵共包含
4,157个official test样本，其中对角线正确样本数为3,161。相对旧30轮的
0.7541496271，本次提高0.6255个百分点，多正确26张。

## 4. 相位与可训练参数

| 项目 | 结果 |
|---|---:|
| Phase mean | 3.214946 rad |
| Phase std | 1.556396 rad |
| Phase min | 0.000049 rad |
| Phase max | 6.283134 rad |
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
SHA-256: bf23d56c6f8f14af3b57a914864af16107260b73c1807d43db03807103ed68ad
```

硬件导出根目录：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/hardware_export_10cm_normal_polarity/
```

正式相位 BMP：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/hardware_export_10cm_normal_polarity/phase_to_play/mnist4_single_layer_17um_10cm.bmp
SHA-256: f7e50a067240e97ef33b1f0f9bbb5042eb204f87e2f917ae18d0166b929047bb
```

实验室交付 ZIP：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/mnist4_single_layer_17um_10cm_lab_bundle.zip
大小: 52,090,417 bytes
成员数: 582（`zip.testzip()`通过）
SHA-256: 31bc33cea7e51c2966d14a6ebe9d6d0cab664ace83e8361d300422ba91faf59f
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

- `0.7604041376` 是 MNIST official test 数据上的正式仿真精度，不是实际光路精度。
- 实际硬件精度必须完成 `formal_fixed_random_100_per_class` 的 400 张采集与评估后才能报告。
- `demo_topk` 的 40 张样本经过仿真正确性与边界筛选，存在选择偏差，只能用于对齐和演示，不得作为准确率报告。
