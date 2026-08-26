# Formal simulation result

运行日期：2026-08-26。正式运行使用配置
`configs/release/mnist4_single_layer_17um_5cm.yaml`，GPU为RTX 4090（物理GPU 4）。

## 数据和模型

```text
MNIST classes:              0, 1, 2, 3
training samples:           22,279
validation samples:          2,475
official test samples:       4,157
trainable optical params:   228,484
trainable electronic params:      0
best validation epoch:            55
best validation accuracy:     76.77%
official test accuracy:       76.86%
```

测试集没有参与checkpoint选择。测试混淆矩阵（行为真实类别，列为预测类别）：

```text
          pred0  pred1  pred2  pred3
true0       584     13    175    208
true1        68    925     84     58
true2       105     11    836     80
true3        77     28     55    850
```

各类准确率分别为：数字0 `59.59%`、数字1 `81.50%`、数字2 `81.01%`、数字3
`84.16%`。整体显著高于四分类随机水平25%，但数字0是当前单层固定探测布局的主要
薄弱类别；这项结果应作为简单光学demo，不应等同于多层D2NN上限。

最佳相位统计：

```text
raw mean/std:          0.01683 / 1.26634
phase mean/std:        3.14612 / 1.26829 rad
phase min/max:         0.00741 / 6.27753 rad
```

相位已经覆盖接近完整的0–2π范围，不是接近均匀相位的失效mask。

## 服务器产物

根目录：

```text
/DATA/DATA1/guest3/2026OpticsMoE/experiments/d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/
```

重要文件：

```text
checkpoints/best.pt
metrics/training_log.csv
metrics/training_summary.json
metrics/test_metrics.json
phase_initial.png
phase_best.png
hardware_export/phase_to_play/mnist4_single_layer_17um_5cm.bmp
hardware_export/amplitude_to_play/*.bmp
hardware_export/samples.csv
mnist4_single_layer_17um_5cm_hardware_bundle.zip
```

SHA-256：

```text
best.pt:
9fb07d9d911692396702f9ee81b35a45d0332deb9a14cf91a4c23ae1d7cb723b

phase BMP:
6fcd7c894ba914a679174d8508f20b8bc13e1686d8878dbf03d2a51284c5f343

hardware ZIP:
850f162e3e15363f9379e33babeb7c8a2cd3fe9277ab4042445a68ca68178b67
```

导出检查：相位BMP为1920×1200、8-bit灰度；40张振幅BMP均为1024×1024、
8-bit灰度。相位有效区为1016×1016，边界 `[472,82,1488,1098]`，中心
`(980,590)`，并执行既有纵向翻转。振幅有效区边界为 `[273,273,751,751]`，按当前
光路极性执行黑白反相。
