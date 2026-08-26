# Commands

所有命令从仓库根目录执行。

## 1. Smoke test

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um --config experiments/d2nn_mnist4_single_layer_17um/configs/smoke/mnist4_single_layer_17um_5cm_smoke.yaml --phase all
```

## 2. 正式训练、官方测试集评估并导出BMP

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um --config experiments/d2nn_mnist4_single_layer_17um/configs/release/mnist4_single_layer_17um_5cm.yaml --phase all
```

正式配置使用MNIST `0/1/2/3` 的全部官方训练样本，按类别固定划分90%训练和10%验证；
官方测试集只做最终报告。最佳模型按验证准确率选择。

## 3. 从已有checkpoint重新测试

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um --config experiments/d2nn_mnist4_single_layer_17um/configs/release/mnist4_single_layer_17um_5cm.yaml --phase test --checkpoint experiments/d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/checkpoints/best.pt
```

## 4. 只重新导出硬件BMP

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um --config experiments/d2nn_mnist4_single_layer_17um/configs/release/mnist4_single_layer_17um_5cm.yaml --phase export --checkpoint experiments/d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/checkpoints/best.pt
```

## 5. 评估实验CCD图像

先用四菲涅尔焦点确定原始相机坐标中的ROI和翻转关系。采集文件应使用
`samples.csv` 中的 `key` 作为文件主名，然后执行：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um.ccd_evaluate --config experiments/d2nn_mnist4_single_layer_17um/configs/release/mnist4_single_layer_17um_5cm.yaml --manifest experiments/d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/hardware_export/samples.csv --ccd-dir ccd_captured --output-dir ccd_evaluation --roi LEFT,TOP,RIGHT,BOTTOM --flip-vertical --flip-horizontal
```

`--flip-vertical`、`--flip-horizontal` 只填写实际标定得到的方向；不需要的参数应删除。
若实验室端已经提前裁剪到准确ROI，可以不传 `--roi`。程序会面积重采样到478×478，
不会执行背景扣除。

