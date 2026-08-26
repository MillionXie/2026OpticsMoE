# Commands

所有命令都从仓库根目录执行。本文件只记录命令，不是 `.sh` 文件。

正式配置：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_robust_raw.yaml
```

## 1. 测试

```bash
python -m pytest experiments/d2nn_mnist4_single_layer_17um_10cm_v2/tests -q
```

## 2. Smoke（需要 GPU，但本次未启动）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/smoke/mnist4_single_layer_17um_10cm_v2_smoke.yaml --phase all
```

## 3. 正式 60 轮训练、official test 与导出

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -u -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_robust_raw.yaml --phase all 2>&1 | tee experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_robust_raw/formal_train.log
```

输出目录：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_robust_raw/
```

## 4. 单独测试最佳 checkpoint

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_robust_raw.yaml --phase test --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_robust_raw/checkpoints/best.pt
```

## 5. 单独导出硬件文件

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_robust_raw.yaml --phase export --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_robust_raw/checkpoints/best.pt
```

相位文件名为 `mnist4_single_layer_17um_10cm_v2.bmp`，不会与旧工程混淆。

## 6. 实验室 formal400 自动顺序采集

先加载 stage 中唯一的 v2 phase BMP，再执行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\d2nn_mnist4_single_layer_17um_10cm\lab_hardware_config.yaml --stage-dir experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_robust_raw\hardware_export_10cm_v2_proportional_roi\formal_fixed_random_100_per_class
```

相机必须直接保存 478×478、8-bit 灰度图。v2 评估器拒绝 resize，也不做强度
归一化、非线性压缩或背景扣除：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.ccd_evaluate --config experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_robust_raw\hardware_export_10cm_v2_proportional_roi\lab_model_config.yaml --manifest experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_robust_raw\hardware_export_10cm_v2_proportional_roi\formal_fixed_random_100_per_class\samples.csv --ccd-dir experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_robust_raw\hardware_export_10cm_v2_proportional_roi\formal_fixed_random_100_per_class\ccd_captured --output-dir experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_robust_raw\hardware_export_10cm_v2_proportional_roi\formal_fixed_random_100_per_class\hardware_evaluation_raw
```

只有实测 Fresnel 对应关系证明需要时，才给评估命令增加 `--flip-vertical` 或
`--flip-horizontal`；不要用图像效果主观选择翻转。
