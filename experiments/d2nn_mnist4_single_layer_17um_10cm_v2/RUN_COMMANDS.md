# Commands

所有命令都从仓库根目录执行；本文件只是命令记录，不是 `.sh`。

正式配置：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust.yaml
```

## 1. 单元测试

```bash
python -m pytest experiments/d2nn_mnist4_single_layer_17um_10cm_v2/tests -q
```

## 2. 正式训练、固定测试和硬件导出

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -u -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust.yaml --phase all 2>&1 | tee experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust/formal_train.log
```

输出目录：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust/
```

## 3. 只测试 best checkpoint

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust.yaml --phase test --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust/checkpoints/best.pt
```

## 4. 只导出硬件文件

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2 --config experiments/d2nn_mnist4_single_layer_17um_10cm_v2/configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust.yaml --phase export --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm_v2/runs/mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust/checkpoints/best.pt
```

相位文件为 `mnist4_single_layer_17um_10cm_v2.bmp`。振幅极性是修正后的
`255=白/透光，0=黑/遮光`。

## 5. 实验室电脑自动播放与采集

先在相位 SLM 加载导出的唯一 phase BMP，再执行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\d2nn_mnist4_single_layer_17um_10cm\lab_hardware_config.yaml --stage-dir experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust\hardware_export_10cm_v2_notebook_mse_light_robust\formal_fixed_random_100_per_class
```

## 6. 原始 CCD 四区求和评估

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.ccd_evaluate --config experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust\hardware_export_10cm_v2_notebook_mse_light_robust\lab_model_config.yaml --manifest experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust\hardware_export_10cm_v2_notebook_mse_light_robust\formal_fixed_random_100_per_class\samples.csv --ccd-dir experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust\hardware_export_10cm_v2_notebook_mse_light_robust\formal_fixed_random_100_per_class\ccd_captured --output-dir experiments\d2nn_mnist4_single_layer_17um_10cm_v2\runs\mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust\hardware_export_10cm_v2_notebook_mse_light_robust\formal_fixed_random_100_per_class\hardware_evaluation_raw
```

相机需直接输出已对准的 478×478 灰度 ROI。评估器拒绝 resize，也不做归一化、
非线性、动态拉伸或背景扣除。只有 Fresnel 标定确认确需翻转时，才给评估命令加
`--flip-vertical` 或 `--flip-horizontal`。
