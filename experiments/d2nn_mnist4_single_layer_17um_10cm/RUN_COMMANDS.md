# Commands

所有命令从仓库根目录执行。这里只记录命令，不是 `.sh` 文件。

正式配置：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/configs/release/mnist4_single_layer_17um_10cm_notebook_mse.yaml
```

## 1. 单元测试与 smoke

```bash
python -m pytest experiments/d2nn_mnist4_single_layer_17um_10cm/tests -q
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um_10cm --config experiments/d2nn_mnist4_single_layer_17um_10cm/configs/smoke/mnist4_single_layer_17um_10cm_notebook_mse_smoke.yaml --phase all
```

## 2. 正式训练、官方测试集评估与 BMP 导出

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um_10cm --config experiments/d2nn_mnist4_single_layer_17um_10cm/configs/release/mnist4_single_layer_17um_10cm_notebook_mse.yaml --phase all
```

正式输出：

```text
experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/
```

## 3. 从 checkpoint 单独评估

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um_10cm --config experiments/d2nn_mnist4_single_layer_17um_10cm/configs/release/mnist4_single_layer_17um_10cm_notebook_mse.yaml --phase test --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/checkpoints/best.pt
```

## 4. 从 checkpoint 重新导出硬件文件

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um_10cm --config experiments/d2nn_mnist4_single_layer_17um_10cm/configs/release/mnist4_single_layer_17um_10cm_notebook_mse.yaml --phase export --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/checkpoints/best.pt
```

导出中包含正常极性的振幅 BMP、10 cm 相位 BMP、探测区定义、demo 与正式随机测试 manifest。

## 5. 打成一个实验室 ZIP

```bash
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_package --export-dir experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/hardware_export_10cm_normal_polarity --output experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/mnist4_single_layer_17um_10cm_lab_bundle.zip
```

正式交付不要添加 `--omit-vendor-sdk`。ZIP 内包含振幅/相位 BMP、manifest、轻量 Python 运行代码、Meadowlark 与 TUCam SDK 目录及环境文件。

## 6. 实验室电脑安装

解压 ZIP，进入解压目录：

```powershell
python -m pip install -r experiments\d2nn_mnist4_single_layer_17um_10cm\requirements-lab.txt
```

先填写：

```text
experiments\d2nn_mnist4_single_layer_17um_10cm\lab_hardware_config.yaml
```

其中 `camera.device_roi_xywh` 必须来自四焦点标定；LUT、曝光与输入强度范围也必须按实际设备确认。

## 7. 正式 400 张硬件测试

校验设备、文件数量、phase SHA256 和 ROI：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase validate --stage-dir payload\formal_fixed_random_100_per_class
```

手动把程序提示的唯一 10 cm phase BMP 加载到相位 SLM，然后自动顺序播放振幅、采集同名 CCD：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase acquire --stage-dir payload\formal_fixed_random_100_per_class
```

计算准确率、混淆矩阵和逐样本结果：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase evaluate --stage-dir payload\formal_fixed_random_100_per_class
```

首次不要加 `--clear-output`。只有明确要覆盖已采集 CCD 时才使用该参数。

## 8. 快速演示组

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase all --stage-dir payload\demo_topk --allow-biased-demo-metric
```

这组样本经过仿真正确性与边界筛选，输出字段是
`demo_success_rate`，不是 `accuracy`；它只能用于光路演示与快速排障。
不加 `--allow-biased-demo-metric` 时，`evaluate` 和 `all` 会主动拒绝 demo。
