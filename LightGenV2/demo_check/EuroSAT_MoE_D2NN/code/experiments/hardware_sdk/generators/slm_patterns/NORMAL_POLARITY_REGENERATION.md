# 17 µm 振幅 SLM 正常极性重新生成说明

## 唯一极性合同

```text
振幅BMP 255 = 白色 = 透光/亮
振幅BMP   0 = 黑色 = 遮光/暗
振幅导出不做黑白反相
相位BMP不因振幅极性修正而改变，仍保留既有纵向翻转
```

新产物都带 `normal_polarity` 或 `normal`，不要与历史 `_inv` / `inverted` 目录混用。
以下命令均从仓库根目录执行。

## 1. 基础双SLM标定图

```powershell
python -m experiments.hardware_sdk.generators.dual_slm_alignment --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml
```

输出：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment_normal_polarity/
```

包含黑/白场、十字、边框、518/478等孔径、MoE4边框/单专家、规则棋盘、非规则大块及其
配对相位光栅。清单 `alignment_manifest.json` 记录每一对文件、中心、翻转和SHA256。

## 2. 棋盘/大块与相位倍率扫描

```powershell
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um_normal_scale_sweep.yaml
```

输出：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_normal_large_blocks_k0p1/
```

倍率从 `k=1` 开始，先以0.0005细扫到±0.005，再以0.01扫到±0.1。大块X/Y光栅分开，
单张相位图只有一个方向。实验配对以 `scale_sweep_manifest.csv` 为准。

## 3. 1/4/9菲涅尔阵列

```powershell
python -m experiments.hardware_sdk.generators.fresnel_phase_array --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_phase_array_17um_8um.yaml
```

输出：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_phase_array_532nm_17um_8um_normal_polarity/
```

固定播放 `amplitude_bmp/amplitude_uniform_white_1024x1024.bmp`（全255均匀照明），再播放
5/10/15 cm的1、4、9阵列相位。相位有效区仍为1016×1016、中心 `(980,590)`、边界
`[472,82,1488,1098]`。先用 `flip_coded` 建立对应，再用 `uniform` 精确拟合ROI。

## 4. 离线手写数字与0～255曝光标定

```powershell
python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml --generate-only

python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml
```

两组输出位于四层Caltech101工程的
`hardware_sessions/_calibration_17um/`。数字前景为255、背景为0；曝光标定按命令灰度
0→255递增，不做反相。它们是硬件/曝光检查，不是MNIST4分类模型输入。

## 5. MNIST4已训练模型重新导出

极性修正只影响振幅BMP编码，不改变训练时的光场振幅，也不需要重新训练。服务器上从
已有 `best.pt` 导出：

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.d2nn_mnist4_single_layer_17um --config experiments/d2nn_mnist4_single_layer_17um/configs/release/mnist4_single_layer_17um_5cm.yaml --phase export --checkpoint experiments/d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/checkpoints/best.pt
```

输出：

```text
experiments/d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/hardware_export_normal_polarity/
```

其中40张输入振幅图为正常极性；训练得到的相位BMP应与旧导出逐字节相同。

## 6. 当前服务器打包产物

以下ZIP已经生成在各自数据目录，不提交Git：

```text
generators/slm_patterns/generated/dual_slm_17um_8um_alignment_normal_polarity.zip
SHA256 1452448a2119d848ba5ffbeed863ce1ca0a2f42111d9f304e0035ff823c6940c

generators/slm_patterns/generated/dual_slm_17um_8um_normal_large_blocks_k0p1.zip
SHA256 3f448f99e1da2cdcd8c424ee8fb65018c6210da341b63deab2f8254fbc7d2a9a

generators/slm_patterns/generated/fresnel_phase_array_532nm_17um_8um_normal_polarity.zip
SHA256 9214b8ad7da582554b66ca11fb816bed6a8a9f417f2db9d770a37e1a903cdf81

qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/calibration_17um_normal_polarity.zip
SHA256 981d2bd633f8f2c57d8b57a0d8d7399fe7d97ef30b0843d000f14359ff1b6431

d2nn_mnist4_single_layer_17um/runs/mnist4_single_layer_17um_5cm/mnist4_single_layer_17um_5cm_hardware_bundle_normal_polarity.zip
SHA256 7067f1bd9cf0d4fe3cee0bc3f857007a0b24fd04fa65396cd8cc9923ef63673c
```
