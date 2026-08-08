# Hardware SDK — Complete Commands

以下命令均从仓库根目录执行：

```powershell
cd D:\code\guest\2026OpticsMoE
python -m pip install -r experiments\hardware_sdk\requirements-light.txt
```

## 0. 目录约定

```text
vendor_sdk/                         厂商 SDK，不放实验数据
data/amplitude_to_play/             当前要播放的 1920×1080 振幅 BMP
data/ccd_captured/                  相机原始 NPY/TIFF
data/processed/                     统一后处理结果
artifacts/calibration/masks/        ROI/背景/曝光标定 BMP
artifacts/calibration/results/      background、变换矩阵、响应曲线
artifacts/demos/                    独立设备 demo 输出
artifacts/logs/                     采集 manifest
```

每轮正式采集前只清理 `data/amplitude_to_play/` 和 `data/ccd_captured/`；不要
删除 `vendor_sdk/`、标定结果或配置。

## 1. 新 TUCam/Mosaic CCD 自检

确认以下文件存在：

```text
experiments\hardware_sdk\vendor_sdk\camera_tucam_mosaic\TUCam.py
experiments\hardware_sdk\vendor_sdk\camera_tucam_mosaic\lib\x64\TUCam.dll
```

只打开相机，不打开 SLM：

```powershell
python -m experiments.hardware_sdk.tools.camera_smoke_test `
  --config experiments\hardware_sdk\configs\acquisition\tucam_windows.json `
  --output-dir experiments\hardware_sdk\artifacts\demos\tucam_smoke `
  --frames 3
```

定量原始帧为 `frame_*.npy`，`frame_*_preview.png` 只用于肉眼检查。

## 2. 生成 ROI/曝光标定 BMP

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration generate `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

主要输出：

```text
artifacts\calibration\masks\amplitude\amplitude_zero.bmp
artifacts\calibration\masks\phase\phase_zero.bmp
artifacts\calibration\masks\amplitude\verify_roi_5points.bmp
artifacts\calibration\masks\amplitude\verify_roi_5rectangles.bmp
artifacts\calibration\masks\amplitude\verify_roi_outline.bmp
artifacts\calibration\masks\amplitude\coarse_*.bmp
artifacts\calibration\masks\amplitude\fine_*.bmp
artifacts\calibration\masks\exposure\gray_*.bmp
artifacts\calibration\masks\manifest.csv
```

其中五点图的四角是实际 956×956 ROI 角点；五矩形图使用相同中心位置，
轮廓图直接显示 ROI 四边。

## 3. 完整标定流程

### 3.1 统一系统背景

保持激光开启；相位和振幅 SLM 都加载 zero BMP：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration background `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

### 3.2 粗 ROI

相机先保持完整视野：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration coarse `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

查看：

```text
artifacts\calibration\results\tucam\coarse_calibration.json
artifacts\calibration\results\tucam\coarse_overlay.png
```

把建议的 `left/top/width/height` 填入厂商软件。TUCam 四项都必须是 4
像素的倍数。程序结束时会保持五点验证图，亦可人工改载五矩形或 ROI 轮廓图。

### 3.3 精标定

在厂商软件设置并保持硬件 ROI 后：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration fine `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

输出 `calibration.json`、`fine_overlay.png` 和 `residual_vectors.png`。

### 3.4 灰度/曝光检查

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

查看 `slm_response.csv`、原始/归一化响应曲线和曝光 preview，人工决定正式
曝光；程序不会自动改正式配置。

## 4. 正式振幅播放与原始 CCD 采集

1. 把本轮全部 1920×1080、8-bit 灰度 BMP 放入：

```text
data\amplitude_to_play\
```

2. 手动加载当前相位 mask。

3. 运行新 CCD：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\acquisition\tucam_windows.json `
  --clear-output
```

输出：

```text
data\ccd_captured\*.npy
artifacts\logs\tucam\capture_manifest.csv
artifacts\logs\tucam\resolved_devices.json
```

采集阶段不 resize、不扣背景、不归一化。

## 5. 正式帧统一后处理

```powershell
python -m experiments.hardware_sdk.workflows.batch_postprocess `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml `
  --input-dir experiments\hardware_sdk\data\ccd_captured `
  --calibration experiments\hardware_sdk\artifacts\calibration\results\tucam\calibration.json `
  --background experiments\hardware_sdk\artifacts\calibration\results\tucam\background.npy `
  --output-dir experiments\hardware_sdk\data\processed
```

流程为：原始帧减统一 background → affine/homography → 原采样密度规则正方形
→ `INTER_AREA` 缩小到 956×956。定量输出不做逐图归一化。

## 6. 继续使用旧 DVP CCD

SDK 位于：

```text
vendor_sdk\camera_dvp_legacy\
```

正式采集：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\acquisition\dvp_windows.json `
  --clear-output
```

旧 CCD 标定：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\calibration_config.yaml
python -m experiments.hardware_sdk.workflows.roi_calibration background --config experiments\hardware_sdk\configs\calibration_config.yaml
python -m experiments.hardware_sdk.workflows.roi_calibration coarse --config experiments\hardware_sdk\configs\calibration_config.yaml
python -m experiments.hardware_sdk.workflows.roi_calibration fine --config experiments\hardware_sdk\configs\calibration_config.yaml
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\calibration_config.yaml
```

`configs\acquisition_windows.json` 继续作为旧 DVP 兼容配置。

## 7. SLM 独立测试

振幅 SLM + 旧相机数字顺序 demo：

```powershell
python -m experiments.hardware_sdk.demos.amplitude_camera_demo `
  --config experiments\hardware_sdk\configs\amplitude_camera_digit_demo.yaml
```

相位 SLM dry-run：

```powershell
python -m experiments.hardware_sdk.demos.phase_slm_demo `
  --config experiments\hardware_sdk\configs\phase_slm_demo.yaml `
  --dry-run
```

生成棋盘格、5 cm/10 cm 透镜、字母等常用图案：

```powershell
python -m experiments.hardware_sdk.generators.slm_patterns `
  --config experiments\hardware_sdk\generators\slm_patterns\configs\slm_956.yaml
```

输出位于：

```text
experiments\hardware_sdk\generators\slm_patterns\generated\slm956_calibration\
```

## 8. Grocery 多层光电光交接

`hardware_sdk` 只负责播放一个文件夹并拍摄同名帧。服务器上的模型处理仍按
Grocery 工程的分层 session 进行：

1. 从服务器下载某层 `amplitude_to_play/*.bmp` 到本机
   `data/amplitude_to_play/`；
2. 加载该层相位 mask；
3. 用本文件第 4 节采集；
4. 用第 5 节统一矫正；
5. 将 `data/processed/` 同名文件上传回服务器该层 `ccd_captured/`；
6. 执行 Grocery `RUN_COMMANDS.md` 中对应层的处理/微调命令；
7. 下载下一层振幅并重复。

服务器完整分层命令见：

```text
experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/RUN_COMMANDS.md
```

## 9. 测试

```powershell
cd D:\code\guest\2026OpticsMoE
python -m pytest experiments\hardware_sdk\tests -q
python -m pytest experiments\hardware_sdk\generators\slm_patterns\tests -q
```
