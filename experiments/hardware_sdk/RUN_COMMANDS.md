# Hardware SDK Commands

以下命令均在实验室电脑的 `experiments\hardware_sdk` 目录运行。

## 1. 安装与确认目录

```powershell
cd D:\code\guest\2026OpticsMoE\experiments\hardware_sdk
python -m pip install -r requirements-light.txt
```

新 TUCam SDK 必须保留以下结构：

```text
ccd_2_mosaic\TUCam.py
ccd_2_mosaic\lib\x64\TUCam.dll
```

## 2. 新 TUCam 相机单独自检

不打开 SLM，只测试相机初始化、曝光、取帧、位深和文件保存：

```powershell
python tools\camera_smoke_test.py `
  --config configs\acquisition\tucam_windows.json `
  --output-dir demo_outputs\tucam_smoke `
  --frames 3
```

检查：

```text
demo_outputs\tucam_smoke\frame_*.npy
demo_outputs\tucam_smoke\frame_*_preview.png
demo_outputs\tucam_smoke\camera_smoke_test.json
```

NPY 是定量原始帧；PNG 只用于肉眼检查。

## 3. 新相机正式播放与采集

把振幅 BMP 放入：

```text
amplitude_to_play\
```

确认相位 mask 后运行：

```powershell
python acquire_folder.py `
  --config configs\acquisition\tucam_windows.json `
  --clear-output
```

输出：

```text
ccd_captured\*.npy
logs\tucam\capture_manifest.csv
logs\tucam\resolved_devices.json
```

## 4. 新相机 ROI/背景/曝光标定

```powershell
python roi_calibration.py generate `
  --config configs\calibration\tucam.yaml

python roi_calibration.py background `
  --config configs\calibration\tucam.yaml

python roi_calibration.py coarse `
  --config configs\calibration\tucam.yaml
```

根据 `calibration_results\tucam\coarse_calibration.json` 和 overlay，在 Mosaic/厂商软件中设置硬件 ROI，然后运行：

```powershell
python roi_calibration.py fine `
  --config configs\calibration\tucam.yaml

python roi_calibration.py exposure `
  --config configs\calibration\tucam.yaml
```

## 5. 正式采集后统一处理

```powershell
python batch_postprocess.py `
  --config configs\calibration\tucam.yaml `
  --input-dir ccd_captured `
  --calibration calibration_results\tucam\calibration.json `
  --background calibration_results\tucam\background.npy `
  --output-dir processed_ccd\tucam
```

## 6. 继续使用旧 DVP 相机

旧相机代码未删除。正式采集：

```powershell
python acquire_folder.py `
  --config configs\acquisition\dvp_windows.json `
  --clear-output
```

旧标定配置仍为：

```powershell
python roi_calibration.py generate --config configs\calibration_config.yaml
python roi_calibration.py background --config configs\calibration_config.yaml
python roi_calibration.py coarse --config configs\calibration_config.yaml
python roi_calibration.py fine --config configs\calibration_config.yaml
python roi_calibration.py exposure --config configs\calibration_config.yaml
```

原来的 `configs\acquisition_windows.json` 也继续保留，作为旧 DVP 命令的兼容入口。

## 7. SLM 独立 demo

振幅 SLM + 相机数字顺序测试：

```powershell
python amplitude_camera_demo.py `
  --config configs\amplitude_camera_digit_demo.yaml
```

相位 SLM 图案检查：

```powershell
python phase_slm_demo.py `
  --config configs\phase_slm_demo.yaml `
  --dry-run
```

常用棋盘格、透镜和对齐图案仍由：

```powershell
python -m slm_calibration_bmp_generator `
  --config slm_calibration_bmp_generator\configs\slm_956.yaml
```

生成。
