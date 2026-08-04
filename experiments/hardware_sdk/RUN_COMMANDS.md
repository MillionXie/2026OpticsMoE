# Hardware SDK 命令

以下主流程在实验室 Windows 电脑的 `experiments\hardware_sdk` 目录运行。该工具不需要 Torch，也不执行任何电子后处理。

## 1. 一次性准备

```powershell
cd C:\path\to\2026OpticsMoE\experiments\hardware_sdk
python -m pip install -r requirements-light.txt
py -0p
$env:DVP_PYTHON = "C:\path\to\Python36\python.exe"
```

当前厂商的 `dvp.pyd` 实际链接 `python36.dll`，因此这里必须是 **64 位 CPython 3.6**，不能指向正在运行主程序的 Microsoft Store Python 3.12。如果 `py -0p` 没有列出 3.6，需要先安装/准备一个 64 位 Python 3.6 runtime。只有 CCD 子进程使用它，主采集程序仍使用现有 Python 3.12。

为 DVP 子进程安装唯一必需的 Python 包并验证 ABI：

```powershell
& $env:DVP_PYTHON -m ensurepip
& $env:DVP_PYTHON -m pip install "pip<22" "numpy==1.16.2"
& $env:DVP_PYTHON -c "import sys; sys.path.insert(0, r'$PWD\ccd\lib\windows\python3.6\x64'); import dvp, numpy; print(sys.executable, numpy.__version__)"
```

如果希望以后新开 PowerShell 也有效，可以设置用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable("DVP_PYTHON", "C:\path\to\Python36\python.exe", "User")
```

设置后需要重新打开 PowerShell。仅用 `$env:DVP_PYTHON=...` 时，只对当前窗口有效。

检查 `configs\acquisition_windows.json` 中：

- `amplitude_slm.expected_resolution_wh = [1920,1080]`；
- `settle_delay_ms = 40`；
- 自动曝光关闭；
- 曝光时间和模拟增益符合当前光强；
- 首次标定时 `camera.device_roi_xywh = null`。

## 2. 一般的振幅播放与 CCD 捕获

把本轮所有振幅 BMP 放到：

```text
amplitude_to_play\
```

程序按文件名的字典序播放，因此请保留服务器生成的完整文件名。不要重命名，也不要混入上一层文件。

手工把本轮相位 mask 加载到相位 SLM 后运行：

```powershell
python acquire_folder.py --config configs\acquisition_windows.json --clear-output
```

程序会再次提示确认相位 mask，输入 `y` 后开始。若在外部已经确认，可用：

```powershell
python acquire_folder.py --config configs\acquisition_windows.json --clear-output --yes
```

CCD 原始帧输出到：

```text
ccd_captured\<与输入 BMP 完全相同的 stem>.npy
```

采集清单和实际设备设置输出到：

```text
logs\capture_manifest.csv
logs\resolved_devices.json
```

每层完成后，将整个 `ccd_captured` 文件夹上传到服务器对应层；然后清空 `amplitude_to_play` 与 `ccd_captured`，下载服务器生成的下一层 BMP，再重复同一条命令。

## 3. 棋盘格定位与 CCD ROI

ROI 既可以在相机硬件中设置，也可以保留全帧并由服务器精确裁剪。首次实验推荐后者，避免 DVP 硬件 ROI 对齐约束和坐标系差异。

先生成标定图（可在服务器或任意装有 PyYAML 的电脑运行）：

```bash
python -m experiments.hardware_sdk.slm_calibration_bmp_generator \
  --config experiments/hardware_sdk/slm_calibration_bmp_generator/configs/slm_956.yaml
```

把生成的黑场和棋盘格复制并改名为：

```text
amplitude_to_play\000_black.bmp
amplitude_to_play\001_checkerboard.bmp
```

保持标定所需相位 mask，采集全传感器帧：

```powershell
python acquire_folder.py --config configs\acquisition_windows.json --clear-output
```

为当前 MoE4 光路寻找 `956×956` 物理 ROI：

```powershell
python roi_calibration.py `
  --reference ccd_captured\000_black.npy `
  --checkerboard ccd_captured\001_checkerboard.npy `
  --expected-width 956 --expected-height 956 `
  --output-dir logs\roi_calibration
```

检查：

```text
logs\roi_calibration\ccd_roi_overlay.png
logs\roi_calibration\roi_report.json
```

报告中的 `recommended_roi_xywh=[x,y,956,956]` 有两种用法，二选一：

1. 推荐：相机继续输出全帧；把该值写入服务器 `grocery10_moe4_latest_hardware.yaml` 的 `capture.roi_xywh`。
2. 相机硬裁剪：写入本机 `acquisition_windows.json` 的 `camera.device_roi_xywh`；服务器的 `capture.roi_xywh` 保持 `null`。此时先检查 `resolved_devices.json` 是否读回同样 ROI。

绝不能两边同时裁剪。MoE16 的有效物理 ROI 是 `986×986`，不要误用 MoE4 的 `956×956`。

## 4. 可选设备 demo

数字顺序 demo 和相位 SLM demo 仍保留，但不属于逐层主流程：

```powershell
python -m experiments.hardware_sdk.amplitude_camera_demo `
  --config experiments/hardware_sdk/configs/amplitude_camera_digit_demo.yaml

python -m experiments.hardware_sdk.phase_slm_demo `
  --config experiments/hardware_sdk/configs/phase_slm_demo.yaml --dry-run
```

服务器端逐层电子处理命令见：

```text
experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/RUN_COMMANDS.md
```
