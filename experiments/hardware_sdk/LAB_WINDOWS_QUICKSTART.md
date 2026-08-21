# 实验室 Windows：SLM 重建快速说明

所有命令都从仓库根目录运行，例如：

```powershell
PS D:\code\guest\2026OpticsMoE>
```

实验室电脑只做 SLM/CCD 时无需安装 Qwen 或 CUDA。安装厂商驱动后，用 Python 3.11
执行轻量依赖安装即可：

```powershell
python -m pip install -r experiments\hardware_sdk\requirements-light.txt
```

## 为什么 `--input-dir compact_amplitude` 会报错

相对路径会相对当前 PowerShell 目录解析。位于仓库根目录时，`compact_amplitude`
代表：

```text
D:\code\guest\2026OpticsMoE\compact_amplitude
```

正常情况下，payload 实际位于某个 session stage，例如：

```text
experiments\...\hardware_sessions\vision2_run1\01_vision_expert\compact_amplitude
```

可用下面的命令查找已经复制到实验室电脑的 payload：

```powershell
Get-ChildItem -Path . -Directory -Recurse -Filter compact_amplitude
```

如果没有任何结果，说明服务器生成的 `compact_amplitude` 尚未复制到实验室电脑；代码无法从不存在的目录重建 BMP。

## 推荐命令

先把 `$STAGE` 改成查找到的真实 stage 目录。路径可以是绝对路径，也可以是相对仓库根目录的路径：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation\hardware_sessions\vision2_run1\01_vision_expert"
```

新版工具会自动寻找输入子目录、创建输出子目录，并根据 payload 选择正确的 SLM 尺寸：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um

python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload phase --hardware-profile meadowlark_17um --center-x 980 --center-y 590
```

生成结果分别位于：

```text
$STAGE\amplitude_to_play\*.bmp
$STAGE\phase_to_play\*.bmp
```

`--stage-dir` 是新流程的唯一工作目录。不要在仓库根目录另外创建或移动
`compact_amplitude`、`amplitude_to_play`、`ccd_captured`；重建、采集结果和日志均留在
对应工程的 stage 内：

```text
$STAGE\compact_amplitude\
$STAGE\amplitude_to_play\
$STAGE\phase_to_play\
$STAGE\ccd_captured\
$STAGE\acquisition_logs\
```

旧的显式 `--input-dir/--output-dir/--slm-width/--slm-height` 用法继续兼容。

## 新 17 µm 输入 SLM / 8 µm 相位 SLM

新输入 SLM 不放大，直接把478逻辑像素一对一放入 `1024×1024`：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir "$STAGE\compact_amplitude" --output-dir "$STAGE\amplitude_to_play" --slm-width 1024 --slm-height 1024 --scale-factor 1 --center-x 512 --center-y 512
```

相位端按物理像素间距 `17/8` 栅格化，并允许单独改变相位 SLM 中心：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir "$STAGE\compact_phase" --output-dir "$STAGE\phase_to_play" --slm-width 1920 --slm-height 1200 --logical-pixel-pitch-um 17 --slm-pixel-pitch-um 8 --center-x 980 --center-y 590
```

该过程不会插值相位灰度；每个原生8 µm像素只取一个逻辑相位值。纵向翻转已经在服务器导出的紧凑 phase 中完成。

## 新 Meadowlark 振幅 SLM 与 TUCam 采集

新设备不是原有 HOLOEYE/HDMI 接口，而是 Meadowlark Blink PCIe 的 board-indexed
接口。使用前先安装厂商 Blink Plus PCIe 驱动/runtime，并编辑：

```text
experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

至少确认实际 LUT（默认 30 °C 标定，SDK 也提供 70 °C 版本）并填写四项均为 4 的
倍数的 `camera.device_roi_xywh`。随后先做不打开设备的完整预检：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
```

手动把上面同一张相位 BMP 加载到相位 SLM 后，先采 3 张：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
```

确认后删除试采结果并移除 `--limit 3` 采完整批次。程序按文件名排序执行
`Meadowlark Write_image → ImageWriteComplete → 等待 200 ms → TUCam capture`，
直接保存同名 478×478 uint8 PNG。没有背景扣除、逐图拉伸或中间大图。

当前实验室 ROI 返回 `472×472` 时，配置中的 `saved_frame_resize_mode: auto` 会使用
双线性几何重采样得到 `478×478`；如果以后 ROI 调为不小于 `478×478`，会自动切换为
面积下采样。该选择会记录在 `acquisition_logs/capture_manifest.csv`，不做强度归一化。

## 17 µm Meadowlark 手写数字与曝光校准

两类输出也全部位于 Caltech101 四层工程的
`hardware_sessions/_calibration_17um/`，不会落到仓库根目录。

先只生成 1024×1024、8-bit、17 µm SLM 用的离线手写风格 0～9 顺序图（不依赖
Torch/torchvision 或联网下载）：

```powershell
python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --generate-only
```

连接设备后移除 `--generate-only`，即可顺序播放并采集；结果和顺序对照图仍写入同一校准目录。

生成 0～255 灰阶曝光图、ROI 五点和边界图（不打开硬件）：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

加载配置生成的 `masks\phase\phase_zero.bmp` 后执行曝光响应采集：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

这些图的振幅画布均为 `1024×1024`，有效区为中心 `478×478`，不再使用旧版
1920×1080/956 像素参数。

## 实验室电脑没有 Git 时如何只更新这个工具

在仓库根目录执行以下 PowerShell 命令，下载本次需要的文件：

```powershell
Invoke-WebRequest "https://raw.githubusercontent.com/MillionXie/2026OpticsMoE/main/experiments/hardware_sdk/workflows/reconstruct_slm.py" -OutFile "experiments\hardware_sdk\workflows\reconstruct_slm.py"
Invoke-WebRequest "https://raw.githubusercontent.com/MillionXie/2026OpticsMoE/main/experiments/hardware_sdk/workflows/acquire_folder.py" -OutFile "experiments\hardware_sdk\workflows\acquire_folder.py"
Invoke-WebRequest "https://raw.githubusercontent.com/MillionXie/2026OpticsMoE/main/experiments/hardware_sdk/workflows/roi_calibration.py" -OutFile "experiments\hardware_sdk\workflows\roi_calibration.py"
Invoke-WebRequest "https://raw.githubusercontent.com/MillionXie/2026OpticsMoE/main/experiments/hardware_sdk/devices.py" -OutFile "experiments\hardware_sdk\devices.py"
Invoke-WebRequest "https://raw.githubusercontent.com/MillionXie/2026OpticsMoE/main/experiments/hardware_sdk/drivers/tucam_camera.py" -OutFile "experiments\hardware_sdk\drivers\tucam_camera.py"
Invoke-WebRequest "https://raw.githubusercontent.com/MillionXie/2026OpticsMoE/main/experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml" -OutFile "experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml"
```

如果实验室电脑不能访问 GitHub，不必更新代码也能立即使用旧命令，只需要填写真正存在的完整目录：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir "$STAGE\compact_amplitude" --output-dir "$STAGE\amplitude_to_play" --slm-width 1024 --slm-height 1024 --scale-factor 1 --center-x 512 --center-y 512

python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir "$STAGE\compact_phase" --output-dir "$STAGE\phase_to_play" --slm-width 1920 --slm-height 1200 --logical-pixel-pitch-um 17 --slm-pixel-pitch-um 8 --center-x 980 --center-y 590
```
