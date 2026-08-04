# Hardware SDK：轻量振幅播放与 CCD 采集

这个目录可以单独复制到实验室 Windows 电脑使用。它不理解模型、Vision/Language、层数或电子后处理，只完成一件事：

```text
amplitude_to_play/*.bmp
→ 按文件名排序并预加载到振幅 SLM
→ 每张等待 40 ms
→ CCD 保存同名原始帧
→ ccd_captured/*.npy
```

服务器上的模型代码负责生成每一层的振幅 BMP，以及把这一层 CCD 结果转换成下一层振幅 BMP。相位 SLM 暂时仍由实验员手动换 mask。

## 核心文件

```text
hardware_sdk/
├── acquire_folder.py                  # 通用振幅播放 + CCD 采集
├── roi_calibration.py                 # 棋盘格定位与 ROI 建议
├── devices.py                         # HOLOEYE / DVP 驱动封装
├── dvp_capture_worker.py              # 厂商旧版 Python 的相机子进程
├── configs/acquisition_windows.json   # 实验室 Windows 配置
├── requirements-light.txt             # 仅 NumPy + Pillow，不含 Torch
├── amplitude_to_play/                 # 当前一层待播放 BMP；不会上传 Git
├── ccd_captured/                      # 当前一层 CCD 原始帧；不会上传 Git
└── logs/                              # 播放清单、设备读回与 ROI 报告
```

厂商 SDK 继续放在 `amp_slm/`、`ccd/`、`phase_slm/`，这些目录不会上传 Git。旧的数字/相位 demo 是可选工具，不参与主采集流程。

## 依赖与环境

主控制 Python 只需：

```powershell
python -m pip install -r requirements-light.txt
```

无需安装 PyTorch、Qwen、Transformers 或 CUDA。

HOLOEYE Windows 安装中的：

```text
HEDS_3_2_PYTHON=C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v3.2.2
HEDS_3_2_PYTHON_MODULES=...\python
...\win64\holoeye_slmdisplaysdk.dll
```

是正确的 Windows 结构，不需要 Linux 的 `libholoeye_slmdisplaysdk.so`。配置会展开 `%HEDS_3_2_PYTHON_MODULES%`。

DVP Python 扩展使用旧 ABI 时，让它在独立旧 Python 子进程中运行。当前上传的 Windows `dvp.pyd` 明确链接 `python36.dll`，所以它需要 64 位 CPython 3.6；Microsoft Store Python 3.12 不能直接加载。设置：

```powershell
$env:DVP_PYTHON = "C:\path\to\Python36\python.exe"
```

也可以直接修改 `configs/acquisition_windows.json` 中的 `camera.python_executable`。

## 相机设置

定量采集不能依赖厂商软件上一次打开时的状态。配置会显式关闭自动曝光，并设置曝光、模拟增益、预热帧、丢弃帧和设备 ROI；实际读回值写入 `logs/resolved_devices.json`。

默认先使用全传感器采集并在服务器裁 ROI，最容易排查坐标错误。确认 ROI 后也可在相机侧设置 `device_roi_xywh`，但 DVP 硬件 ROI 可能要求坐标/宽高对齐到特定倍数，应以实际读回值为准。

详细命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
