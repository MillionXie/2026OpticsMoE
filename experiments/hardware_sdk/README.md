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

DVP Python 扩展使用独立厂商环境运行。当前可确认存在的模块位于 `ccd/lib/windows/python3.6/x64/dvp.pyd`，它链接 `python36.dll`，所以默认使用独立的 64 位 Python 3.6 相机环境：

```powershell
$env:DVP_PYTHON = "C:\Users\MMLAB\.conda\envs\miniCamera36\python.exe"
```

程序会把 SDK 目录同时加入 Python 模块路径和 Windows DLL 搜索路径，因此同目录的 `DVPCamera64.dll` 可以被找到。程序也会读取 `dvp.pyd` 的 ABI，并在启动前拒绝 3.6/3.7/3.8 混用。若以后拿到真正的 Python 3.7 版 `dvp.pyd`，只需将 `camera.sdk_path` 和 `DVP_PYTHON` 一起切换到匹配版本。

## 相机设置

定量采集不能依赖厂商软件上一次打开时的状态。配置会显式关闭自动曝光，并设置曝光、模拟增益、预热帧、丢弃帧和设备 ROI；实际读回值写入 `logs/resolved_devices.json`。

默认先使用全传感器采集并在服务器裁 ROI，最容易排查坐标错误。确认 ROI 后也可在相机侧设置 `device_roi_xywh`，但 DVP 硬件 ROI 可能要求坐标/宽高对齐到特定倍数，应以实际读回值为准。

详细命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
