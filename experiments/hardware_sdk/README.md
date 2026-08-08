# Hardware SDK

这是可以复制到实验室 Windows 电脑独立使用的轻量硬件工程。当前新 TUCam
主流程只做三件事：振幅 SLM 文件夹播放、相机原始 ROI 采集、曝光响应检查。

## 唯一主配置

新相机只需要修改：

`configs/tucam_windows.yaml`

正式采集、相机检查、0～9 顺序校验、ROI 图生成和曝光响应检查都读取这一个文件。
旧 DVP 配置单独放在 `configs/legacy/`，不会和当前流程混在一起。

需要人工填写：

```yaml
camera:
  exposure_us: 5000.0
  require_device_roi: true
  device_roi_xywh: [left, top, width, height]
```

TUCam 的四个 ROI 数值都必须是 4 的倍数，且不得超出 2048×2048。厂商软件
中的 ROI 不会传给新的 Python 进程，因此代码会再次通过 SDK 设置，并核对相机
实际返回值。

如果配置的 ROI 为 956×956，程序直接保存相机返回的 956×956 原始 uint8/uint16
帧；不 resize、不扣 background、不归一化、不做几何变换。

## 最简流程

1. 人工确定相机 ROI，并只在 `tucam_windows.yaml` 填写一次。
2. 可选运行 0～9 顺序检查，确认 200 ms 延迟没有错帧。
3. 光路和 ROI 确定后运行一次 0～255 曝光响应检查。
4. 人工选择曝光时间，写回同一个配置。
5. 正式播放文件夹并直接保存相机硬件 ROI 原始帧。

光强、曝光、增益、ROI 或主要光路发生变化后，再重新运行曝光响应检查。当前
流程不采集 background，也不做 background subtraction。

## 目录

```text
hardware_sdk/
├── configs/
│   ├── tucam_windows.yaml       # 当前唯一主配置
│   ├── phase_slm_demo.yaml
│   └── legacy/dvp_windows.json  # 旧相机兼容配置
├── vendor_sdk/                  # 随 Git 同步的厂商 SDK
├── drivers/                     # 相机适配层
├── workflows/                   # 正式文件夹采集和曝光检查
├── demos/                       # 0～9 顺序校验
├── data/
│   ├── amplitude_to_play/
│   └── ccd_captured/
└── artifacts/
    ├── calibration/masks/
    ├── calibration/results/tucam/
    ├── demos/
    └── logs/
```

完整单行命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
