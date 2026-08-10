# Hardware SDK

这是可复制到实验室 Windows 电脑独立使用的轻量硬件工程。它复用现有 HOLOEYE
振幅 SLM、TUCam/Mosaic CCD 和旧 DVP CCD 驱动，不依赖 Torch 或 Qwen。

## 当前主流程

1. 从文件夹顺序播放 1920×1080、8-bit 灰度 BMP。
2. 等待配置的 SLM 稳定时间。
3. 由相机 SDK 采集配置的硬件 ROI。
4. 可选用 area 将硬件 ROI 缩小到 956×956。
5. 用全样本一致的固定范围映射保存 8-bit 灰度 PNG。

主配置只有 [configs/tucam_windows.yaml](configs/tucam_windows.yaml)。实验室通常只需
修改相机 ROI、曝光时间、输入输出目录。`saved_frame_input_range` 默认把 TUCam 的
uint16 0～65535 映射到 PNG 0～255；程序禁止逐图自动拉伸，以保留相对光强。

可选背景扣除与正式采集分离。背景本身、扣除结果和预览均为 PNG；所有目录均在
主 YAML 的 `optional_background` 中指定，执行命令无需再重复传路径。

## 目录

```text
hardware_sdk/
├── configs/                    # 一个 TUCam 主配置 + legacy 配置
├── vendor_sdk/                 # 厂商 SDK
├── drivers/                    # 薄设备适配层
├── workflows/                  # 正式采集、曝光和可选背景
├── demos/                      # 0～9 顺序校验
├── data/
│   ├── amplitude_to_play/
│   ├── ccd_captured/
│   └── ccd_background_subtracted/
└── artifacts/                  # 图案、标定输出和日志
```

完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
