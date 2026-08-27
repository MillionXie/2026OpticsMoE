# Hardware SDK

四点 CCD 透视校正、canonical orientation 合同与 32 点快速亮度标定见
[GEOMETRY_AND_BRIGHTNESS.md](GEOMETRY_AND_BRIGHTNESS.md)。

这是可复制到实验室 Windows 电脑独立使用的轻量硬件工程。它支持新的 Meadowlark
Blink PCIe 1024×1024 振幅 SLM、原有 HOLOEYE SLM、TUCam/Mosaic CCD 和旧 DVP CCD，
不依赖 Torch 或 Qwen。

## 当前主流程

1. 从文件夹顺序播放与振幅 SLM 原生分辨率完全一致的 8-bit 灰度 BMP。
2. 等待配置的 SLM 稳定时间。
3. 由相机 SDK 采集配置的硬件 ROI。
4. 在原始硬件 ROI 上应用本 session 固定的四逻辑顶点 homography，一次性校正旋转、
   镜像、透视和非正方形几何，直接得到 canonical 478×478。
5. 用全样本一致的固定范围映射保存 8-bit 灰度 PNG。

旧的 axis-aligned ROI 后 `area/bilinear resize` 只用于设备 smoke test 与历史数据；不能
作为当前 Qwen 实测微调或 sim-to-real agreement 的正式输入。canonical 输出也不能在
下游再做横/纵翻转。

新 Meadowlark 振幅 SLM 使用
[configs/tucam_meadowlark_1024_windows.yaml](configs/tucam_meadowlark_1024_windows.yaml)；
旧 HOLOEYE 流程继续使用 [configs/tucam_windows.yaml](configs/tucam_windows.yaml)。
实验室通常需要确认 Meadowlark LUT，并修改相机 ROI、曝光时间、输入输出目录。
`saved_frame_input_range` 默认把 TUCam 的
uint16 0～65535 映射到 PNG 0～255；程序禁止逐图自动拉伸，以保留相对光强。

相位 SLM 仍由实验人员手动加载。正式采集命令必须接收对应的 1920×1200 相位 BMP，
校验其格式和尺寸，并把文件名与 SHA256 写入采集日志，从而防止振幅批次与相位 mask
错配；采集代码不会控制或修改相位 SLM。

`reconstruct_slm` 会在重建目录写入 `reconstruction_manifest.csv`，其中
`output_bmp` 是本批次唯一允许播放的 BMP basename。预检、试拍和正式采集都应把该
文件传给 `acquire_folder --file-manifest`；采集器会拒绝路径、重复项和缺失文件，且
不会误播目录中上次实验遗留的 BMP。它也兼容导出流程使用的 `amplitude_file` 和
`amplitude_bmp` 两种清单列名。

可选背景扣除与正式采集分离，而且不属于当前 agreement 合同。只有实际采集了同曝光、
同增益、同光路状态的背景帧时才能使用；不得把灰度 0 帧或臆测常数称为背景。正式
agreement 与网络输入均不做背景扣除。

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
