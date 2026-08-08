# Hardware SDK

这是一个可独立复制到实验室 Windows 电脑使用的轻量硬件工程，不依赖
Torch、Qwen 或训练代码。它负责 SLM 播放、CCD 原始采集、ROI/曝光标定和
离线后处理。

## 目录

```text
hardware_sdk/
├─ vendor_sdk/                         # 随 Git 同步的全部厂商 SDK
│  ├─ amplitude_holoeye/               # HOLOEYE 振幅 SLM
│  ├─ phase_meadowlark/                # Meadowlark 相位 SLM
│  ├─ camera_dvp_legacy/               # 旧 DVP CCD
│  └─ camera_tucam_mosaic/             # 新 Mosaic/TUCam CCD
├─ configs/
│  ├─ acquisition/                     # 新旧 CCD 正式采集配置
│  └─ calibration/                     # 新 CCD 标定配置
├─ drivers/                            # 可替换相机后端
├─ tools/                              # 单设备 smoke test
├─ workflows/                          # 采集、标定、统一后处理
├─ demos/                              # 振幅/相位 SLM 独立 demo
├─ legacy/                             # 旧 DVP worker 与环境安装脚本
├─ generators/slm_patterns/            # 棋盘格、透镜、字母等 BMP
├─ data/
│  ├─ amplitude_to_play/               # 本轮要播放的振幅 BMP
│  ├─ ccd_captured/                    # 原始 CCD 帧
│  └─ processed/                       # 离线矫正后的定量帧
├─ artifacts/
│  ├─ calibration/masks/               # 标定 BMP
│  ├─ calibration/results/             # background/矩阵/曲线
│  ├─ demos/                           # demo 输出
│  └─ logs/                            # 采集 manifest 与设备信息
├─ devices.py                          # 稳定设备接口和 driver 工厂
├─ README.md
└─ RUN_COMMANDS.md
```

`vendor_sdk/` 会提交到 Git；`data/` 和 `artifacts/` 中的运行结果默认忽略，
避免把数百 MB 实验帧混入代码版本。空目录通过 `.gitkeep` 保留。

## 相机切换

上层只调用 `devices.build_camera`，由配置中的 `camera.driver` 选择后端：

- `tucam`：新 Dhyana 400BSI V3 / Mosaic TUCam；
- `dvp_subprocess`：旧 DVP 相机及其独立 Python ABI；
- `dvp`：旧 DVP 同进程模式。

新 TUCam 后端执行：

```text
Api_Init → Dev_Open → 设置曝光/ROI
→ Buf_Alloc → Cap_Start → WaitForFrame
→ 按 frame header、stride 和位深复制原始单色数据
```

公共配置的曝光单位是 μs，传入 TUCam 时转换为 ms。TUCam ROI 的
`left/top/width/height` 必须都是 4 像素的倍数。

## 956×956 ROI 验证图

`roi_calibration.py generate` 会生成：

- `verify_roi_5points.bmp`：中心点和四个精确 ROI 角点，角点跨度就是
  956×956，不使用自动检测所需的 96 像素内缩；
- `verify_roi_5rectangles.bmp`：同样五个位置改为 80×80 实心矩形；
- `verify_roi_outline.bmp`：956×956 ROI 的完整矩形轮廓；
- `verify_5points.bmp`：兼容文件名，内容与修正后的五点图相同。

`marker_margin_px` 只影响自动 coarse/fine 的单标记图，不再影响以上人工
ROI 验证图。

## 数据原则

正式采集默认：

```json
"output_extension": ".npy",
"saved_frame_size_wh": null,
"saved_frame_resize_mode": "none"
```

因此保存相机硬件 ROI 返回的原始 uint8/uint16 帧，不在采集循环中 resize、
扣背景、做几何变换或逐图归一化。`capture_manifest.csv` 记录播放顺序、曝光、
ROI、原始尺寸、保存尺寸、dtype 和 UTC 时间。所有处理统一交给
`batch_postprocess.py`。

完整命令和先后顺序见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
