# Hardware SDK

这是可复制到实验室 Windows 电脑独立使用的轻量硬件工程，不依赖 Torch、
Qwen 或训练代码。它负责振幅 SLM 播放、CCD 原始采集、手动 ROI 配置、统一
background、灰度响应检查和离线后处理。

## 目录

```text
hardware_sdk/
├── vendor_sdk/
│   ├── amplitude_holoeye/       # HOLOEYE 振幅 SLM SDK
│   ├── phase_meadowlark/        # Meadowlark 相位 SLM SDK
│   ├── camera_dvp_legacy/       # 旧 DVP CCD SDK
│   └── camera_tucam_mosaic/     # 新 TUCam/Mosaic CCD SDK
├── configs/                     # 采集、手动 ROI 和设备配置
├── drivers/                     # 可替换的相机适配层
├── workflows/                   # 文件夹采集、background、后处理
├── demos/                       # 0～9 顺序校验和 SLM demo
├── generators/slm_patterns/     # 棋盘、透镜、字母等图案
├── data/
│   ├── amplitude_to_play/       # 本轮要播放的振幅 BMP
│   ├── ccd_captured/            # 相机原始 ROI 帧
│   └── processed/               # 统一离线处理结果
└── artifacts/
    ├── calibration/masks/       # zero、ROI 验证、曝光 BMP
    ├── calibration/results/     # background 和灰度曲线
    ├── demos/                   # 顺序校验结果
    └── logs/                    # 设备信息与采集 manifest
```

`vendor_sdk/` 随 Git 同步；大量相机帧和运行结果默认忽略。

## 手动相机 ROI

厂商软件中的 ROI 不会自动共享给另一个 Python 进程。正式采集、0～9 校验和
background 配置中都必须填写：

```yaml
require_device_roi: true
device_roi_xywh: [left, top, width, height]
```

JSON 中写法相同：

```json
"require_device_roi": true,
"device_roi_xywh": [548, 548, 956, 956]
```

上面的 `548,548` 只是 2048×2048 传感器的近似居中示例，不是你的最终实验
坐标。应把你人工确定的 `left/top/width/height` 填进去。TUCam 四个数均须为
4 的倍数。程序打开相机后会读取 SDK 报告的实际 ROI；与配置不一致就直接报错。

需要同步修改三个地方：

1. `configs/acquisition/tucam_windows.json`：正式文件夹采集；
2. `configs/calibration/tucam.yaml`：background 和曝光检查；
3. `configs/digit_sequence_tucam.yaml`：0～9 顺序/延迟校验。

三处必须使用完全相同的 ROI。

## ROI 验证图

`roi_calibration generate` 只生成供人工观察的图案，不自动识别相机坐标：

- `verify_roi_5points.bmp`：中心和四个 956×956 ROI 角点；
- `verify_roi_5rectangles.bmp`：相同位置的五个实心矩形；
- `verify_roi_outline.bmp`：完整 ROI 外框；
- `amplitude_zero.bmp`、`phase_zero.bmp`：统一 background 图案。

自动 coarse/fine、affine 和 homography 已移除。最终 ROI 由操作者设置。

## Background 顺序

background 必须在最终硬件 ROI 设置后重新采集。流程会：

1. 自动加载振幅 `amplitude_zero.bmp`；
2. 自动加载相位零图，或在相位 SLM 为手动模式时明确提示文件；
3. 等待 `settle_delay_ms`；
4. 打印两个实际文件的绝对路径并等待人工确认；
5. 再次加载振幅零图并等待稳定；
6. 打开相机、验证实际 ROI、采集多帧中位数。

这不是关闭激光得到的 dark frame，而是激光开启、两个 SLM 均为零图时的系统
background。

## 后处理原则

采集循环始终保存相机 SDK 返回的原始 ROI 数据，不 resize、不扣 background、
不归一化。离线后处理仅包含：

```text
corrected = maximum(raw - background, 0)
→ 可选 BOX 面积下采样
```

不再执行 affine、homography、透视 warp、自动配准、逐图归一化、滤波或增强。
如果 raw 与 background 尺寸不同，程序会要求在最终 ROI 下重新拍 background。

完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
