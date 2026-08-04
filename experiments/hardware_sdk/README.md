# Shared hardware SDK layer

该目录是仓库内所有实物光路实验共用的硬件接口。厂商二进制和示例放在同级目录但不上传 Git：

```text
experiments/hardware_sdk/
├── amp_slm/                 # HOLOEYE SDK（本机文件，Git ignore）
├── ccd/                     # DVP SDK（本机文件，Git ignore）
├── phase_slm/               # Meadowlark Blink SDK（本机文件，Git ignore）
├── devices.py               # 公共 SLM / Camera driver
├── dvp_capture_worker.py    # Python 3.5 DVP 常驻采集进程
├── amplitude_camera_demo.py
├── phase_slm_demo.py
└── slm_calibration_bmp_generator/
```

## 振幅 SLM

HOLOEYE driver 会检查原生 runtime、设备分辨率和刷新率；每张 BMP 必须与面板分辨率完全一致，禁止隐式缩放。每个光学平面的整批 BMP 会先 preload 到 GPU，程序等 data handle 进入 `Visible` 后才开始稳定延迟。

上传目录中仍未包含 Linux 原生 `libholoeye_slmdisplaysdk.so`。采集电脑必须安装完整 HOLOEYE Display SDK，并设置：

```bash
export HEDS_3_2_PYTHON=/path/to/installed/HOLOEYE/SDK
```

或在 YAML 中把 `binary_folder` 指向包含该 `.so` 的目录。

BMP 灰度值只是数字驱动值，不自动等于线性光学振幅。偏振器角度、面板 gamma/LUT 和实际透过率需要单独标定。不要再让显示软件做 fit、gamma 或自动对比度。

## DVP 相机

当前 Linux SDK 是 Python 3.5 ABI，因此公共 driver 保持训练/后处理在 Python 3.11，并用 `dvp_capture_worker.py` 在 `dvp35` 环境中持续打开相机。

定量实验不能依赖厂商软件窗口中“上一次”的状态。配置优先级是：

1. 若 `config_file` 非空，先执行 DVP `LoadConfig()`；
2. YAML 中的显式值随后覆盖它；
3. 实际读回的 exposure、analog gain、ROI 等写入 session 的 `resolved_hardware_devices.json`。

推荐固定：

```yaml
auto_exposure: false
exposure_us: 10000.0       # 只是起始值，应按直方图校准
analog_gain: 1.0
anti_flicker_hz: 0
warmup_frames: 3
discard_frames_after_display: 1
```

`settle_delay_ms: 40` 是 SLM 已确认可见后的等待；随后丢弃一个相机帧，再保存下一帧，避免读到上一张 BMP 的缓存帧。因此真实单帧周期会大于 40 ms。若相机支持硬触发，后续应优先改成 SLM/CCD 硬同步。

自动曝光会让不同样本/不同层的标度发生变化，强光时还会饱和；自动增益会改变噪声。因此 MSE/PCC 对照、电子重载和最终准确率实验都应关闭它们。当前 driver 要求单通道 MONO 原始帧；若 SDK 返回 RGB，会直接报错。

## 相位 SLM

主实验暂时仍人工换相位 mask。独立 demo 使用 Meadowlark Blink Windows DLL，明确加载 532 nm LUT，并可把 WFC 以 8-bit 模 256 加到相位 BMP 后显示。Linux 服务器只能执行 `--dry-run` 的图像检查，不能加载 Windows DLL。

完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
