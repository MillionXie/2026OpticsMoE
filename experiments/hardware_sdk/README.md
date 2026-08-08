# Hardware SDK

该目录负责实验室设备控制、标定和原始 CCD 数据整理，不包含 Qwen、Torch 或模型训练逻辑。

## 目录职责

```text
hardware_sdk/
├─ drivers/                       # 相机厂商适配器
│  ├─ tucam_camera.py             # 新 Dhyana 400BSI V3 / TUCam
│  └─ README.md
├─ tools/                         # 独立设备自检工具
│  └─ camera_smoke_test.py
├─ configs/
│  ├─ acquisition/
│  │  ├─ tucam_windows.json       # 新 CCD，当前默认
│  │  └─ dvp_windows.json         # 旧 CCD，继续支持
│  └─ calibration/
│     └─ tucam.yaml               # 新 CCD 标定配置
├─ acquire_folder.py              # 振幅 SLM 播放 + 原始 CCD 采集
├─ roi_calibration.py             # background/coarse/fine/exposure
├─ batch_postprocess.py           # 正式采集后的统一离线处理
├─ calibration_common.py          # 标定共享纯函数
├─ devices.py                     # 稳定设备接口与工厂
├─ amplitude_camera_demo.py       # 旧振幅 SLM demo，保留兼容
├─ phase_slm_demo.py              # 旧相位 SLM demo，保留兼容
└─ slm_calibration_bmp_generator/ # 独立常用 SLM 图案生成器
```

本机厂商文件仍放在：

```text
amp_slm/          HOLOEYE 示例/SDK
phase_slm/        Meadowlark Blink SDK
ccd/              旧 DVP SDK
ccd_2_mosaic/     新 TUCam/Mosaic SDK 与官方 demo/PDF
```

这些二进制文件与许可证相关，已被 Git 忽略；GitHub 和服务器只同步共享适配代码、配置与文档。

## 相机后端

所有上层程序只调用：

```python
camera = build_camera(config, config_base)
camera.open()
camera.capture(path)
camera.close()
```

配置中的 `camera.driver` 决定后端：

- `tucam`：当前新 CCD，Dhyana 400BSI V3；
- `dvp_subprocess`：旧 DVP 相机及旧 Python ABI；
- `dvp`：旧 DVP 同进程模式。

新 TUCam 适配器严格参考上传的 `00_init_open.py`、`05_wait_frame.py`、`06_roi_mode.py` 和 `25_set_exposure.py`：

```text
Api_Init → Dev_Open → exposure/ROI
→ Buf_Alloc → Cap_Start
→ Buf_WaitForFrame
→ 复制原始 mono uint8/uint16 buffer
```

不调用 JPEG，不做显示拉伸，不把 16-bit 图像转成 8-bit。`exposure_us` 在公共配置中统一使用微秒，传入 TUCam 前按厂商 demo 转换成毫秒。

新相机配置采用厂商规格：原生 2048×2048、6.5 μm 像素、单色。正式使用时仍以 `device_info()` 和实际帧 metadata 为准。
TUCam 硬件 ROI 的 `left/top/width/height` 必须都是 4 像素的倍数；标定配置会按这一约束生成合法建议值。

## 正式采集原则

默认配置：

```json
"output_extension": ".npy",
"saved_frame_size_wh": null,
"saved_frame_resize_mode": "none"
```

因此采集循环只保存相机硬件 ROI 返回的原始帧：

- 不 resize；
- 不扣背景；
- 不做 affine/homography；
- 不逐图归一化；
- 保留 uint8/uint16 位深；
- manifest 保存帧号、UTC 时间、曝光、ROI、原始尺寸和 dtype。

所有处理统一交给 `batch_postprocess.py`。

## 标定流程

`roi_calibration.py` 提供：

- `generate`：两张 zero BMP、五点粗标定、九点精标定和曝光灰度块；
- `background`：激光开启、两块 SLM 均为零图时的系统背景；
- `coarse`：完整视野下估计硬件 ROI；
- `fine`：硬件 ROI 下比较 affine/homography；
- `exposure`：固定窗口检查灰度 0～255 响应与饱和。

系统 background 不是关闭激光得到的 dark frame。统一扣除方式为：

```python
corrected = maximum(raw.astype(float32) - background, 0)
```

`batch_postprocess.py` 先在接近 CCD 原采样密度下完成几何矫正，再以 `cv2.INTER_AREA` 缩小到 956×956，不在 warp 时直接低分辨率输出。

## 安装与测试

```powershell
python -m pip install -r requirements-light.txt
python -m pytest tests -q
```

新 TUCam SDK 直接使用当前 64-bit Windows Python 和 ctypes；旧 DVP 相机仍使用原来的独立兼容 Python，不互相影响。

完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
