# Hardware SDK 完整命令

以下命令均从仓库根目录执行：

```powershell
cd D:\code\guest\2026OpticsMoE
python -m pip install -r experiments\hardware_sdk\requirements-light.txt
```

## 1. 手动填写相机 ROI

先在厂商软件中人工确定 ROI，记录：

```text
left, top, width, height
```

然后将同一组数写入下面三个文件中的 `device_roi_xywh`：

```text
experiments\hardware_sdk\configs\acquisition\tucam_windows.json
experiments\hardware_sdk\configs\calibration\tucam.yaml
experiments\hardware_sdk\configs\digit_sequence_tucam.yaml
```

示例仅说明格式：

```json
"require_device_roi": true,
"device_roi_xywh": [548, 548, 956, 956]
```

TUCam 的 `left/top/width/height` 都必须是 4 的倍数。不要把示例坐标当作实验
标定结果。`null` 会被正式流程拒绝，以免误拍完整传感器画面。

## 2. 相机单独检查

```powershell
python -m experiments.hardware_sdk.tools.camera_smoke_test `
  --config experiments\hardware_sdk\configs\acquisition\tucam_windows.json `
  --output-dir experiments\hardware_sdk\artifacts\demos\tucam_smoke `
  --frames 3
```

检查输出的 `frame_*_preview.png`，并核对原始帧尺寸就是配置的 ROI 尺寸。

## 3. 生成零图和人工 ROI 验证图

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration generate `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

输出：

```text
artifacts\calibration\masks\amplitude\amplitude_zero.bmp
artifacts\calibration\masks\phase\phase_zero.bmp
artifacts\calibration\masks\amplitude\verify_roi_5points.bmp
artifacts\calibration\masks\amplitude\verify_roi_5rectangles.bmp
artifacts\calibration\masks\amplitude\verify_roi_outline.bmp
artifacts\calibration\masks\exposure\gray_000.bmp ... gray_255.bmp
```

这一步不再估计 ROI，不生成 affine/homography。

## 4. 0～9 播放和采集顺序校验

先确认 `digit_sequence_tucam.yaml` 已填写最终 ROI。运行：

```powershell
python -m experiments.hardware_sdk.demos.amplitude_camera_demo `
  --config experiments\hardware_sdk\configs\digit_sequence_tucam.yaml
```

输出：

```text
artifacts\demos\tucam_digit_sequence\input_bmp\
artifacts\demos\tucam_digit_sequence\ccd_captured\
artifacts\demos\tucam_digit_sequence\capture_order.csv
artifacts\demos\tucam_digit_sequence\input_vs_capture_order.png
```

联系表左侧是指令图，右侧是同序号 CCD 帧。若出现前一张、重复帧或错序，依次调：

```yaml
settle_delay_ms: 200
discard_frames_after_display: 1
```

建议先将 `settle_delay_ms` 调到 300～500 ms 确认绝对正确，再逐步降低。若图像
稳定但始终落后一帧，将 `discard_frames_after_display` 从 1 调为 2。

只生成 0～9 BMP、暂不打开设备：

```powershell
python -m experiments.hardware_sdk.demos.amplitude_camera_demo `
  --config experiments\hardware_sdk\configs\digit_sequence_tucam.yaml `
  --generate-only
```

## 5. 在最终 ROI 下采集统一 background

必须先完成第 1、4 步。保持激光开启，然后运行：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration background `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

程序会加载振幅零图；相位 SLM 为手动模式时，会打印 `phase_zero.bmp` 的路径并
等待确认。输出：

```text
artifacts\calibration\results\tucam\background.npy
artifacts\calibration\results\tucam\background.tif
artifacts\calibration\results\tucam\background_preview.png
artifacts\calibration\results\tucam\background_metadata.json
```

只要 ROI、曝光、增益或光路位置发生改变，就重新采集 background。

## 6. 可选：0～255 灰度响应检查

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml
```

输出响应 CSV、原始曲线、归一化曲线和典型灰度预览。程序不会自动修改正式曝光。

## 7. 正式振幅播放和原始 CCD 采集

把本轮全部 1920×1080、8-bit 灰度 BMP 放入：

```text
experiments\hardware_sdk\data\amplitude_to_play\
```

手动加载本层相位 mask，然后运行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\acquisition\tucam_windows.json `
  --clear-output
```

输出原始 ROI 帧：

```text
experiments\hardware_sdk\data\ccd_captured\*.npy
experiments\hardware_sdk\artifacts\logs\tucam\capture_manifest.csv
experiments\hardware_sdk\artifacts\logs\tucam\resolved_devices.json
```

## 8. 批量后处理

默认只扣除 background，不缩放：

```powershell
python -m experiments.hardware_sdk.workflows.batch_postprocess `
  --config experiments\hardware_sdk\configs\calibration\tucam.yaml `
  --input-dir experiments\hardware_sdk\data\ccd_captured `
  --background experiments\hardware_sdk\artifacts\calibration\results\tucam\background.npy `
  --output-dir experiments\hardware_sdk\data\processed
```

处理公式只有：

```text
maximum(raw.astype(float32) - background, 0)
```

需要将更大的硬件 ROI 做面积下采样到 956×956 时，才在 `tucam.yaml` 中设置：

```yaml
postprocess:
  resize_enabled: true
  target_width: 956
  target_height: 956
  resize_mode: area
```

不会执行 affine、homography、透视矫正或逐图归一化。

## 9. 旧 DVP 相机

旧驱动和 SDK 仍保留：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\acquisition\dvp_windows.json `
  --clear-output
```

## 10. 常用 SLM 图案

```powershell
python -m experiments.hardware_sdk.generators.slm_patterns `
  --config experiments\hardware_sdk\generators\slm_patterns\configs\slm_956.yaml
```

## 11. Grocery 多层实验交接

每层执行：

1. 从服务器下载该层 `amplitude_to_play/*.bmp`；
2. 清理本地 `data/amplitude_to_play/`，复制本层 BMP；
3. 手动加载该层相位 mask；
4. 执行第 7 节采集；
5. 执行第 8 节 background 扣除；
6. 把 `data/processed/` 同名文件上传到服务器对应层的 `ccd_captured/`；
7. 执行 Grocery 工程 `RUN_COMMANDS.md` 对应层的电子处理/微调；
8. 下载下一层振幅并重复。

## 12. 测试

```powershell
python -m pytest experiments\hardware_sdk\tests -q
python -m pytest experiments\hardware_sdk\generators\slm_patterns\tests -q
```
