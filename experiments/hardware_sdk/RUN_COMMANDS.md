# Hardware SDK 命令

所有命令都从仓库根目录执行，均为可直接复制的单行命令。

## MoE4 CCD 采集后统一处理（独立于训练工程）

先编辑 `experiments/hardware_sdk/configs/process_moe4_ccd_956.yaml` 中的统一
`roi_xywh`。它应框住旧 `2×2` 专家的整个有效区域；程序缩放完整 ROI 到
`956×956`，输出 8-bit 灰度 PNG，不做翻转，也不会中心裁剪到 `224×224`。

    python -m experiments.hardware_sdk.workflows.process_moe4_ccd --config experiments/hardware_sdk/configs/process_moe4_ccd_956.yaml --input-dir experiments/hardware_sdk/data/ccd_captured_raw --output-dir experiments/hardware_sdk/data/ccd_captured_moe4_956

默认对整个文件夹估计一组共同的百分位强度范围，不会逐图归一化。相机黑电平和饱和
值已知时，把 `intensity.mode` 改为 `fixed_range` 并填写 `black_level/white_level`。

## 1. 安装

    python -m pip install -r experiments\hardware_sdk\requirements-light.txt

## 2. 唯一主配置

当前 TUCam、振幅 SLM、正式采集、顺序校验、曝光检查和可选背景都读取：

    experiments\hardware_sdk\configs\tucam_windows.yaml

实验前至少填写：

    camera:
      exposure_us: 5000.0
      require_device_roi: true
      device_roi_xywh: [left, top, width, height]
      saved_frame_size_wh: [956, 956]
      saved_frame_resize_mode: area
      saved_frame_bit_depth: 8
      saved_frame_input_range: [0, 65535]

TUCam 的 ROI 四项必须为 4 的倍数。程序先核对 SDK 返回的原始帧是否等于硬件
ROI，再核对保存结果是否等于 956×956，因此硬件 ROI 为 1200×1200 时也可正确
压缩。保存格式默认为 8-bit 灰度 PNG。0～65535 到 0～255 是固定线性映射，不会
对每张图单独拉伸。

## 3. 相机单独检查

    python -m experiments.hardware_sdk.tools.camera_smoke_test --config experiments\hardware_sdk\configs\tucam_windows.yaml --output-dir experiments\hardware_sdk\artifacts\demos\tucam_smoke --frames 3

## 4. 生成 ROI 观察图和曝光图案

    python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\tucam_windows.yaml

生成的 5 点、5 矩形、ROI 外框和灰度块位于：

    experiments\hardware_sdk\artifacts\calibration\masks

## 5. 0～9 播放顺序校验

    python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_windows.yaml

只生成图案、不打开设备：

    python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_windows.yaml --generate-only

当前 `settle_delay_ms: 200` 表示 SLM 切图后等待 200 ms，再请求相机曝光。相机曝光
时间由 `camera.exposure_us` 控制。

## 6. 0～255 曝光响应检查

    python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\tucam_windows.yaml

结果位于：

    experiments\hardware_sdk\artifacts\calibration\results\tucam

光强、曝光、增益、ROI 或主要光路不变时通常只需执行一次。

## 7. 正式采集

把本层全部 1920×1080、8-bit 灰度 BMP 放入 YAML 的 `input_dir`，手动加载本层
相位 mask，然后运行：

    python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_windows.yaml --clear-output

默认输入与输出：

    experiments\hardware_sdk\data\amplitude_to_play\*.bmp
    experiments\hardware_sdk\data\ccd_captured\*.png

采集 manifest 与设备实际配置位于：

    experiments\hardware_sdk\artifacts\logs\tucam

正式采集不扣背景、不做几何变换、不逐图归一化。

## 8. 可选背景扣除

它不属于正式采集。背景、待处理图像和输出目录统一在主 YAML 中设置：

    optional_background:
      frames: 10
      zero_amplitude_bmp: ../artifacts/calibration/masks/amplitude/amplitude_zero.bmp
      source_capture_dir: ../data/ccd_captured
      background_dir: ../artifacts/optional_background/current_layer
      background_filename: background.png
      corrected_output_dir: ../data/ccd_background_subtracted
      output_extension: .png

保持采集目标时的相位 mask 不变，采集全零振幅背景：

    python -m experiments.hardware_sdk.workflows.optional_background capture --config experiments\hardware_sdk\configs\tucam_windows.yaml

对 YAML 指定的正式采集目录扣背景：

    python -m experiments.hardware_sdk.workflows.optional_background subtract --config experiments\hardware_sdk\configs\tucam_windows.yaml --clear-output

背景保存为 `background.png`，扣除结果也保存为 8-bit PNG。计算为
`maximum(raw.astype(float32) - background, 0)`，不覆盖原图、不归一化。每个相位层
应单独采集背景，只需修改 YAML 中的目录名称。

## 9. Grocery 多层实验

每层重复：下载该层振幅 BMP到 `amplitude_to_play` → 手动加载相位 mask → 正式采集
→ 将同名 PNG 上传到服务器对应 CCD 目录 → 执行该层电子处理/微调 → 下载下一层
振幅 BMP。

## 10. 旧 DVP 相机

    python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\legacy\dvp_windows.json --clear-output

旧 DVP 配置不受 TUCam 8-bit PNG 默认值影响。

## 11. 测试

    python -m pytest experiments\hardware_sdk\tests -q

    python -m pytest experiments\hardware_sdk\generators\slm_patterns\tests -q
