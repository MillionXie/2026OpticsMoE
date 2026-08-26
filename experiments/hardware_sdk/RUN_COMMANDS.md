# Hardware SDK 命令

所有命令都从仓库根目录执行，均为可直接复制的单行命令。

## 紧凑 SLM payload 重建

服务器只需传输 `478×478` 的 8-bit PNG。实验室用下面的确定性规则恢复完整画布：

    python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir compact_amplitude --output-dir amplitude_to_play --slm-width 1920 --slm-height 1080 --scale-factor 2
    python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir compact_phase --output-dir phase_to_play --slm-width 1920 --slm-height 1200 --scale-factor 2 --center-x 980 --center-y 590

每个逻辑像素严格重复为 `2×2`，再精确居中补零；basename 不变，输出目录中的
`reconstruction_manifest.csv` 记录源/目标 SHA256 和有效区坐标。

新 `17 µm` 输入 SLM 使用一像素对一像素；新相位导出使用物理 pitch，而不是整数放大：

    python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir compact_amplitude --output-dir amplitude_to_play --slm-width 1024 --slm-height 1024 --scale-factor 1 --center-x 512 --center-y 512
    python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir compact_phase --output-dir phase_to_play --slm-width 1920 --slm-height 1200 --logical-pixel-pitch-um 17 --slm-pixel-pitch-um 8 --center-x 980 --center-y 590

生成两块新 SLM 的棋盘格、光栅、MoE4 分区和孔径扫描 BMP：

    python -m experiments.hardware_sdk.generators.dual_slm_alignment --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml

## MoE4 CCD 采集后统一处理（独立于训练工程）

先编辑 `experiments/hardware_sdk/configs/process_moe4_ccd_956.yaml` 中的统一
`roi_xywh`。它应框住旧 `2×2` 专家的整个有效区域；程序缩放完整 ROI 到
`956×956`，输出旧流程兼容的 8-bit 灰度 PNG，不做翻转，也不会中心裁剪到 `224×224`。

    python -m experiments.hardware_sdk.workflows.process_moe4_ccd --config experiments/hardware_sdk/configs/process_moe4_ccd_956.yaml --input-dir experiments/hardware_sdk/data/ccd_captured_raw --output-dir experiments/hardware_sdk/data/ccd_captured_moe4_956

该命令只用于处理已经存在的旧 16-bit 文件。新采集由 `tucam_windows.yaml` 直接保存
`478×478` 8-bit PNG，不再生成需要二次处理的 956 中间图。强度使用统一的
`0→65535` 映射。这里不估计、
不扣除背景，也不对每张图自行拉伸；只有相机标定范围变化时才统一修改
`black_level/white_level`。命令行覆盖路径相对当前工作目录解析，YAML 内路径相对
YAML 文件解析。

## 1. 安装

    python -m pip install -r experiments\hardware_sdk\requirements-light.txt

## 2. 唯一主配置

旧 HOLOEYE 振幅 SLM 流程读取：

    experiments\hardware_sdk\configs\tucam_windows.yaml

新 1024×1024 Meadowlark Blink PCIe 振幅 SLM 与 TUCam 联合采集读取：

    experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml

使用新配置前必须完成三件事：

1. 在实验室 Windows 电脑安装 Meadowlark Blink Plus PCIe 驱动/runtime；仓库里的 DLL
   不是 PCIe 设备驱动的替代品。
2. 确认 `amplitude_slm.lut_file` 对应真实 SLM 和温度。配置默认指向上传 SDK 中的
   `slm7930_at532_30C.lut`；若实验使用 70 °C 标定，必须改为 `_70C.lut`。
3. 填写 `camera.device_roi_xywh`；四项都必须为 4 的倍数。

实验前至少填写：

    camera:
      exposure_us: 5000.0
      require_device_roi: true
      device_roi_xywh: [left, top, width, height]
      saved_frame_size_wh: [478, 478]
      saved_frame_resize_mode: auto
      saved_frame_bit_depth: 8
      saved_frame_input_range: [0, 65535]

TUCam 的 ROI 四项必须为 4 的倍数。程序先核对 SDK 返回的原始帧是否等于硬件
ROI，再核对保存结果是否等于 478×478。`auto` 对较大 ROI 使用 area，下采样；对
当前实验室的 472×472 ROI 使用 bilinear，小幅上采样到 478×478。保存格式默认为
8-bit 灰度 PNG。0～65535 到 0～255 是固定线性映射，不会对每张图单独拉伸。

## 3. 相机单独检查

    python -m experiments.hardware_sdk.tools.camera_smoke_test --config experiments\hardware_sdk\configs\tucam_windows.yaml --output-dir experiments\hardware_sdk\artifacts\demos\tucam_smoke --frames 3

## 4. 生成 ROI 观察图和曝光图案

python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\tucam_windows.yaml

新 1024×1024、17 µm Meadowlark 版本：

    python -m experiments.hardware_sdk.workflows.roi_calibration generate --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml

生成的 5 点、5 矩形、ROI 外框和灰度块位于：

    experiments\hardware_sdk\artifacts\calibration\masks

## 5. 0～9 播放顺序校验

    python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_windows.yaml

只生成图案、不打开设备：

python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_windows.yaml --generate-only

新 17 µm Meadowlark 版本：

    python -m experiments.hardware_sdk.demos.amplitude_camera_demo --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --generate-only

当前 `settle_delay_ms: 200` 表示 SLM 切图后等待 200 ms，再请求相机曝光。相机曝光
时间由 `camera.exposure_us` 控制。

## 6. 0～255 曝光响应检查

python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\tucam_windows.yaml

新 17 µm Meadowlark 版本：

    python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml

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

### 新 Meadowlark 1024×1024 振幅 SLM + TUCam

假设 `$STAGE` 是实验室电脑上的当前层目录。先把服务器传来的 478×478 紧凑振幅
重建为原生 1024×1024 BMP；这一步不做缩放，逻辑像素与振幅 SLM 像素一一对应：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
```

重建清单位于 `$STAGE\amplitude_to_play\reconstruction_manifest.csv`；下面的预检、
试拍和正式采集都以它作为严格 allowlist，避免播放目录内残留 BMP。

在打开硬件前，校验全部振幅 BMP、指定相位 BMP、两套 SDK、LUT 和相机 ROI：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --validate-only
```

手动在相位 SLM 上加载同一个 `--phase-mask` 文件。首次只采 3 张，确认播放、曝光和
basename 对应关系：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --limit 3 --clear-output
```

确认无误后清空上述 3 张并采完整批次：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --clear-output
```

stage 模式会自动要求 `phase_to_play/` 中恰好存在一张 BMP，并将其用于校验和记录。
程序不会打开相位 SLM；开始前会显示相位文件名和 SHA256，实验人员确认屏幕上加载
的是同一个文件后输入 `y`。振幅 BMP 必须是原生
1024×1024、8-bit 灰度 BMP，驱动拒绝隐式缩放、翻转和逐图归一化。

## 8. 四层光电实验

每层重复：下载该层振幅 BMP到 `amplitude_to_play` → 手动加载相位 mask → 正式采集
→ 将同名 PNG 上传到服务器对应 CCD 目录 → 执行该层电子处理/微调 → 下载下一层
振幅 BMP。

## 9. 旧 DVP 相机

    python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\legacy\dvp_windows.json --clear-output

旧 DVP 配置不受 TUCam 8-bit PNG 默认值影响。

## 10. 测试

    python -m pytest experiments\hardware_sdk\tests -q

    python -m pytest experiments\hardware_sdk\generators\slm_patterns\tests -q
