# Qwen 光路实验：唯一操作顺序

本文是本实验唯一的操作入口。所有 PowerShell 命令都在仓库根目录
`E:\code\guest\2026OpticsMoE` 执行。不要再编辑旧 `hardware.yaml`，不要运行
`Get-FileHash`，也不要手工填写 contract 路径或 SHA。

## 0. 准备一次环境和 LUT

```powershell
Set-Location E:\code\guest\2026OpticsMoE
conda activate xml
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-lab.txt
```

确认此文件真实存在：

```text
experiments\hardware_sdk\vendor_sdk\amplitude_meadowlark\LUT Files\slm7930_at532-70c-pixel-2.lut
```

如果实验室电脑上的 LUT 文件名不同，把实际 `.lut` 文件放到上面的 `LUT Files`
目录，并在下一步只改文件名。路径中虽然有空格，但不需要手写引号或反斜杠转义。

## 1. 唯一需要编辑的文件

打开：

```powershell
notepad experiments\lab_qwen\LAB_CONFIG.yaml
```

该文件只包含三类信息：

```yaml
amplitude_lut_filename: slm7930_at532-70c-pixel-2.lut
camera_exposure_us: 5000.0
logical_corners_full_sensor_xy:
  top_left: [1626, 281]
  top_right: [358, 285]
  bottom_right: [363, 1547]
  bottom_left: [1631, 1545]
```

四点是 CCD 的 `2048×2048` 全传感器坐标，完全不要求是 4 的倍数。标签描述
光场的逻辑方位。当前 CCD 左右镜像，因此逻辑 `top_left` 出现在画面右侧是正常的。

如果还没有测四点，把四项暂时都写成 `null`；不能只留一部分为 `null`：

```yaml
logical_corners_full_sensor_xy:
  top_left: null
  top_right: null
  bottom_right: null
  bottom_left: null
```

保存后运行唯一准备命令：

```powershell
python -m experiments.lab_qwen.prepare_lab
```

它总会生成：

```text
experiments\lab_qwen\generated\bootstrap_hardware.yaml
```

四点齐全时还会自动生成：

```text
experiments\lab_qwen\generated\formal_hardware.yaml
experiments\lab_qwen\generated\detector_homography_478.contract.json
experiments\lab_qwen\generated\prepare_report.json
```

你当前这组四点会自动得到硬件 ROI `[292,216,1404,1396]`。程序先向四周留
64 px 余量，再把硬件 ROI 向外对齐到 4 的倍数；四点透视校正最终输出严格的
`478×478` 正方形。看到 `"status": "ready"` 后即可正式采集。SHA 已写入生成配置，
无需复制或计算。

## 2. 双 SLM 对齐

按顺序检查三对图案。每次先在相位 SLM 软件中加载该目录的
`phase_1920x1200.bmp`，再运行对应命令；振幅图由 Python 自动播放。

### 2.1 棋盘格

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_qwen\calib\dual\01_checker_c64 `
  --output-dir experiments\lab_qwen\work\dual\01_checker_c64\ccd `
  --log-dir experiments\lab_qwen\work\dual\01_checker_c64\log `
  --file-manifest experiments\lab_qwen\calib\dual\01_checker_c64\amplitude_manifest.csv `
  --phase-mask experiments\lab_qwen\calib\dual\01_checker_c64\phase_1920x1200.bmp `
  --clear-output
```

### 2.2 大块图案，X 光栅

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_qwen\calib\dual\02_large_blocks_c48_x `
  --output-dir experiments\lab_qwen\work\dual\02_large_blocks_c48_x\ccd `
  --log-dir experiments\lab_qwen\work\dual\02_large_blocks_c48_x\log `
  --file-manifest experiments\lab_qwen\calib\dual\02_large_blocks_c48_x\amplitude_manifest.csv `
  --phase-mask experiments\lab_qwen\calib\dual\02_large_blocks_c48_x\phase_1920x1200.bmp `
  --clear-output
```

### 2.3 大块图案，Y 光栅

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_qwen\calib\dual\03_large_blocks_c48_y `
  --output-dir experiments\lab_qwen\work\dual\03_large_blocks_c48_y\ccd `
  --log-dir experiments\lab_qwen\work\dual\03_large_blocks_c48_y\log `
  --file-manifest experiments\lab_qwen\calib\dual\03_large_blocks_c48_y\amplitude_manifest.csv `
  --phase-mask experiments\lab_qwen\calib\dual\03_large_blocks_c48_y\phase_1920x1200.bmp `
  --clear-output
```

三组都应满足白色振幅区出现对应光栅，边界接近像素级重合。此处振幅约定为
`255=白/通光，0=黑/遮光`。

## 3. Fresnel 距离、方向和四角点

振幅 SLM 固定播放 `calib\fresnel\A_WHITE.bmp`。相位 SLM 分别手动加载以下文件。

### 3.1 单点 P1：寻找 10 cm 焦面和方向

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_qwen\calib\fresnel `
  --output-dir experiments\lab_qwen\work\fresnel\P1 `
  --log-dir experiments\lab_qwen\work\fresnel\P1_log `
  --file-manifest experiments\lab_qwen\calib\fresnel\amplitude_manifest.csv `
  --phase-mask experiments\lab_qwen\calib\fresnel\P1_POINT.bmp `
  --clear-output
```

### 3.2 四点 P4：读取四个逻辑角点

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_qwen\calib\fresnel `
  --output-dir experiments\lab_qwen\work\fresnel\P4 `
  --log-dir experiments\lab_qwen\work\fresnel\P4_log `
  --file-manifest experiments\lab_qwen\calib\fresnel\amplitude_manifest.csv `
  --phase-mask experiments\lab_qwen\calib\fresnel\P4_POINT.bmp `
  --clear-output
```

从这张全传感器图中读取四个焦点坐标，按光场逻辑身份填入 `LAB_CONFIG.yaml`。
不要按画面上的从左到右顺序擅自重新命名。P4 只用于拟合。

### 3.3 九点 P9：独立检查中心和畸变

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\bootstrap_hardware.yaml `
  --input-dir experiments\lab_qwen\calib\fresnel `
  --output-dir experiments\lab_qwen\work\fresnel\P9 `
  --log-dir experiments\lab_qwen\work\fresnel\P9_log `
  --file-manifest experiments\lab_qwen\calib\fresnel\amplitude_manifest.csv `
  --phase-mask experiments\lab_qwen\calib\fresnel\P9_POINT.bmp `
  --clear-output
```

如果 POINT 焦点太小不便观察，只把相位文件分别换成同编号的 `P1_CROSS.bmp`、
`P4_CROSS.bmp`、`P9_CROSS.bmp`；振幅仍然必须是全白 `A_WHITE.bmp`。

填好四点后再次运行：

```powershell
python -m experiments.lab_qwen.prepare_lab
```

必须看到 `"status": "ready"`。从下一步开始，所有采集都固定使用
`generated\formal_hardware.yaml`，不再使用 bootstrap。

## 4. 32 灰度 × 3 帧亮度/曝光标定

相位 SLM 手动加载：

```text
experiments\lab_qwen\calib\exposure\phase\phase_zero.bmp
```

运行：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_qwen\generated\formal_hardware.yaml
```

结果位于：

```text
experiments\lab_qwen\results\exposure\brightness_response.png
experiments\lab_qwen\results\exposure\slm_response.csv
```

若存在饱和，只改 `LAB_CONFIG.yaml` 的 `camera_exposure_us`，重新运行
`prepare_lab`，再重新跑本步骤。不要去修改生成的 `formal_hardware.yaml`。

## 5. 先做设备只读校验

把 `agree\04_language_global\phase_to_play` 中唯一的 BMP 加载到相位 SLM，然后运行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\agree\04_language_global `
  --validate-only
```

此命令不拍摄，只检查 LUT、SDK、相机、ROI、contract、相位尺寸和振幅文件。

## 6. 仿真—实测 CCD 一致性

保持上一步相位不变，采集 32 张：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\agree\04_language_global `
  --clear-output
```

评价 PCC、SSIM、NRMSE、余弦相似度、能量和质心误差，并绘图：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate `
  --session-dir experiments\lab_qwen\agree `
  --stages language_global

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_report `
  --evaluation-dir experiments\lab_qwen\agree\agreement_evaluation `
  --output-dir experiments\lab_qwen\results\agreement
```

## 7. 最后一层 quick210 快速实验

相位 SLM 加载 `last\04_language_global\phase_to_play` 中唯一的 BMP，然后依次运行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\last\04_language_global `
  --clear-output

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir experiments\lab_qwen\last `
  --device auto `
  --epochs 10

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report `
  --root experiments\lab_qwen `
  --session-dir experiments\lab_qwen\last `
  --output-dir experiments\lab_qwen\results\last
```

末层准确率和图表位于 `results\last`，离线微调不会重新加载完整 Qwen。

## 8. 四层逐层采集和服务器微调

顺序固定为：

```text
01_vision_expert -> 02_vision_global -> 03_language_expert -> 04_language_global
```

每层都遵循同一循环：实验室采集当前层 → 上传当前层 CCD → 服务器微调 → 服务器
导出下一层 → 下载下一层 → 再采集。不能同时采完四层，因为后一层输入依赖前一层实测 CCD。

### 8.1 实验室采集第一层

加载 `four\01_vision_expert\phase_to_play` 中唯一相位 BMP：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four\01_vision_expert `
  --clear-output
```

把第一层 CCD 和采集日志上传到服务器已有 session：

```powershell
scp -P 24096 -r experiments\lab_qwen\four\01_vision_expert\ccd_captured guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/01_vision_expert/
scp -P 24096 -r experiments\lab_qwen\four\01_vision_expert\acquisition_logs guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/01_vision_expert/
```

### 8.2 服务器微调第一层并导出第二层

在服务器仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage vision_expert --phase finetune --epochs 20 --upstream-source measured

CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage vision_global --phase export --upstream-source measured
```

实验室下载第二层：

```powershell
scp -P 24096 -r guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/02_vision_global experiments\lab_qwen\four\
```

### 8.3 第二层：采集、上传、微调、导出第三层

加载 `four\02_vision_global\phase_to_play` 中唯一相位 BMP，然后：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four\02_vision_global --clear-output
scp -P 24096 -r experiments\lab_qwen\four\02_vision_global\ccd_captured guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/02_vision_global/
scp -P 24096 -r experiments\lab_qwen\four\02_vision_global\acquisition_logs guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/02_vision_global/
```

服务器：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage vision_global --phase finetune --epochs 20 --upstream-source measured

CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_vision_global.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage language_expert --phase export --upstream-source measured
```

实验室下载第三层：

```powershell
scp -P 24096 -r guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/03_language_expert experiments\lab_qwen\four\
```

### 8.4 第三层：采集、上传、微调、导出第四层

加载 `four\03_language_expert\phase_to_play` 中唯一相位 BMP，然后：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four\03_language_expert --clear-output
scp -P 24096 -r experiments\lab_qwen\four\03_language_expert\ccd_captured guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/03_language_expert/
scp -P 24096 -r experiments\lab_qwen\four\03_language_expert\acquisition_logs guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/03_language_expert/
```

服务器：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_vision_global.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage language_expert --phase finetune --epochs 20 --upstream-source measured

CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_language_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage language_global --phase export --upstream-source measured
```

实验室下载第四层：

```powershell
scp -P 24096 -r guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/04_language_global experiments\lab_qwen\four\
```

### 8.5 第四层：采集、上传并完成最终微调

加载 `four\04_language_global\phase_to_play` 中唯一相位 BMP，然后：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four\04_language_global --clear-output
scp -P 24096 -r experiments\lab_qwen\four\04_language_global\ccd_captured guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/04_language_global/
scp -P 24096 -r experiments\lab_qwen\four\04_language_global\acquisition_logs guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/04_language_global/
```

服务器：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_language_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage language_global --phase finetune --epochs 20 --upstream-source measured
```

最终 checkpoint：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_language_global.pt
```

## 9. 不再使用的旧操作

- 不手写 `camera.device_roi_xywh`。
- 不手写 `camera.detector_geometry.contract_file` 或 SHA。
- 不运行 `detector_homography fit/apply`。
- 不准备 `raw_roi.npy`、`rectified_478.tif` 或 `<FIT命令输出的SHA>`。
- 不编辑 `generated` 和 `internal` 目录中的任何文件。
