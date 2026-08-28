# 当前唯一操作顺序

以下命令均在 PowerShell 的仓库根目录
`E:\code\guest\2026OpticsMoE` 执行。先运行 `conda activate xml`；不要激活 ZIP 内旧
虚拟环境。

## 0. 只修改这两个配置文件

1. `experiments\lab_qwen\config\hardware.yaml`
   - `camera.device_roi_xywh`：相机硬件 ROI，四个值均需被 4 整除；
   - `camera.exposure_us` 与 `exposure_calibration.exposure_times_us`：保持一致；
   - `amplitude_slm.lut_file`：按实际温度选 30C/70C；
   - 完成四点拟合后，把 `detector_geometry.enabled` 改成 `true`，填写合同路径和 SHA-256。
2. `experiments\lab_qwen\config\geometry.yaml`
   - 填写 P4 的四个逻辑顶点在 CCD 全图中的坐标；不能只按画面位置排序。

`config\bootstrap.yaml` 只用于尚不知道 ROI 时的全传感器标定；其中只需同步修改 LUT
温度和临时曝光。先检查硬件环境：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\bootstrap.yaml `
  --input-dir experiments\lab_qwen\calib\fresnel `
  --output-dir experiments\lab_qwen\work\smoke `
  --phase-mask experiments\lab_qwen\calib\fresnel\P1_POINT.bmp `
  --file-manifest experiments\lab_qwen\calib\fresnel\amplitude_manifest.csv `
  --validate-only --limit 1
```

## 1. 两个 SLM 对齐

只使用 `experiments\lab_qwen\calib\dual` 中配成对的 A/P BMP；目录内
`k1_pair_manifest.json` 给出一一对应关系。先规则棋盘，再不规则大块。振幅极性为
255=白/透光。
相位中心已设为 `(980,590)`，并保留原来的上下翻转。

## 2. CCD 距离与 ROI：Fresnel

振幅始终播放：

```text
experiments\lab_qwen\calib\fresnel\A_WHITE.bmp
```

相位依次播放：`P1_POINT.bmp`（移动 CCD 找 10 cm 焦面）、`P4_POINT.bmp`（四顶点）、
`P9_POINT.bmp`（独立检查中点/中心）。如果点太小不便肉眼观察，改播同编号
`*_CROSS.bmp`。相位灰度 0 是 0 rad，不是遮光；振幅没有单开口。

例如保存 P4 全传感器图（P1/P9 只需替换相位文件和输出目录名）：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\bootstrap.yaml `
  --input-dir experiments\lab_qwen\calib\fresnel `
  --output-dir experiments\lab_qwen\work\fresnel_p4 `
  --phase-mask experiments\lab_qwen\calib\fresnel\P4_POINT.bmp `
  --file-manifest experiments\lab_qwen\calib\fresnel\amplitude_manifest.csv --yes
```

填完 `geometry.yaml` 后生成固定合同：

```powershell
python -m experiments.hardware_sdk.workflows.detector_homography fit `
  --config experiments\lab_qwen\config\geometry.yaml `
  --output experiments\lab_qwen\config\geometry.json
Get-FileHash experiments\lab_qwen\config\geometry.json -Algorithm SHA256
```

把输出 SHA-256 与 `geometry.json` 路径填入 `hardware.yaml`，并启用
`detector_geometry.enabled: true`（`contract_file: geometry.json`）。正式网络采集后不再
额外左右/上下翻转。

## 3. 32 灰度×3 帧曝光标定

相位 SLM 加载 `calib\fresnel\P1_POINT.bmp`，然后执行：

```powershell
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_qwen\config\hardware.yaml --yes
```

查看 `experiments\lab_qwen\results\exposure\brightness_response.png`。出现饱和就降低
`exposure_us` 后重跑；该流程不把灰度 0 当背景，也不做背景扣除。

## 4. 仿真与实测光路差异

Agreement 数据已在 `experiments\lab_qwen\agree`。加载
`agree\04_language_global\phase_to_play` 中唯一相位，自动播放该阶段：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --stage-dir experiments\lab_qwen\agree\04_language_global --clear-output --yes
```

计算 PCC、SSIM、NRMSE、余弦相似度、能量比例和质心误差，并画图：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate `
  --session-dir experiments\lab_qwen\agree --stage language_global
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_report `
  --evaluation-dir experiments\lab_qwen\agree\agreement_evaluation `
  --output-dir experiments\lab_qwen\results\agreement
```

## 5. 最后一层快速正式测试

加载 `last\04_language_global\phase_to_play` 的相位，采集 210 帧：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --stage-dir experiments\lab_qwen\last\04_language_global --clear-output --yes
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir experiments\lab_qwen\last --device auto --epochs 10
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report `
  --root experiments\lab_qwen `
  --session-dir experiments\lab_qwen\last `
  --output-dir experiments\lab_qwen\results\last
```

## 6. 四层逐层采集与微调

ZIP 已含第一层 `experiments\lab_qwen\four\01_vision_expert`。实验室采集：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\config\hardware.yaml `
  --stage-dir experiments\lab_qwen\four\01_vision_expert --clear-output --yes
```

把 `four` 目录传回服务器仓库。服务器 GPU4 对当前层微调并导出下一层，随后再把
`four` 目录传回实验室。四层严格按以下顺序循环：

```text
vision_expert -> vision_global -> language_expert -> language_global
```

服务器命令（把 `STAGE` 与 `NEXT` 改成当前/下一层；最后一层不再 export NEXT）：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CONFIG=$PROJECT/configs/release/stage2_joint_hardware_canonical_ccd.yaml
SESSION=experiments/lab_qwen/four
MODULE=experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge
CUDA_VISIBLE_DEVICES=4 python -m $MODULE \
  --config $CONFIG --checkpoint experiments/lab_qwen/model/ema.pt \
  --session-dir $SESSION --stage STAGE --phase finetune
CUDA_VISIBLE_DEVICES=4 python -m $MODULE \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_STAGE.pt \
  --session-dir $SESSION --stage NEXT --phase export --upstream-source measured
```
