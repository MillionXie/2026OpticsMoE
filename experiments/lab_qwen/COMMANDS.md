# 新实验室完整流程：唯一命令文档

本文件按实际执行顺序书写。所有 PowerShell 命令均在本独立工程根目录
`E:\code\guest\qwen_mnist4_full_lab` 执行。实验人员只编辑
`experiments\lab_qwen\LAB_CONFIG.yaml`，不要编辑 `generated` 目录，也不需要手算
ROI、透视变换或 SHA-256。

## 0. 解压与环境

把完整 ZIP 解压后，先执行：

```powershell
Set-Location E:\code\guest\qwen_mnist4_full_lab
conda activate xml
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-lab.txt
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-offline-finetune.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

确认下面两个实验室专用文件确实存在：

```text
experiments\hardware_sdk\vendor_sdk\amplitude_meadowlark\LUT Files\slm7930_at532-70c-pixel-2.lut
experiments\hardware_sdk\vendor_sdk\camera_tucam_mosaic\TUCam.dll
```

不要激活包内 `.venv_capture`；全程使用 `xml` 环境。

## 1. 唯一配置和自动生成 ROI

打开唯一需要编辑的文件：

```powershell
notepad experiments\lab_qwen\LAB_CONFIG.yaml
```

只需填写 LUT 文件名、曝光时间和四个逻辑角点：

```yaml
amplitude_lut_filename: slm7930_at532-70c-pixel-2.lut
camera_exposure_us: 5000.0
logical_corners_full_sensor_xy:
  top_left: [1626, 281]
  top_right: [358, 285]
  bottom_right: [363, 1547]
  bottom_left: [1631, 1545]
```

四点是 CCD 的 2048×2048 全传感器坐标，不要求是 4 的倍数。标签表示光场的逻辑
方位；当前系统左右镜像，所以逻辑左上角出现在相机画面右侧是正常的。若换光路且还没
测量四点，先把四项全部改成 `null`，不能只留部分为 `null`。

每次修改后只运行：

```powershell
python -m experiments.lab_qwen.prepare_lab
```

四点齐全时必须看到 `"status": "ready"`。程序自动生成：

```text
experiments\lab_qwen\generated\formal_hardware.yaml
experiments\lab_qwen\generated\detector_homography_478.contract.json
experiments\lab_qwen\generated\prepare_report.json
```

当前四点自动得到硬件 ROI `[292,216,1408,1396]`；程序先留 64 px 余量，再按本机
TUCam 的实测约束向外对齐：`left/top/height` 为 4 的倍数、`width` 为 8 的倍数，
最后透视校正为严格 478×478。不要把宽度改回 1404；相机会将其静默截成 1400，
从而触发 ROI mismatch 并使透视合同失效。

## 2. 双 SLM 像素级对齐

每组都先用相位 SLM 软件加载目录内的 `phase_1920x1200.bmp`，再运行对应命令；
Python 只控制高速振幅 SLM 和 CCD。

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

### 2.2 大块图案与 X 光栅

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

### 2.3 大块图案与 Y 光栅

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

振幅约定固定为 `255=白/通光，0=黑/遮光`。白色区域应出现对应光栅，边界应接近
像素级重合。

## 3. Fresnel 距离、方向与四角点

振幅 SLM 始终播放全白 `calib\fresnel\A_WHITE.bmp`。相位 SLM 分别加载下面文件。

### 3.1 P1：寻找 10 cm 焦面

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

### 3.2 P4：读取四个逻辑角点

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

将四个焦点的全传感器坐标按逻辑身份填入 `LAB_CONFIG.yaml`，再次运行：

```powershell
python -m experiments.lab_qwen.prepare_lab
```

必须看到 `"status": "ready"`。之后全部正式采集只用 `generated\formal_hardware.yaml`。

### 3.3 P9：独立检查中心和畸变

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

若点太小，可把 `P1/P4/P9_POINT.bmp` 换成同编号的 `CROSS.bmp`；振幅仍为全白。

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

结果在 `experiments\lab_qwen\results\exposure`。若饱和，只改 `LAB_CONFIG.yaml` 中的
曝光，再运行 `prepare_lab` 和本步骤。

## 5. MNIST-4 简单任务：先 quick40，再 formal400

任务只识别数字 0、1、2、3。输入为 478×478 有效场，单层相位，532 nm、17 µm、
传播 10 cm、1.10° k 空间截止。CCD 分类严格使用四个 59×59 区域的原始强度和
argmax；不做 CCD 后归一化、非线性、背景扣除或再次缩放。四点透视校正只负责把
相机坐标恢复为模型的 478×478 坐标。

四个可选 mask 按建议顺序为：

```text
post_robust_best
mid_robust_energy
pre_robust_best
early_robust
```

### 5.1 生成 quick40 会话

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_session `
  --profile quick40 `
  --mask post_robust_best `
  --output-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40
```

相位 SLM 手动加载下面目录内唯一的 BMP：

```text
experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40\phase_to_play
```

采集前可先在实验室电脑生成与 40 张输入同名的黑白仿真 CCD。灰度图是黑底白光的
0–255 线性显示，另附严格 0/255 二值图：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.simulation_agreement `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40 `
  --export-simulation-only `
  --device auto `
  --batch-size 4
```

结果在 `quick40\simulation_reference_monochrome`。灰度 PNG 只用于查看和按文件名配对；
采集后的正式 PCC/SSIM 仍由原始 CCD 强度和 float 仿真重新计算，不使用显示 PNG
代替原始数据。

先只读检查设备，然后采集：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase validate `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40

python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase acquire `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40 `
  --clear-output
```

计算 quick40 诊断成功率和仿真—CCD 相似度：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase evaluate `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40 `
  --allow-quick40-diagnostic

python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase agreement `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40 `
  --device auto `
  --batch-size 4
```

quick40 只用于对齐、曝光和选 mask，不能作为论文准确率。可把上述 mask 名和输出目录
依次换成另外三个候选，多加载几张 mask 比较。

### 5.2 正式 formal400

选好 mask 后生成 400 张固定随机样本；下面仍以 `post_robust_best` 为例：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_session `
  --profile formal400 `
  --mask post_robust_best `
  --output-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\formal400
```

加载 `formal400\phase_to_play` 中唯一 BMP，然后依次运行：

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase validate `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\formal400

python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase acquire `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\formal400 `
  --clear-output

python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase evaluate `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\formal400

python -m experiments.d2nn_mnist4_single_layer_17um_10cm_v2.lab_pipeline `
  --phase agreement `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\formal400 `
  --device auto `
  --batch-size 4
```

正式输出：

```text
formal400\hardware_evaluation\hardware_metrics_raw.json
formal400\hardware_evaluation\hardware_predictions_raw.csv
formal400\hardware_evaluation\paper_evaluation\figures
formal400\simulation_agreement\agreement_summary.json
formal400\simulation_agreement\per_sample_agreement.csv
formal400\simulation_agreement\figures
formal400\simulation_agreement\measured_grayscale_8bit
formal400\simulation_agreement\simulation_grayscale_8bit
formal400\simulation_agreement\measured_binary_8bit
formal400\simulation_agreement\simulation_binary_8bit
```

相似度报告包含 PCC、signal-PCC、SSIM、shape-NRMSE、余弦相似度、能量比、质心误差、
理论信号区外能量比例、饱和率、仿真/实测预测一致率。相似度中的形状归一化只用于分析，
不会进入 MNIST 分类。

## 6. Qwen 仿真—实测光场一致性

完整包已经在每个 `amplitude_to_play` 中提供与 BMP 哈希绑定的
`reconstruction_manifest.csv`；不要删除该文件，也不需要再次运行
`reconstruct_slm`。

相位 SLM 加载 `agree\04_language_global\phase_to_play` 中唯一 BMP：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\agree\04_language_global `
  --validate-only

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\agree\04_language_global `
  --clear-output

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate `
  --session-dir experiments\lab_qwen\agree `
  --stages language_global

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_report `
  --evaluation-dir experiments\lab_qwen\agree\agreement_evaluation `
  --output-dir experiments\lab_qwen\results\agreement
```

### 6.1 形状输入 × 形状相位 mask：独立仿真—光路一致性

这一步不依赖 Qwen checkpoint，也不评价分类准确率。它固定使用 532 nm、17 µm
逻辑像素、518×518 传播画布、478×478 有效区、10 cm 和 0.65° k 空间截止，生成
6 个非对称振幅形状 × 6 个几何相位 mask，共 36 个采集。先生成完整会话：

```powershell
python -m experiments.lab_qwen.shape_agreement generate `
  --output-dir experiments\lab_qwen\shape_agreement
```

如果相位 SLM 中心不再是 `[980,590]`，在生成时明确传入实测中心：

```powershell
python -m experiments.lab_qwen.shape_agreement generate `
  --output-dir experiments\lab_qwen\shape_agreement `
  --phase-center-x 980 `
  --phase-center-y 590
```

生成后只按下面这一个新文件中的 6 组命令顺序操作；每组先手动加载该目录唯一的
相位 BMP，再由程序连续播放 6 张振幅 BMP：

```text
experiments\lab_qwen\shape_agreement\RUN_COMMANDS.md
```

36 帧全部采集完成后运行：

```powershell
python -m experiments.lab_qwen.shape_agreement evaluate `
  --session-dir experiments\lab_qwen\shape_agreement
```

正式结果在：

```text
experiments\lab_qwen\shape_agreement\shape_agreement_results\shape_agreement_summary.json
experiments\lab_qwen\shape_agreement\shape_agreement_results\metrics_per_pair.csv
experiments\lab_qwen\shape_agreement\shape_agreement_results\metrics_summary_by_phase.csv
experiments\lab_qwen\shape_agreement\shape_agreement_results\figures
```

主指标为 transport-quantized 仿真参考、线性强度域、固定 canonical 方向下的 PCC、
signal-PCC、SSIM、shape-NRMSE、余弦相似度、质心误差、能量比例、理论信号区外能量
和饱和率。程序不做背景扣除、逐帧 min-max、逐帧配准或自动挑选翻转方向。
`best_orientation_diagnostic` 只负责提示四点角标是否填反，不能代替正式主指标。

不要在已有实测 CCD 的目录上使用 `--overwrite`；该参数会重建整个形状会话。

## 7. Qwen 最后一层 quick210 快速验证

相位 SLM 加载 `last\04_language_global\phase_to_play` 中唯一 BMP：

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

## 8. Qwen 四层逐层采集与微调

固定顺序：

```text
01_vision_expert -> 02_vision_global -> 03_language_expert -> 04_language_global
```

每一层都执行同一闭环：实验室采当前层 → 上传当前层 CCD 与日志 → 服务器微调 →
服务器导出下一层 → 下载下一层 → 再采集。不能预先同时采完四层，因为下一层输入依赖
上一层实测 CCD。

### 8.1 第一层 vision_expert

加载 `four\01_vision_expert\phase_to_play` 中唯一 BMP：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four\01_vision_expert --clear-output
scp -P 24096 -r experiments\lab_qwen\four\01_vision_expert\ccd_captured guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/01_vision_expert/
scp -P 24096 -r experiments\lab_qwen\four\01_vision_expert\acquisition_logs guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/01_vision_expert/
```

服务器仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage vision_expert --phase finetune --epochs 20 --upstream-source measured
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage vision_global --phase export --upstream-source measured
```

下载第二层：

```powershell
scp -P 24096 -r guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/02_vision_global experiments\lab_qwen\four\
```

### 8.2 第二层 vision_global

加载第二层唯一相位 BMP：

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

```powershell
scp -P 24096 -r guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/03_language_expert experiments\lab_qwen\four\
```

### 8.3 第三层 language_expert

加载第三层唯一相位 BMP：

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

```powershell
scp -P 24096 -r guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/04_language_global experiments\lab_qwen\four\
```

### 8.4 第四层 language_global

加载第四层唯一相位 BMP：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four\04_language_global --clear-output
scp -P 24096 -r experiments\lab_qwen\four\04_language_global\ccd_captured guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/04_language_global/
scp -P 24096 -r experiments\lab_qwen\four\04_language_global\acquisition_logs guest3@202.120.62.181:/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/04_language_global/
```

服务器最终微调：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_hardware_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_language_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1 --stage language_global --phase finetune --epochs 20 --upstream-source measured
```

最终 checkpoint：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four210_run1/checkpoints/after_language_global.pt
```

## 9. 明确不再使用的旧流程

- 不手填 `camera.device_roi_xywh`。
- 不手填 contract 路径或 SHA，也不运行 `Get-FileHash`。
- 不运行 `detector_homography fit/apply`。
- 不准备未命名的 `raw_roi.npy` 或 `rectified_478.tif`。
- 不使用 Holoeye、旧振幅 SLM、旧相机或旧 `lab_hardware_config.yaml`。
- 不对已经透视校正的 CCD 再做左右/上下翻转；逻辑镜像由四点 homography 一次解决。
