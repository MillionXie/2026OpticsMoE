# 新实验室完整流程：唯一命令文档

本文件按实际执行顺序书写。所有 PowerShell 命令均在统一仓库根目录
`E:\code\guest\2026OpticsMoE` 执行。实验人员只编辑
`experiments\lab_qwen\LAB_CONFIG.yaml`，不要编辑 `generated` 目录，也不需要手算
ROI、透视变换或 SHA-256。

本包的 Qwen 主模型固定为强噪声续训版本：训练时在每层仿真 CCD 强度上加入相对
干净单帧均值的截断偏置高斯噪声（均值 6%、标准差 5%、截断到 -4%～16%），光学
融合系数下限为 1%。正式 EMA 检查点 SHA-256 为
`39bd00bd0f5a8d01f99c65dd4566b9e602a68bcc8a1660f27b6829fad9d4a2e1`，固定测试
Top-1/Top-3 为 82.0%/93.5%；测试结果没有参与 checkpoint 选择。

## 0. 解压与环境

把完整 ZIP 解压后，先执行：

```powershell
Set-Location E:\code\guest\2026OpticsMoE
conda activate xml
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-lab.txt
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-offline-finetune.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若要把四层逐层微调也放到实验室 RTX 5060 上，再安装完整本地微调依赖。此文件不含
`torch`，不会覆盖已经可用的 CUDA PyTorch：

```powershell
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-local-four-stage.txt
```

确认下面两个实验室专用文件确实存在：

```text
experiments\hardware_sdk\vendor_sdk\amplitude_meadowlark\LUT Files\slm7930_at532_70C.lut
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

包内同时提供厂商 30°C/70°C LUT 和本次设备正在使用的
`slm7930_at532-70c-pixel-2.lut`。当前默认使用 `pixel-2`；若切换设备或温度，必须先
换回该设备对应的原始 LUT，并重新做下面的 LUT 标定，不能沿用另一台设备生成的结果。

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

当前 P1/P4/P9 均使用老师 MATLAB 的方窗二次菲涅尔公式；不再生成或使用旧
`CROSS.bmp`。振幅始终为全白。P4 四个焦点对应 478×17 µm 有效场在 8 µm
相位面上的四个顶点，顶点间距为 1015.75 个相位像素。

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

### 4.1 用密集 CCD 响应重新标定振幅 LUT

普通 32 点曲线用于快速检查曝光，不直接生成 LUT。当前 `pixel-2` 曲线在灰度约 80
附近出现真正暗态，0→80 与 80→255 属于两条不同的光强调制支路；把整条 U 形曲线
直接做普通三次样条会产生错误的多值反函数。本工具默认采集 64 个灰度、每灰度 3 帧，
自动选择“暗态→较亮端点”中动态范围更大的单调支路，先用 PAVA 保序拟合抑制小噪声，
再做分段线性反插值，最终仍生成厂商可加载的 256 行 `gray DAC` LUT。

网络输入 BMP 表示光场振幅，因此默认：

```text
目标场振幅 A = gray / 255
目标 CCD 强度 I = A²
```

不要为了让图上的 CCD 强度成为直线而误用线性强度 LUT；只有做单独对照时才把
`target_transfer` 改成 `linear_intensity`。

先确认 `LAB_CONFIG.yaml` 当前选择的是要作为基准的旧 LUT：

```yaml
amplitude_lut_filename: slm7930_at532-70c-pixel-2.lut
amplitude_lut_calibration:
  gray_point_count: 64
  frames_per_gray: 3
  target_transfer: field_amplitude
  output_lut_filename: slm7930_at532-70c-pixel-2_linearized-amplitude.lut
```

运行 `prepare_lab` 后，在相位 SLM 上固定加载并始终保持：

```text
experiments\lab_qwen\calib\exposure\phase\phase_zero.bmp
```

相位 WFC/额外相位修正必须关闭，自动曝光必须关闭。然后一条命令完成旧 LUT 密集扫描、
新 LUT 拟合、新 LUT 重新加载和第二次密集验证：

```powershell
python -m experiments.hardware_sdk.workflows.amplitude_lut_calibration all `
  --config experiments\lab_qwen\generated\formal_hardware.yaml
```

默认共采集 `64×3×2=384` 帧。程序会在两次硬件扫描前显示当前 LUT 和相位零图确认，
按提示确认即可。不要加 `--yes` 跳过第一次正式设备确认。

结果位于：

```text
experiments\lab_qwen\results\lut_calibration\slm7930_at532-70c-pixel-2_linearized-amplitude\
  base_scan\slm_response.csv
  verification_scan\slm_response.csv
  lut_mapping.csv
  lut_fit_report.json
  final_lut_report.json
  lut_calibration.png
  lut_calibration.svg
```

新 LUT 位于：

```text
experiments\hardware_sdk\vendor_sdk\amplitude_meadowlark\LUT Files\slm7930_at532-70c-pixel-2_linearized-amplitude.lut
```

拟合中的 `(I-I_dark)/(I_bright-I_dark)` 只用于建立 LUT 反函数和验证误差，不会加入
Qwen/MNIST 的 CCD 后处理；正式网络帧仍按原有固定范围保存和读取。

程序绝不会覆盖旧 LUT。只有 `final_lut_report.json` 中
`recommended_for_use=true` 时，才把 `LAB_CONFIG.yaml` 的
`amplitude_lut_filename` 改为新文件名并重新运行 `prepare_lab`。如果验证未通过，继续
使用 `slm7930_at532-70c-pixel-2.lut`，先检查曝光、偏振器、相位零图和光路稳定性。

若硬件扫描已经完成但拟合阶段中断，可分别恢复：

```powershell
python -m experiments.hardware_sdk.workflows.amplitude_lut_calibration fit `
  --config experiments\lab_qwen\generated\formal_hardware.yaml

python -m experiments.hardware_sdk.workflows.amplitude_lut_calibration verify `
  --config experiments\lab_qwen\generated\formal_hardware.yaml
```

确实需要重做并替换同名“生成 LUT”时，显式增加 `--overwrite-generated-lut`；该参数也
不会覆盖当前基准旧 LUT。

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

### 5.2 先做 20 帧时序与方向诊断

在正式采集 40/400 张之前，建议先运行一次。它固定选择数字 0、1、2、3 各一张，按
`0/50/100/200/400 ms` 五档等待依次播放，因此总共采集 20 帧。每个数字的 400 ms
结果作为参考，用 PCC、增益对齐 NMAE 和平均亮度比例判断较短等待是否已经稳定：

```powershell
python -m experiments.lab_qwen.mnist_timing_diagnostic `
  --phase all `
  --stage-dir experiments\lab_qwen\mnist4_sessions\post_robust_best\quick40 `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --clear-output
```

这条命令仍会要求确认相位 SLM 上显示的是该 quick40 目录中的唯一相位 BMP。结果在：

```text
quick40\timing_diagnostic\timing_summary.json
quick40\timing_diagnostic\timing_metrics_per_capture.csv
quick40\timing_diagnostic\timing_summary.png
quick40\timing_diagnostic\mnist4_detector_regions_overlay.png
```

重点查看 `recommended_formal_slm_settle_delay_ms`，再把它填回
`LAB_CONFIG.yaml -> capture_timing.formal_slm_settle_delay_ms`，并重新运行
`python -m experiments.lab_qwen.prepare_lab`。`mnist4_detector_regions_overlay.png`
是在四点透视校正后的 478×478 CCD 上画出的四个真实 59×59 判别区，并标出
`CANONICAL TOP/LEFT/RIGHT/BOTTOM`；它只画框，不改变任何分类像素值。若预测亮区与
数字标签对应但画面方向文字不符合物理预期，应回到四个逻辑角点的命名检查，而不要在
后处理里再随意翻转。

需要理解的时序参数只有下面四类：

- `formal_slm_settle_delay_ms`：最主要的等待；从 `ImageWriteComplete` 返回到开始 CCD
  采集调用之间的时间。
- `discard_frames_after_display`：每次换图后丢弃 CCD 连续流中的旧帧，默认 1。它也会
  增加实际耗时，但不是固定毫秒 sleep。
- `camera_warmup_frames`：相机打开时只丢一次的预热帧，默认 3，不影响逐图设置。
- `camera_exposure_us`：曝光积分时间。曝光过高看饱和比例，过低看 `p99`；它不是 SLM
  稳定等待。

时序诊断的 `timing_capture_manifest.csv` 会分别记录 SLM 写入完成、实际 sleep、CCD
丢帧/采集/透视校正/保存和整周期耗时。时序诊断不会对正式 MNIST 分类增加归一化、
背景扣除或非线性；增益对齐只用于比较同一个数字在不同等待时间下是否稳定。

### 5.3 正式 formal400

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

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1.agreement_evaluate `
  --session-dir experiments\lab_qwen\agree `
  --stages language_global

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1.agreement_report `
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

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1.offline_quick_finetune `
  --session-dir experiments\lab_qwen\last `
  --device auto `
  --epochs 100 `
  --selection-policy development `
  --development-per-class 2 `
  --early-stopping-patience 15

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1.result_report `
  --root experiments\lab_qwen `
  --session-dir experiments\lab_qwen\last `
  --output-dir experiments\lab_qwen\results\last
```

## 8. Qwen 四层逐层采集与微调

本次正式全数据工程固定在：

```powershell
Set-Location E:\code\guest\qwen_mnist4_early_robust_full_data_lab
conda activate xml
```

正式 profile 固定为 `accuracy_first_full`，阶段目录固定为
`experiments\lab_qwen\four_accuracy_first_full`。旧的 `four` 目录只保留作
210 帧快速诊断，不得与正式 CCD、checkpoint 混用。

固定顺序：

```text
01_vision_expert -> 02_vision_global -> 03_language_expert -> 04_language_global
```

每一层都在实验室电脑完成同一闭环：采当前层 → 本地微调 → 本地导出并重建下一层
BMP → 再采集。最终离线包已经包含 Qwen3-VL-Embedding-2B 和 Caltech101；程序强制
离线加载，模型缺失或不完整时直接报错，不会访问 Hugging Face。不能预先同时采完
四层，因为下一层输入依赖上一层实测 CCD。

本地微调最多跑 100 epoch。每类固定保留 3 张 gallery、20 张 sealed test，其余数据
用于训练及固定 development 选模。checkpoint 按 `development Top-1` 选择；Top-1
相同时选 development CE 更低者。sealed test 不参与选模，只在恢复最佳 checkpoint
后评估一次。默认 patience=15，若连续 15 epoch 没有改进会提前停止。RTX 5060 默认
3 类×2 图，推理 batch=2；若显存仍不足，把命令附加
`--pk-classes 2 --inference-batch-size 1`。

每层先确认相位 SLM 已加载该层 `phase_to_play` 中唯一 BMP，再执行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four_accuracy_first_full\01_vision_expert --clear-output
python -m experiments.lab_qwen.local_four_stage --profile accuracy_first_full --stage vision_expert --epochs 100

python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four_accuracy_first_full\02_vision_global --clear-output
python -m experiments.lab_qwen.local_four_stage --profile accuracy_first_full --stage vision_global --epochs 100

python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four_accuracy_first_full\03_language_expert --clear-output
python -m experiments.lab_qwen.local_four_stage --profile accuracy_first_full --stage language_expert --epochs 100

python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four_accuracy_first_full\04_language_global --clear-output
python -m experiments.lab_qwen.local_four_stage --profile accuracy_first_full --stage language_global --epochs 100
```

每次 `local_four_stage` 都保存当前层最佳 checkpoint，并自动导出、重建下一层。每层
结果在该层的 `finetune_metrics.json` 和 `finetune_train_log.csv`。最终 checkpoint：

```text
experiments\lab_qwen\four_accuracy_first_full\checkpoints\after_language_global.pt
```

新版 `acquire_folder --stage-dir` 检测到只有 `compact_amplitude` 时会自动执行
17 μm 1:1 重建；一般不需要再手工运行 `reconstruct_slm`。

## 9. 明确不再使用的旧流程

- 不手填 `camera.device_roi_xywh`。
- 不手填 contract 路径或 SHA，也不运行 `Get-FileHash`。
- 不运行 `detector_homography fit/apply`。
- 不准备未命名的 `raw_roi.npy` 或 `rectified_478.tif`。
- 不使用 Holoeye、旧振幅 SLM、旧相机或旧 `lab_hardware_config.yaml`。
- 不对已经透视校正的 CCD 再做左右/上下翻转；逻辑镜像由四点 homography 一次解决。
