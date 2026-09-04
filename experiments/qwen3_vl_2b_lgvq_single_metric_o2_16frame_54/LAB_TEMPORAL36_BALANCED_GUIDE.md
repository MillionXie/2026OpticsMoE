# Temporal-36 均衡模型实验包（先读本文）

本 ZIP 是一个可独立解压运行的 Temporal quality 实验工程，固定对应 **36 帧、6×6 并行、37×37 专家、Top-2 光路由**。包内含完整 2250 train + 558 test 冻结 Qwen 前端缓存、正式权重、六张相位 SLM BMP、标定图案、Meadowlark/TUCam 控制代码以及逐阶段微调代码。

不得把本包的 checkpoint、缓存、BMP 或 session 与 9 帧、16 帧、Spatial 包混用。每个样本的一张 1024×1024 振幅 BMP 已经同时承载 36 帧，不是每个样本播放 36 次。

## 1. 已锁定结果与硬件合同

- 正常光电：SRCC 0.8454、PLCC 0.8650、KRCC 0.6394、RMSE 7.183、MAE 5.451。
- 同权重屏蔽光路：SRCC 0.2333。
- 视觉专家测试选择占比：26.30% / 29.99% / 19.91% / 23.80%。
- 序列专家测试选择占比：25.99% / 24.55% / 25.27% / 24.19%。
- 532 nm，10 cm，17 µm 逻辑像素，518×518 画布，中央 478×478 有效区。
- 36 个 `77×77` lane 以 pitch 79 排成 6×6；每 lane 内四个 `37×37` 专家，专家 pitch 40、缝隙 3。总跨度 472，未扩大 ROI。
- 振幅 SLM：1024×1024、17 µm、8-bit，`255=亮/透光`。
- 相位 SLM：1920×1200、8 µm；默认中心 `(980,590)`，按原合同只做纵向翻转。
- 六次传播：视觉 router、视觉 expert、视觉 global、序列 router、序列 expert、序列 global。
- 冻结 Qwen 前端已离线缓存；实验电脑推理和微调不加载 Transformer/Attention。

## 2. 解压、环境、唯一配置

所有命令都从 ZIP 解压后的工程根目录运行：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -m pip install -r experiments\hardware_sdk\requirements-light.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若完整性检查失败，不要继续。实验人员只编辑：

```text
experiments\lab_lgvq\LAB_CONFIG.yaml
```

第一次必须使用包内原厂 LUT `slm7930_at532-70c-pixel-2.lut`。师姐的台架应按自己的入射角、光强和曝光标定新 LUT；其他台架生成的线性化 LUT 不能直接视为等价。

## 3. 标定顺序

1. 使用 `experiments\lab_lgvq\calib\dual_slm_k1` 完成两个 SLM 的 k=1 对齐。
2. 振幅加载 Fresnel 目录的 `A_WHITE.bmp`，相位依次加载 `P1_POINT.bmp`、`P4_POINT.bmp`、`P9_POINT.bmp`，确认 10 cm 和方向；P4 测四个逻辑角点。
3. 在 `LAB_CONFIG.yaml` 填 LUT 文件名/可选 SHA、曝光、相位中心和翻转、四个 CCD 全传感器逻辑角点。
4. 执行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

需要重标 LUT 时：

```powershell
python -m experiments.hardware_sdk.workflows.amplitude_lut_calibration all `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

只有验证报告 `recommended_for_use=true` 后，才切换 `LAB_CONFIG.yaml` 中的 LUT 文件名和 SHA，并重新运行 `prepare_lab`。

## 4. 固定变量

```powershell
$M = 'experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.hardware_bridge'
$C = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54\configs\deployment\temporal36_balanced_lab.yaml'
$K = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54\deployment\checkpoints\best_observed_test_checkpoint.pt'
$S = 'experiments\lab_lgvq\sessions\temporal36_balanced'
$H = 'experiments\lab_lgvq\generated\formal_hardware.yaml'

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 `
  --config $C --phase preflight
```

正式实验时，下文每一条 `export-pass` 都保留 `--all-data`。快速联调时则全部删掉 `--all-data`，固定使用默认 64 train + 32 test。第一次导出会封存 session 样本集合，中途改变规模会被拒绝。

## 5. 六次采集与四次逐层微调

### A. 视觉 router 与视觉 expert

```powershell
python -m $M export-pass --config $C --checkpoint $K --session-dir $S `
  --optical-pass vision_router --all-data --batch-size 16 --device cuda
# 手动确认 $S\01_vision_router\phase_to_play\vision_router.bmp 已显示
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H `
  --stage-dir "$S\01_vision_router" --clear-output
python -m $M validate-capture --config $C --checkpoint $K --session-dir $S `
  --optical-pass vision_router

python -m $M export-pass --config $C --checkpoint $K --session-dir $S `
  --optical-pass vision_expert --all-data --batch-size 16 --device cuda
# 手动确认 $S\02_vision_expert\phase_to_play\vision_expert.bmp 已显示
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H `
  --stage-dir "$S\02_vision_expert" --clear-output
python -m $M validate-capture --config $C --checkpoint $K --session-dir $S `
  --optical-pass vision_expert

python -m $M finetune --config $C --checkpoint $K --session-dir $S `
  --stage vision_expert --epochs 100 --test-interval 5 --batch-size 16 --device cuda
$K = "$S\checkpoints\after_vision_expert_best_test.pt"
```

### B. 视觉 global

```powershell
python -m $M export-pass --config $C --checkpoint $K --session-dir $S `
  --optical-pass vision_global --all-data --batch-size 16 --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H `
  --stage-dir "$S\03_vision_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $K --session-dir $S `
  --optical-pass vision_global
python -m $M finetune --config $C --checkpoint $K --session-dir $S `
  --stage vision_global --epochs 100 --test-interval 5 --batch-size 16 --device cuda
$K = "$S\checkpoints\after_vision_global_best_test.pt"
```

### C. 序列 router 与序列 expert

```powershell
python -m $M export-pass --config $C --checkpoint $K --session-dir $S `
  --optical-pass language_router --all-data --batch-size 16 --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H `
  --stage-dir "$S\04_language_router" --clear-output
python -m $M validate-capture --config $C --checkpoint $K --session-dir $S `
  --optical-pass language_router

python -m $M export-pass --config $C --checkpoint $K --session-dir $S `
  --optical-pass language_expert --all-data --batch-size 16 --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H `
  --stage-dir "$S\05_language_expert" --clear-output
python -m $M validate-capture --config $C --checkpoint $K --session-dir $S `
  --optical-pass language_expert

python -m $M finetune --config $C --checkpoint $K --session-dir $S `
  --stage language_expert --epochs 100 --test-interval 5 --batch-size 16 --device cuda
$K = "$S\checkpoints\after_language_expert_best_test.pt"
```

### D. 序列 global 与最终评估

```powershell
python -m $M export-pass --config $C --checkpoint $K --session-dir $S `
  --optical-pass language_global --all-data --batch-size 16 --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H `
  --stage-dir "$S\06_language_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $K --session-dir $S `
  --optical-pass language_global

python -m $M finetune --config $C --checkpoint $K --session-dir $S `
  --stage language_global --epochs 100 --test-interval 5 --batch-size 16 --device cuda
$K = "$S\checkpoints\after_language_global_best_test.pt"
python -m $M evaluate --config $C --checkpoint $K --session-dir $S `
  --stage language_global --batch-size 16 --device cuda
```

最终指标和逐视频预测位于 `$S\final_evaluation`。每次微调每 5 epoch 测试，并按最高 test SRCC 保存；本项目不划 validation。

## 6. 不可修改的处理

`ccd_captured` 是硬件 ROI 原始强度经四点单应性矫正到 canonical 478×478，不做逐张 min-max、背景扣除或显示用 log。模型内部对仿真与实测统一执行非负截断、单帧均值归一化、相对强度截断和 `log1p`。相位 BMP 已按配置做硬件方向变换，播放器不得二次翻转。

项目的 `deployment\evidence` 包含正式仿真指标、六级光场 PNG/PDF、原始强度统计和速度/专家均衡图；六张正式相位 BMP 位于 `deployment\hardware_masks\phase_slm_1920x1200`。
