# Temporal-16 均衡模型实验包（先读本文）

本 ZIP 是一个可独立解压运行的 Temporal quality 实验工程，固定对应 **16 帧、4×4 并行、54×54 专家、Top-2 光路由**。包内含完整 2250 train + 558 test 冻结 Qwen 前端缓存、正式权重、六张相位 SLM BMP、标定图案、Meadowlark/TUCam 控制代码以及逐阶段微调代码。

不得把本包的 checkpoint、缓存、BMP 或 session 与 9 帧、36 帧、Spatial 包混用。每个样本的一张 1024×1024 振幅 BMP 已经同时承载 16 帧，不是每个样本播放 16 次。

## 1. 已锁定结果与硬件合同

- 正常光电：SRCC 0.8374、PLCC 0.8612、KRCC 0.6278、RMSE 7.043、MAE 5.418。
- 同权重屏蔽光路：SRCC 0.5087。
- 视觉专家测试选择占比：29.56% / 24.94% / 23.54% / 21.95%。
- 序列专家测试选择占比：25.36% / 28.23% / 23.57% / 22.85%。
- 532 nm，10 cm，17 µm 逻辑像素，518×518 画布，中央 478×478 有效区。
- 振幅 SLM：1024×1024、17 µm、8-bit，`255=亮/透光`。
- 相位 SLM：1920×1200、8 µm；默认中心 `(980,590)`，按原合同只做纵向翻转。
- 六次传播：视觉 router、视觉 expert、视觉 global、序列 router、序列 expert、序列 global。
- 冻结 Qwen 前端已离线缓存；实验电脑推理和微调不加载 Transformer/Attention。

## 2. 解压后的唯一入口

所有命令都从 ZIP 解压后的工程根目录运行。PowerShell：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -m pip install -r experiments\hardware_sdk\requirements-light.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若 `VERIFY_BUNDLE.py` 不通过，不要继续。之后只编辑：

```text
experiments\lab_lgvq\LAB_CONFIG.yaml
```

第一次必须保留包内原厂 LUT `slm7930_at532-70c-pixel-2.lut`。新实验台应按自身入射角、光强和曝光重新标定 LUT，不要直接复制另一台实验台生成的线性化 LUT。

## 3. 标定顺序

1. 使用 `experiments\lab_lgvq\calib\dual_slm_k1` 完成两个 SLM 的 k=1 对齐。
2. 振幅加载 Fresnel 目录的 `A_WHITE.bmp`，相位依次加载 `P1_POINT.bmp`、`P4_POINT.bmp`、`P9_POINT.bmp`，确认 10 cm 和方向；P4 测四个逻辑角点。
3. 在 `LAB_CONFIG.yaml` 填 LUT 文件名/可选 SHA、曝光、相位中心和翻转、四个 CCD 全传感器逻辑角点。
4. 生成硬件合同并检查曝光：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

如需在师姐自己的台架重新标 LUT：

```powershell
python -m experiments.hardware_sdk.workflows.amplitude_lut_calibration all `
  --config experiments\lab_lgvq\generated\formal_hardware.yaml
```

只有验证报告 `recommended_for_use=true` 后，才把 `LAB_CONFIG.yaml` 的文件名和 SHA 改成新 LUT，再运行一次 `prepare_lab`。标 LUT 时的曝光必须避免饱和；正式曝光可以另行选择，但改变光强/曝光后必须重做曝光检查。

## 4. 固定变量与完整性检查

```powershell
$M = 'experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.hardware_bridge'
$C = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54\configs\deployment\temporal16_balanced_lab.yaml'
$K = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54\deployment\checkpoints\best_observed_test_checkpoint.pt'
$S = 'experiments\lab_lgvq\sessions\temporal16_balanced'
$H = 'experiments\lab_lgvq\generated\formal_hardware.yaml'

python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 `
  --config $C --phase preflight
```

正式全数据实验时，下文每一条 `export-pass` 都必须带 `--all-data`。快速联调时则全部去掉 `--all-data`，默认固定为 64 train + 32 test。第一次导出后 session 会封存样本集合，中途切换规模会被拒绝。

## 5. 六次采集与四次逐层微调

以下顺序不能跳跃。每次 `acquire_folder --clear-output` 只会清除对应 stage 旧 CCD；确认需要重拍后再使用。

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

最终指标和逐视频预测位于 `$S\final_evaluation`。每次微调均每 5 epoch 测试，并保存最高 test SRCC；本实验按项目约定不使用 validation。

## 6. CCD 与方向约束

`ccd_captured` 是硬件 ROI 原始强度经四点单应性矫正到 canonical 478×478 后的结果，不做逐张 min-max、背景扣除或显示用 log。模型内部对仿真与实测使用同一套：非负截断、单帧均值归一化、相对强度截断、`log1p`。相位 BMP 已按 `LAB_CONFIG.yaml` 做硬件方向变换，播放器不得再次翻转。

仿真报告、六级光场 PNG/PDF、原始强度统计和速度/均衡图位于项目的 `deployment\evidence`；六张正式相位 BMP 位于 `deployment\hardware_masks\phase_slm_1920x1200`。
