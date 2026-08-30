# 全数据实验命令

所有命令均在下列实验包根目录执行。正式 profile 固定为
`accuracy_first_full`，目录固定为 `four_accuracy_first_full`。

## 0. 仿真 CCD feature 是什么

不能用 phase preview 代替 CCD feature。服务器应从同一 checkpoint 导出四层的：

- `theoretical_ccd/ideal_model_fp32/*.npz`：未量化 phase/振幅的原始线性强度；
- `theoretical_ccd/transport_quantized/*.npz`：按实际 8-bit BMP 量化后重新仿真的线性强度；
- `ccd_feature_visualization/*/network_input_224/*.npz`：网络实际读取的
  非负截断、单帧均值归一化、相对强度截断、log1p、224×224 pooling 后 feature；
- 同名 PNG 和 CONTACT_SHEET：只用于观看，不能作为数值评估输入。

服务器导出命令：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff.agreement_export \
  --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/configs/hardware/accuracy_first_full_agreement.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/runs/accuracy_first_floor0p1_leak0to5/ema_best_train_loss_checkpoint.pt \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/hardware_sessions/accuracy_first_full_theoretical_ccd \
  --stages vision_expert vision_global language_expert language_global \
  --upstream-source simulation

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff.ccd_feature_gallery \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/hardware_sessions/accuracy_first_full_theoretical_ccd
```

实验室采集同一目录内 `amplitude_to_play` 与 `phase_to_play` 后，使用
`agreement_evaluate` 计算 linear 和 network_input 两个域的 PCC、SSIM、shape NRMSE、
能量比、质心误差和饱和率。禁止用显示 PNG 计算正式指标。

```powershell
Set-Location E:\code\guest\qwen_mnist4_early_robust_full_data_lab
conda activate xml
```

不要在 `2026OpticsMoE` 或旧实验包中混跑这些命令。

## 0. 只需首次执行

```powershell
python -m experiments.lab_qwen.prepare_lab
```

检查：3500 μs、新线性 LUT、四点 homography、478×478 输出均正确。

## 1. 第一层 vision_expert

手动加载：

`experiments\lab_qwen\four_accuracy_first_full\01_vision_expert\phase_to_play\vision_expert.bmp`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four_accuracy_first_full\01_vision_expert `
  --clear-output

python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage vision_expert `
  --epochs 100
```

## 2. 第二层 vision_global

手动加载 `02_vision_global\phase_to_play\vision_global.bmp`，然后：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four_accuracy_first_full\02_vision_global `
  --clear-output

python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage vision_global `
  --epochs 100
```

## 3. 第三层 language_expert

手动加载 `03_language_expert\phase_to_play\language_expert.bmp`，然后：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four_accuracy_first_full\03_language_expert `
  --clear-output

python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage language_expert `
  --epochs 100
```

## 4. 第四层 language_global

手动加载 `04_language_global\phase_to_play\language_global.bmp`，然后：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four_accuracy_first_full\04_language_global `
  --clear-output

python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage language_global `
  --epochs 100
```

每层采集前都必须确认相位 SLM 显示的是该层 BMP。`--clear-output` 会清空本层旧 CCD 输出，确认目录无重要旧数据再执行。
