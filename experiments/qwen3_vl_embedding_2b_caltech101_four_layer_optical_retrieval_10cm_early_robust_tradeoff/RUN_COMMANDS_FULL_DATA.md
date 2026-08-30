# 全数据实验命令

所有命令均在新实验包根目录执行。正式 profile 固定为 `accuracy_first_full`，目录固定为 `four_accuracy_first_full`。

## 0. 只需首次执行

```powershell
conda activate xml
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
