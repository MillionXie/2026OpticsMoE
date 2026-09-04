# Spatial-4 光电视频质量评价：实验室完整流程

本包只对应 LGVQ 的 **Spatial quality**。输入提示词固定为：

> Please evaluate the spatial quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.

模型使用 4 帧、2×2 帧并行、109×109 光学专家、光学 Top-2 router、532 nm、17 μm、10 cm；推理网络没有 Transformer/Attention，也没有电子 router。Qwen3-VL 的冻结图像 patch/position embedding、文本 tokenizer/embedding、14 个固定质量通道和 VGG 前端均已缓存，因此实验室离线运行时不下载也不加载完整 Qwen。

## 0. 解压与完整性检查

在 ZIP 解压根目录打开 PowerShell：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

$P = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
$M = 'experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.hardware_bridge'
$C = "$P\configs\deployment\spatial4_lab.yaml"
$W0 = "$P\deployment\checkpoints\best_observed_test_checkpoint.pt"
$S = 'experiments\lab_lgvq\sessions\spatial4_formal01'
$H = 'experiments\lab_lgvq\generated\formal_hardware.yaml'
```

## 1. 只编辑一个实验台配置并完成标定

只编辑 `experiments\lab_lgvq\LAB_CONFIG.yaml`：填写 Meadowlark LUT 文件名和 SHA256、曝光、相位 SLM 中心/原代码规定的翻转关系，以及 CCD 全传感器上的四个逻辑角点。之后执行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config $H
```

正式保存的是单应性矫正后的 478×478 原始强度；文件阶段不做逐帧 min-max、背景扣除或 log。仿真 CCD 与实测 CCD 进入网络时共用同一套非负截断、单帧均值归一化、相对强度截断和 `log1p`。

## 2. 六个光学 pass、四次本地微调

每个 pass 都必须依次执行“导出 → 相位 SLM 手动加载该目录唯一 BMP → 振幅 SLM/CCD 采集 → 校验”。后一个 pass 必须从前一阶段微调后的 checkpoint 重新导出。第一次走通可去掉 `--all-data`；正式实验保留 `--all-data`（2250 train + 558 test）。

### A. 视觉 router 与视觉专家

```powershell
python -m $M export-pass --config $C --checkpoint $W0 --session-dir $S --optical-pass vision_router --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\01_vision_router" --clear-output
python -m $M validate-capture --config $C --checkpoint $W0 --session-dir $S --optical-pass vision_router

python -m $M export-pass --config $C --checkpoint $W0 --session-dir $S --optical-pass vision_expert --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\02_vision_expert" --clear-output
python -m $M validate-capture --config $C --checkpoint $W0 --session-dir $S --optical-pass vision_expert

python -m $M finetune --config $C --checkpoint $W0 --session-dir $S --stage vision_expert --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W1 = "$S\checkpoints\after_vision_expert_best_test.pt"
```

### B. 视觉全局层

```powershell
python -m $M export-pass --config $C --checkpoint $W1 --session-dir $S --optical-pass vision_global --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\03_vision_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $W1 --session-dir $S --optical-pass vision_global
python -m $M finetune --config $C --checkpoint $W1 --session-dir $S --stage vision_global --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W2 = "$S\checkpoints\after_vision_global_best_test.pt"
```

### C. 序列 router 与序列专家

```powershell
python -m $M export-pass --config $C --checkpoint $W2 --session-dir $S --optical-pass language_router --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\04_language_router" --clear-output
python -m $M validate-capture --config $C --checkpoint $W2 --session-dir $S --optical-pass language_router

python -m $M export-pass --config $C --checkpoint $W2 --session-dir $S --optical-pass language_expert --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\05_language_expert" --clear-output
python -m $M validate-capture --config $C --checkpoint $W2 --session-dir $S --optical-pass language_expert

python -m $M finetune --config $C --checkpoint $W2 --session-dir $S --stage language_expert --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W3 = "$S\checkpoints\after_language_expert_best_test.pt"
```

### D. 序列全局层和最终测试

```powershell
python -m $M export-pass --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\06_language_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global
python -m $M finetune --config $C --checkpoint $W3 --session-dir $S --stage language_global --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W4 = "$S\checkpoints\after_language_global_best_test.pt"
python -m $M evaluate --config $C --checkpoint $W4 --session-dir $S --stage language_global --batch-size 16 --device cuda
```

本项目按老师要求不划验证集：每 5 epoch 在完整 test 上计算 Spatial SRCC，保留历史最高 test SRCC。最终指标、逐样本预测和 router 使用率均写入 `$S\final_evaluation`。

## 3. 不能混用的内容

- Spatial 4 帧 mask/缓存不能与 Temporal 9 帧或旧 16 帧互换。
- 相位 BMP 已按实验台配置翻转；播放软件不要再次翻转。
- 每个 pass 只加载本 pass 目录中的唯一相位 BMP。
- 不要用 `strict=False` 绕过权重合同；不要添加 Attention、Transformer 或电子 router。
- `amplitude_layout_1024x1024` 是几何预览，不是正式样本振幅。
