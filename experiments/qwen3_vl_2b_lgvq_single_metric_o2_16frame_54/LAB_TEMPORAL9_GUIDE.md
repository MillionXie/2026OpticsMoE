# Temporal-9 光电视频质量实验：从解压到最终结果

本包只对应 LGVQ Temporal quality 的正式 9 帧候选。它固定使用光学 Top-2、3×3
帧并行、77×77 视觉专家、109×109 序列专家、532 nm、17 µm 逻辑像素和 10 cm
传播。推理中没有 Transformer 或 Attention；Qwen3-VL 的冻结 patch/position
embedding 和文本 tokenizer/embedding 已缓存进包，无需联网下载 Qwen。

## 0. 解压后检查

在 ZIP 解压根目录打开 PowerShell：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

唯一模型配置：

```powershell
$P = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
$M = 'experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.hardware_bridge'
$C = "$P\configs\deployment\temporal9_lab.yaml"
$W0 = "$P\deployment\checkpoints\best_observed_test_checkpoint.pt"
$S = 'experiments\lab_lgvq\sessions\temporal9_formal01'
$H = 'experiments\lab_lgvq\generated\formal_hardware.yaml'
```

## 1. 只编辑实验台配置并完成标定

只编辑 `experiments\lab_lgvq\LAB_CONFIG.yaml`：填写 Meadowlark LUT 文件名及
SHA256、曝光时间、相位 SLM 中心/翻转关系，以及 CCD 全传感器四个逻辑角点。标定图
位于 `experiments\lab_lgvq\calib`。然后执行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config $H
```

正式采集保存的是硬件 ROI 经固定单应性矫正后的 478×478 强度；文件本身不做背景
扣除、逐帧 min-max 或 log。网络读取仿真/实测 CCD 时统一执行：非负截断、单帧均值
归一化、相对强度截断、`log1p`。

## 2. 六个 pass 与四次本地微调

每一小节都必须按“导出 → 采集 → 校验 → 微调”执行。第一次试台可去掉
`--all-data`（64 train + 32 test）；正式实验保留 `--all-data`（2250 train + 558
test）。采集命令会自动把 478×478 紧凑 PNG 以 1:1 放入 1024×1024 振幅 SLM。

### A. 视觉 router 与专家

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

### C. 序列 router 与专家

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

### D. 序列全局层与最终测试

```powershell
python -m $M export-pass --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\06_language_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global
python -m $M finetune --config $C --checkpoint $W3 --session-dir $S --stage language_global --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W4 = "$S\checkpoints\after_language_global_best_test.pt"
python -m $M evaluate --config $C --checkpoint $W4 --session-dir $S --stage language_global --batch-size 16 --device cuda
```

每 5 epoch 在完整 test 上计算一次 Temporal SRCC，并保存历史最佳权重；本项目按老师
要求明确允许 test 参与选模。最终指标位于 `$S\final_evaluation`。

## 3. 绝对不能混用的内容

- 9 帧正式相位与旧 4/16 帧相位不能互换；每个 pass 只加载自身目录中唯一 BMP。
- 后一 pass 必须用前一阶段微调后的 checkpoint 重新导出，不能六个 pass 一次性拍完。
- 相位 BMP 已按 `LAB_CONFIG.yaml` 做硬件方向翻转；播放软件不要再翻转。
- `amplitude_layout_1024x1024` 只是几何预览，不能当作正式样本振幅。
- 不要以 `strict=False` 加载 checkpoint，不要插入 Attention/Transformer/电子 router。

