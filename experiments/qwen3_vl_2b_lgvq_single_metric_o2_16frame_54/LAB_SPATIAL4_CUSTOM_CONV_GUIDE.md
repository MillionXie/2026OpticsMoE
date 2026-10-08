# Spatial-4 自写卷积光电模型：实验室完整流程

本包只评价 LGVQ **Spatial quality**，每个视频输出一个连续 MOS。固定 prompt 为：

> Please evaluate the spatial quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.

模型使用 4 帧 2x2 并行、109x109 光学专家、光学 Top-2 Router、532 nm、17 um、
10 cm 和 478x478 有效光场。电子 E1 中的额外图像模块由本项目直接用 Conv、GroupNorm、
GELU 和 Linear 编写，共 316,568 个参数；包内不依赖任何第三方图像骨干或其特征缓存。
训练、评估和硬件微调不会加载完整 Qwen，Vision/Language Transformer block 执行数均为 0。

## 0. 解压、校验与变量

从 ZIP 解压根目录打开 PowerShell：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

$P = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
$M = 'experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.hardware_bridge'
$C = "$P\configs\deployment\spatial4_custom_conv_lab.yaml"
$W0 = "$P\deployment\checkpoints\best_observed_test_checkpoint.pt"
$S = 'experiments\lab_lgvq\sessions\spatial4_custom_formal01'
$H = 'experiments\lab_lgvq\generated\formal_hardware.yaml'
```

先做软件侧预检；输出必须明确写出 `vision_blocks_executed: 0`、
`language_blocks_executed: 0` 和 `status: ready`：

```powershell
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.run --config $C --phase preflight
```

## 1. 实验台配置与标定

只编辑 `experiments\lab_lgvq\LAB_CONFIG.yaml`：填写 Meadowlark LUT 文件名及 SHA256、
曝光、相位 SLM 中心和 CCD 全传感器四个逻辑角点。然后执行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config $H
```

正式保存单应性矫正后的 478x478 原始强度。文件阶段不做逐帧 min-max、背景扣除或 log；
仿真与实测进入网络时共用相同的非负截断、均值归一化、相对强度截断和 `log1p`。

## 2. 六个光学 pass 与四次本地微调

每个 pass 都按“导出 -> 手动加载该目录唯一相位 BMP -> 振幅 SLM/CCD 采集 -> 校验”执行。
正式实验保留 `--all-data`，即 2250 train + 558 test。

### A. 视觉 Router、视觉专家

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

### C. 语言 Router、语言专家

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

### D. 语言全局层与最终测试

```powershell
python -m $M export-pass --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\06_language_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global
python -m $M finetune --config $C --checkpoint $W3 --session-dir $S --stage language_global --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W4 = "$S\checkpoints\after_language_global_best_test.pt"
python -m $M evaluate --config $C --checkpoint $W4 --session-dir $S --stage language_global --batch-size 16 --device cuda
```

按当前论文协议不划验证集，每 5 epoch 查看完整 test SRCC 并保留历史最好权重。最终指标、
逐视频预测、同 checkpoint 光学旁路对照和 Router 使用率写入 `$S\final_evaluation`。

## 3. 禁止混用

- 不要把 Spatial-4 的 mask、缓存或 checkpoint 与 Temporal 模型互换。
- 相位 BMP 已按硬件合同导出，播放软件不要再次翻转。
- 每个 pass 只加载本 pass 目录的唯一相位 BMP。
- 不要用 `strict=False` 绕过权重合同。
- 不要在正式图中增加 Attention、Transformer、电子 Router 或独立 MOS 旁路。
- 完整 Qwen 只在从新原始视频重新生成前端缓存时需要；正常实验室闭环不需要它。
