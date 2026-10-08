# LGVQ Spatial-4 正常加光正式实验包

本包对应正式仿真 checkpoint：Spatial SRCC `0.671008`。只包含这一份正常加光权重，不包含随机相位、去光或逐 epoch checkpoint。去光与最后相位随机对照仅作为 JSON/Markdown 数值证据随包提供。

> **交接包边界：**这是给师姐及其 AI 做工程适配的轻量源码/权重包，不是携带 LGVQ 数据的离线复现包。数据集、四帧 RGB 缓存、Qwen 图像/文本前端缓存以及厂商 SDK 二进制均被有意排除。下面的命令描述适配完成后的目标流程；在师姐的 AI 补齐或重建 `configs/deployment/spatial4_readout_1m_lab.yaml` 所列输入之前，不要直接执行 preflight 或全量微调。

模型固定为 4 帧 `2x2` 并行、109×109 光学专家、光学 Top-2 Router、四次同 RMS O/E 融合、532 nm、17 μm 振幅 SLM、8 μm 相位 SLM、10 cm 和 478×478 有效光场。学生推理不执行 Qwen Transformer/Attention block；冻结的 Qwen 图像 patch/位置前端和文本 tokenizer/embedding 已离线缓存。

> **更新后的交付边界（以此段为准）：**轻量包仍不携带 LGVQ 原始数据、约
> 6.6 GiB 的派生缓存或厂商 SDK 二进制，但已经包含从原始 LGVQ 重建四帧 RGB、
> Qwen 图像/文本前端及冻结 Conv5 输入所需的代码，并随包携带本项目专用的
> 339,312 参数 Conv5 预处理资产。接收方已有 LGVQ 和官方 Qwen3-VL-2B
> 本地权重即可按 `documentation/DATA_ADAPTER_CONTRACT.md` 重建缓存并推理。
> `quality_stem_state.pth` 不是第二个模型版本，而是第一层电子残差所需的冻结
> 输入变换。

## 0. 解压、校验与固定变量

在 ZIP 解压根目录打开 PowerShell：

```powershell
conda activate xml
python VERIFY_BUNDLE.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

$P = 'experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
$M = 'experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.hardware_bridge'
$C = "$P\configs\deployment\spatial4_readout_1m_lab.yaml"
$W0 = "$P\deployment\checkpoints\best_observed_test_checkpoint.pt"
$S = 'experiments\lab_lgvq\sessions\spatial4_readout_1m_formal01'
$H = 'experiments\lab_lgvq\generated\formal_hardware.yaml'
```

先验证最新版配置与 checkpoint 匹配：

```powershell
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.run --config $C --phase preflight
```

输出必须为 `status: ready`，并明确写出 Vision/Language Qwen block 执行数均为 `0`。

## 1. 配置实验台与标定

只编辑：

```text
experiments\lab_lgvq\LAB_CONFIG.yaml
```

填写 Meadowlark LUT 文件名及 SHA256、曝光、相位 SLM 中心、CCD 全传感器四个逻辑角点，然后运行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
python -m experiments.hardware_sdk.workflows.roi_calibration exposure --config $H
```

正式 CCD 文件保存为单应性矫正后的 478×478 原始强度。保存阶段不做逐图 min-max、背景扣除或 log；进入网络后与仿真共用相同的非负截断、均值归一化、相对强度截断和 `log1p`。

## 2. 六个光学 pass 与四次微调

每个 pass 都按“导出振幅 → 手动加载本 pass 唯一相位 BMP → CCD 采集 → 校验”执行。正式实验保留 `--all-data`，即 2250 train + 558 test。每 5 epoch 测一次完整 test，保存最高 test SRCC。

### Vision Router 与 Vision Expert

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

### Vision Global

```powershell
python -m $M export-pass --config $C --checkpoint $W1 --session-dir $S --optical-pass vision_global --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\03_vision_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $W1 --session-dir $S --optical-pass vision_global
python -m $M finetune --config $C --checkpoint $W1 --session-dir $S --stage vision_global --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W2 = "$S\checkpoints\after_vision_global_best_test.pt"
```

### Language Router 与 Language Expert

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

### Language Global 与最终评估

```powershell
python -m $M export-pass --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global --all-data --device cuda
python -m experiments.hardware_sdk.workflows.acquire_folder --config $H --stage-dir "$S\06_language_global" --clear-output
python -m $M validate-capture --config $C --checkpoint $W3 --session-dir $S --optical-pass language_global
python -m $M finetune --config $C --checkpoint $W3 --session-dir $S --stage language_global --epochs 100 --batch-size 16 --test-interval 5 --device cuda
$W4 = "$S\checkpoints\after_language_global_best_test.pt"
python -m $M evaluate --config $C --checkpoint $W4 --session-dir $S --stage language_global --batch-size 16 --device cuda
```

最终指标、逐视频预测、同 checkpoint 去光对照和 Router 使用率位于 `$S\final_evaluation`。

## 3. 包内固定相位 BMP

无需重新导出时，六张正常加光的最佳相位位于：

```text
experiments\qwen3_vl_2b_lgvq_single_metric_o2_16frame_54\deployment\hardware_masks\phase_slm_1920x1200\
```

顺序为 `vision_router → vision_expert → vision_global → language_router → language_expert → language_global`。播放程序不要再次翻转。

## 4. 禁止混用

- 只能使用 `spatial4_readout_1m_lab.yaml` 和本包 `$W0`；不要改回旧 `spatial4_custom_conv_lab.yaml`。
- 不要与 Temporal checkpoint、mask 或缓存混用。
- 不要用 `strict=False` 绕过权重合同。
- 不要在正式推理中增加 Attention、Transformer、电子 Router 或独立 MOS 旁路。
- 完整 Qwen 仅在从新原始视频重新生成前端缓存时需要，正常硬件闭环及本地微调不加载它。

架构、正式指标及消融审计见 `documentation/`。当前同 checkpoint 正常加光 SRCC 为 `0.671008`，全部去光为 `0.615902`；随机最后相位的五种子均值为 `0.670974`，说明完整光路有效，但最后单独一张相位在当前读出中利用较弱。
