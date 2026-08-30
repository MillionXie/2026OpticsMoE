# 完整实验包：从这里开始

本文件对应 `qwen_mnist4_early_robust_tradeoff_local_finetune_full_lab.zip`。
该 ZIP 是独立完整包，不需要先下载或覆盖旧包。请解压到短路径，例如
`E:\code\guest\qwen_early_robust_full_lab`，所有命令都在该根目录执行。

## 1. 只编辑实验配置

确认 `experiments\lab_qwen\LAB_CONFIG.yaml` 中仍是已经实测通过的 3500 μs
曝光、新线性 LUT、当前四顶点。然后运行：

```powershell
conda activate xml
python -m experiments.lab_qwen.prepare_lab
```

不要用本包覆盖实验室已验证的 LUT 文件。`prepare_lab` 输出必须明确显示所选
LUT 存在、曝光为 3500 μs、homography contract 已生成。

## 2. 首选 accuracy-first（仿真 Top-1 85%）

当前相位 SLM 可以继续保持全零，直到准备正式采第一层。正式采集时手动加载：

`experiments\lab_qwen\four_accuracy_first\01_vision_expert\phase_to_play\vision_expert.bmp`

确认相位屏显示的是这张文件，再运行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\lab_qwen\generated\formal_hardware.yaml --stage-dir experiments\lab_qwen\four_accuracy_first\01_vision_expert --clear-output
```

该目录已经包含 210 张可直接播放的 1024×1024 振幅 BMP，不需要重建。采完后：

```powershell
python -m experiments.lab_qwen.local_four_stage --profile accuracy_first --stage vision_expert --epochs 100
```

程序按 development Top-1、同分再按 CE 选择 checkpoint；100 张 sealed test
只在选模后评一次。随后自动生成并重建第二层。第二次实验依次加载并采集：

`experiments\lab_qwen\four_accuracy_first\02_vision_global\phase_to_play\vision_global.bmp`

然后把命令中的 stage 改为 `vision_global`。后两层依次为
`language_expert`、`language_global`。

## 3. balanced 备选

只有 accuracy-first 对环境变化敏感或需要论文 trade-off 时才完整采 balanced。
目录改为 `four_balanced`，本地命令必须改为 `--profile balanced`。两个 profile
的 checkpoint 和会话目录完全隔离，禁止交叉加载。

## 4. 结果判定

最终要求是四层后 sealed-test Top-1 ≥78%。建议的停止检查和全部服务器仿真
结果见：

- `experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff\RESULTS.md`
- `experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff\RUN_COMMANDS.md`

若 PCC/SSIM、方向或 ROI 不合格，先修光路/配置，不要靠增加微调 epoch 掩盖。
