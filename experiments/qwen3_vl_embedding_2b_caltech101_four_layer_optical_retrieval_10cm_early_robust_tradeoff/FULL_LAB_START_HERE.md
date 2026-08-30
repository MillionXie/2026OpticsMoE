# 实验室完整流程：全数据逐层微调

本工程有两种互不混用的数据口径：

- `accuracy_first_full`：正式全数据流程。每类保留 3 张 gallery、20 张 sealed test，其他图像全部用于训练和 development 选模。
- `accuracy_first`：仅用于快速检查的 210 帧流程，每类 10 train、1 gallery、10 test。

正式实验使用 `accuracy_first_full`。仿真 sealed-test Top-1 为 85.0%；该数值不是实测保证值，四层实测目标为最终 Top-1 不低于 78%。

## 1. 全量微调的准确含义

每采完一层，该层之后的全部紧凑电子网络、尚未采集的光学分支参数和 retrieval readout 都参与训练。已经采集的上游层必须冻结，否则下一层已经播放的输入会失效。

Qwen3-VL-Embedding-2B 的冻结视觉/语言骨干不解冻。实验室 RTX 5060 不适合对 2B 骨干做全参数训练；Caltech101-10 数据规模也不足以安全微调 2B 参数。这不是只训练最后一个 MLP。

checkpoint 按固定 development Top-1 选择，同分再比较 development CE。sealed test 不参与选模，只在恢复最佳 checkpoint 后评估一次。

## 2. 旧工程参数迁移

新工程放在独立目录，不覆盖旧工程。下列经过实测的硬件参数必须迁移：

- `LAB_CONFIG.yaml` 中的 3500 μs 曝光；
- `slm7930_at532-70c-pixel-2_linearized-amplitude-3500us.lut`；
- CCD 四个逻辑角点、ROI margin、60 ms settle delay、丢帧数和 warmup 帧数；
- Meadowlark、TUCam SDK 和设备编号。

迁移后在新工程根目录执行：

```powershell
conda activate xml
python -m experiments.lab_qwen.prepare_lab
```

检查输出中的 LUT 文件、`camera_exposure_us: 3500.0`、四点 homography 和输出 478×478 均正确。不要复制旧的 `generated` 文件；它们应由新工程重新生成。

## 3. 第一层采集

第一层目录：

`experiments\lab_qwen\four_accuracy_first_full\01_vision_expert`

先确认 `amplitude_to_play` 已存在并包含全部 BMP。手动把相位 SLM 从纯黑改为：

`phase_to_play\vision_expert.bmp`

确认相位图确实加载后执行：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four_accuracy_first_full\01_vision_expert `
  --clear-output
```

采完后执行本地全数据微调：

```powershell
python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage vision_expert `
  --epochs 100
```

程序最多训练 100 epoch，development 连续 15 epoch 无提升会提前停止，恢复 development 最优 checkpoint 后只评一次 sealed test，并自动导出、重建第二层输入。

## 4. 后三层

依次重复“加载本层 phase BMP → 采 CCD → 本地微调”：

1. `vision_global`
2. `language_expert`
3. `language_global`

命令只替换 `--stage`。各层目录依次为 `02_vision_global`、`03_language_expert`、`04_language_global`。不得跨 profile 复制 checkpoint 或 CCD。

## 5. 结果位置

每层指标：`<stage_dir>\finetune_metrics.json`

逐层 checkpoint：`experiments\lab_qwen\four_accuracy_first_full\checkpoints\after_<stage>.pt`

最后以第四层 `finetune_metrics.json` 中的 `sealed_test` 为正式实测结果。若低于 78%，先检查 PCC/SSIM、方向、ROI、饱和率和 LUT；不要反复观察 test 后挑 epoch。

`balanced_full` 是第二套鲁棒性 trade-off，流程相同，但 profile 和目录必须全部改为 `balanced_full` / `four_balanced_full`。先完成 accuracy-first，暂不要求采 balanced。
