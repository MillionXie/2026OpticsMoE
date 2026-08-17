# 两级 Vision 实光路协议

实验只采集 YAML 中指定的硬件子集，默认如下：SALICON `256 train + 64 eval`，ISIC `512 + 128`，LSP `512 + 128`。每层上传约为 `样本数 × 478×478 uint8`，不再传 16-bit 原图或 900 多像素的中间图。

## 文件职责

实验室电脑只做以下工作：

1. 把服务器的 `compact_amplitude/` 和唯一 phase mask 重建成完整 SLM 图。
2. 播放、固定 ROI、area resize，直接保存 `478×478`、灰度、8-bit PNG。
3. 保持 basename 不变，把 `ccd_captured/` 上传服务器。

实验室端不翻转 CCD。服务器按 YAML 的 `ccd_flip_vertical/horizontal` 解释方向。当前没有暗场或背景采集，所以整个流程不做背景扣除。

服务器只保留 manifest、compact payload、实测 `478 uint8` CCD、checkpoint 和指标；不会生成 `simulation_ccd/` 或 `ccd_registered/*.pt` 缓存。

## 固定执行顺序

```text
联合训练 checkpoint
  → export vision_expert
  → 实验室重建/采集/上传
  → finetune vision_expert
  → after_vision_expert.pt
  → export vision_global（会读取已上传的 expert CCD）
  → 实验室重建/采集/上传
  → finetune vision_global
  → after_vision_global.pt
```

采集某层后，该层以及它之前的 phase/router 会被冻结，只微调最新 CCD 下游的电子 Mixer、readout、融合门、尚未采集的光学层和任务 decoder。因此不会出现“改完上游后，刚采集的 CCD 已经不再对应当前网络”的问题。

## 实验室重建与采集

将下文的 `STAGE_DIR` 替换为相应的 `01_vision_expert` 或 `02_vision_global`：

```bash
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir STAGE_DIR/compact_amplitude --output-dir STAGE_DIR/amplitude_to_play --slm-width 1920 --slm-height 1080 --scale-factor 2

python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir STAGE_DIR/compact_phase --output-dir STAGE_DIR/phase_to_play --slm-width 1920 --slm-height 1200 --scale-factor 2 --center-x 980 --center-y 590

python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments/hardware_sdk/configs/tucam_windows.yaml --input-dir STAGE_DIR/amplitude_to_play --output-dir STAGE_DIR/ccd_captured --clear-output
```

`--center-x/--center-y` 是完整 SLM 画布中的有效区中心坐标，原点在左上角，x 向右、y 向下；两项必须同时设置。当前相位 SLM 标定为 `(980,590)`，相对几何中心 `(960,600)` 即向右 20 px、向上 10 px。对于 `956×956` 有效区，最终边界为 `(502,112,1458,1068)`，会写入 `reconstruction_manifest.csv`。振幅 SLM 未传中心参数，因此继续使用几何中心。

相机工作流必须输出目标大小；若相机原 ROI 不是 478，则在实验室端用 area resize 一次变为 478。不要在服务器再次裁剪或重复缩放。

## 通用硬件命令格式

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task TASK --config CONFIG --checkpoint CHECKPOINT --session-dir SESSION --stage vision_expert --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task TASK --config CONFIG --checkpoint CHECKPOINT --session-dir SESSION --stage vision_expert --phase finetune --epochs 20
```

第二层必须把 `CHECKPOINT` 换成 `SESSION/checkpoints/after_vision_expert.pt`。如需临时减小采集规模，可同时传入 `--train-limit 64 --eval-limit 32`；同一个 session 的两个 stage 必须使用完全相同的 limit。
