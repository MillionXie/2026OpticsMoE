# Qwen Caltech101 10 cm 光路实验独立包

这个 ZIP 同时承担两件事：

1. 实验室 Windows 电脑只安装轻量环境，即可重建 1024×1024 振幅 BMP、顺序播放 Meadowlark 振幅 SLM，并用 TUCam 直接保存 478×478、8-bit CCD PNG；
2. 保存与本次实验严格对应的 checkpoint、四张相位 BMP、quick210 理论振幅、配置/代码和可选训练证据，便于把 CCD 上传回服务器后继续微调。

实验室电脑不需要 Torch、Transformers 或 Qwen 权重。模型推理、理论振幅导出和硬件微调仍在服务器执行。

## 1. 固定物理合同

| 项目 | 固定值 |
|---|---:|
| 波长 | 532 nm |
| 每层传播距离 | 0.10 m |
| 逻辑光场 | 478×478 |
| 振幅 SLM | 1024×1024，17 µm，逻辑像素 1:1 播放 |
| 相位 SLM | 1920×1200，8 µm，按物理坐标最近邻栅格化 |
| 相位中心 | 以配置/导出 report 为准，当前训练流程允许修改 |
| 每层光学融合门控 | 数学硬下限 0.10；本包不改变训练值 |
| 振幅极性 | 255=白/亮/透光，0=黑/暗/遮光 |
| CCD 上传 | 478×478、8-bit、灰度 PNG，不翻转、不做背景扣除 |

禁止混用旧的 `_inv`/`inverted` 振幅。实验室程序不会控制相位 SLM；操作者必须手动加载该 stage 唯一的相位 BMP，程序只核对其尺寸和 SHA-256。

## 2. ZIP 内容与空间策略

解压后的关键结构如下：

```text
README_LAB_AND_SERVER.md
bundle_manifest.json
payload/
├── checkpoint/                       # 本包唯一选定checkpoint
├── four_phase_export/
│   ├── compact_phase/                # 4张478 PNG
│   ├── phase_bmp/                    # 4张1920×1200 BMP
│   └── phase_export_report.json
└── quick210/
    ├── manifest.csv                  # 210个样本的唯一顺序/键
    └── 04_language_global/
        ├── compact_amplitude/        # 210张478 PNG理论振幅
        ├── compact_amplitude_manifest.csv
        ├── compact_phase/
        ├── phase_to_play/            # 第四层唯一相位BMP
        ├── transport_spec.json
        └── ccd_captured/              # 采集时自动建立
experiments/hardware_sdk/             # 轻量控制代码、配置、可选vendor SDK
reference/qwen_project_source/        # 服务器端代码/配置快照，仅供追溯
reference/training_evidence/           # 仅在打包时显式选择才存在
```

`bundle_manifest.json` 对每个归档文件记录类别、大小和 SHA-256，并记录本包物理合同。ZIP 外还会生成同名 `.zip.json`，其中包含整个 ZIP 的 SHA-256。

为了节省传输/磁盘，本包明确不包含：Caltech101 图片、Hugging Face/模型下载 cache、teacher/feature cache、CCD、可重建的 `amplitude_to_play` 全尺寸 BMP，以及除显式 checkpoint 之外的其他 checkpoint。210 张 1024×1024 BMP 在实验室重建，采完即可删除；应保留 compact PNG、manifest、CCD 和 acquisition log。

## 3. 服务器生成 quick210 与打包

以下变量在服务器仓库根目录设置：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust
RUN=$PROJECT/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust
CKPT=$RUN/ema_best_train_loss_checkpoint.pt
PHASE_EXPORT=$RUN/hardware_phase_export
QUICK_SESSION=$PROJECT/hardware_sessions/quick_language_global_10cm_robust_run1
```

先导出四张相位 BMP：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.export_phase_bmps --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $CKPT --output-dir $PHASE_EXPORT
```

再用前三层仿真导出第四层 quick210 理论振幅。该独立配置固定为每类10张 train、10张 test、1张 gallery，共210张；不能当正式全量 test 指标：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_quick_last_stage_10x10.yaml --checkpoint $CKPT --session-dir $QUICK_SESSION --stage language_global --phase export --upstream-source simulation
```

最后打包。这一步只有文件校验、哈希和 ZIP 压缩，不调用 GPU：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.lab_package --checkpoint $CKPT --phase-export-dir $PHASE_EXPORT --quick-session-dir $QUICK_SESSION --output $RUN/qwen_caltech101_10cm_quick210_lab_bundle.zip --include-evidence --run-dir $RUN
```

正式交给一台没有 SDK 的实验室电脑时不要加 `--omit-vendor-sdk`。只有实验室电脑已经有同版本 vendor SDK、且只想制作小型开发包时才允许加这个参数。已存在同名 ZIP 时必须显式加 `--overwrite`，避免误覆盖。

## 4. 实验室电脑一次性准备

使用 64-bit Python 3.10 或 3.11；Python、Meadowlark/TUCam DLL 和操作系统必须同为64位。解压 ZIP，PowerShell 进入解压根目录：

```powershell
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\requirements-lab.txt
```

编辑：

```text
experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

至少确认：

- `amplitude_slm.lut_file` 与真实 SLM/温度一致；
- `camera.exposure_us` 合适；
- `camera.device_roi_xywh=[left,top,width,height]` 是四焦点标定得到的实际 ROI，且满足相机 SDK 的4像素整除要求；
- `saved_frame_size_wh=[478,478]`、`saved_frame_bit_depth=8` 不要修改；
- 相位 SLM 上加载的 BMP 与当前 stage 的 `phase_to_play` 唯一文件一致。

## 5. quick210：只测第四层

在解压根目录执行：

```powershell
$STAGE = "payload\quick210\04_language_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
```

手动加载：

```text
payload\quick210\04_language_global\phase_to_play\language_global.bmp
```

先试拍3张。`--clear-output` 会清空当前 stage 已有采集，首次或明确重拍时才使用：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
```

确认极性、曝光、ROI、文件 stem 和相位均正确后，全量重拍210张：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

上传回服务器的最小内容是：

```text
04_language_global/ccd_captured/*.png
04_language_global/acquisition_logs/*
```

放回服务器原 `$QUICK_SESSION/04_language_global/` 后微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_quick_last_stage_10x10.yaml --checkpoint $CKPT --session-dir $QUICK_SESSION --stage language_global --phase finetune --upstream-source simulation --epochs 10
```

输出：

```text
$QUICK_SESSION/checkpoints/after_language_global.pt
$QUICK_SESSION/04_language_global/finetune_metrics.json
```

该结果的准确表述是“前三层仿真 + 第四层实测”，不是四层全部实测，也不是正式全量 test。

## 6. 正式四层逐层采集与微调

固定正式 session：

```bash
SESSION=$PROJECT/hardware_sessions/four_layer_10cm_robust_run1
```

严格顺序为：

```text
vision_expert → vision_global → language_expert → language_global
```

每层都执行同一个闭环：服务器 export → 把该 stage 传到实验室 → 重建振幅/采 CCD → CCD 和 acquisition_logs 传回同一路径 → 服务器 finetune。下一层必须使用上一层输出的 `after_<stage>.pt`，不能一直使用初始 checkpoint。

### 6.1 Vision expert

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $CKPT --session-dir $SESSION --stage vision_expert --phase export --upstream-source measured
```

实验室 stage：`01_vision_expert`。采集上传后：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $CKPT --session-dir $SESSION --stage vision_expert --phase finetune --upstream-source measured --epochs 20
```

### 6.2 Vision global

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $SESSION/checkpoints/after_vision_expert.pt --session-dir $SESSION --stage vision_global --phase export --upstream-source measured
```

实验室 stage：`02_vision_global`。采集上传后：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $SESSION/checkpoints/after_vision_expert.pt --session-dir $SESSION --stage vision_global --phase finetune --upstream-source measured --epochs 20
```

### 6.3 Language expert

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $SESSION/checkpoints/after_vision_global.pt --session-dir $SESSION --stage language_expert --phase export --upstream-source measured
```

实验室 stage：`03_language_expert`。采集上传后：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $SESSION/checkpoints/after_vision_global.pt --session-dir $SESSION --stage language_expert --phase finetune --upstream-source measured --epochs 20
```

### 6.4 Language global

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $SESSION/checkpoints/after_language_expert.pt --session-dir $SESSION --stage language_global --phase export --upstream-source measured
```

实验室 stage：`04_language_global`。采集上传后：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config $PROJECT/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint $SESSION/checkpoints/after_language_expert.pt --session-dir $SESSION --stage language_global --phase finetune --upstream-source measured --epochs 20
```

最终输出：

```text
$SESSION/checkpoints/after_language_global.pt
$SESSION/04_language_global/finetune_metrics.json
```

### 每个正式 stage 的实验室命令

把当前层目录复制到解压根目录下的任意位置，并设置 `$STAGE`：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

## 7. 交接前检查

- 对比 ZIP 外 `.zip.json` 的 `zip_sha256`；
- 在 `bundle_manifest.json` 查 checkpoint、四张 phase BMP 和210张 compact amplitude 的哈希；
- quick stage 必须是 `language_global`、`upstream_source=simulation`、`samples=210`；
- 正式逐层 stage 必须是 `upstream_source=measured`；
- 每张 CCD 与播放 BMP 同 stem，不能手工重命名；
- 不做凭空背景扣除；不在上传完成后对同一 stage 再运行 `--clear-output`；
- 只上传 compact/manifest/CCD/log，`amplitude_to_play` 可重建，不必回传服务器；
- `best_observed_test_checkpoint.pt` 是 test 选择偏置的诊断上界，不应用作默认固定评估或默认硬件 checkpoint；默认使用 train-only 选择的 `ema_best_train_loss_checkpoint.pt`。
