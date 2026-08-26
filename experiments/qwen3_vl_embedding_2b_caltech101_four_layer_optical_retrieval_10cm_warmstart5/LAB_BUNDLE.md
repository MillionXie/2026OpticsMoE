# Warmstart5 实验室独立包

这个包对应唯一固定的 Stage-B EMA checkpoint：光电融合系数硬下限为 5%，
初始化值为 5.5%，固定仿真测试 Top-1 为 81.00%。打包器会同时核对 checkpoint、
四张 phase、quick210 transport 和离线末层 cache/state 的 SHA-256；任一来源不一致
都会拒绝生成 ZIP。

这里的 5% 是残差融合系数下限，而不是光能量占比：

```text
alpha = 0.05 + 0.95 * sigmoid(raw_gate)
output = electronic + alpha * optical_delta
```

## 1. 两套相互独立的实验室环境

仅播放和采集时不需要 Torch、Transformers、Qwen 或模型权重参与计算：

```powershell
python -m venv .venv_capture
.\.venv_capture\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-lab.txt
```

如果实验室电脑还要直接微调第四层后面的轻量电子 tail，另建环境：

```powershell
python -m venv .venv_offline
.\.venv_offline\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\requirements-offline-finetune.txt
```

第二套环境只加载约 255,811 个可训练参数，不加载 Qwen、Transformers、原始
Caltech101 图片或光学传播器。弱 CPU 也能运行；若机器已有适配的 CUDA PyTorch，
可使用 `--device cuda` 加速。

## 2. ZIP 内容与明确排除项

核心目录如下：

```text
README_LAB_AND_SERVER.md
bundle_manifest.json
payload/
├── checkpoint/                       # 仅一个固定 Stage-B EMA checkpoint
├── four_phase_export/                # 四层 compact phase + 1920×1200 phase BMP
└── quick210/
    ├── manifest.csv                  # 100 train + 10 gallery + 100 test
    └── 04_language_global/
        ├── compact_amplitude/        # 210 张 478×478 uint8 PNG
        ├── compact_phase/
        ├── phase_to_play/            # 固定 language_global phase BMP
        └── offline_downstream/
            ├── cache.pt              # 冻结边界：Language Block-2 输入 [L,192]
            ├── downstream_state.pt   # 255,811 参数的初始轻量 tail
            └── contract.json
experiments/hardware_sdk/             # 重建、Meadowlark 播放、TUCam 采集及 vendor SDK
reference/training_evidence/           # Stage A/B 小型日志与固定测试 JSON
reference/qwen_project_source/         # 服务器训练代码/配置快照，仅用于追溯
```

ZIP 不包含 Caltech101 原图、Hugging Face/Qwen cache、teacher cache、CCD、
`amplitude_to_play` 全尺寸 BMP 或额外 checkpoint。`offline_downstream/cache.pt`
是唯一刻意保留的轻量 latent cache，用于在实验室电脑跳过大模型前向。

## 3. 第四层快速采集

从解压后的 ZIP 根目录执行。先把逻辑振幅重建为 1024×1024、17 μm、1:1 播放
BMP；该大文件只在实验室本地生成，不需回传：

```powershell
$STAGE = "payload\quick210\04_language_global"

python -m experiments.hardware_sdk.workflows.reconstruct_slm `
  --stage-dir $STAGE `
  --payload amplitude `
  --hardware-profile meadowlark_17um
```

手动向相位 SLM 加载：

```text
payload\quick210\04_language_global\phase_to_play\language_global.bmp
```

首次使用新相机时，先在
`experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml` 中把
`camera.device_roi_xywh` 填成实验标定得到的 `[left, top, width, height]`；四个值须满足
TUCam 对齐要求。模板保留 `null` 是为了阻止未经标定的 ROI 被误当作正式数据。

然后先做只读校验，再正式播放 210 张振幅并采集：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --validate-only

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --clear-output
```

相机必须直接保存为与 manifest key 同名的 210 张 `478×478`、8-bit、PIL mode
`L` PNG。不要做背景扣除、拉伸、翻转、resize 或 16-bit 中间文件。配置中的
CCD flip 会在模型读入时统一执行。

## 4. 可选：实验室离线微调第四层 tail

采集后先严格检查文件集合、尺寸、hash、split、cache 和 state：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir payload\quick210 `
  --device cpu `
  --validate-only
```

校验通过后微调 10 epoch：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir payload\quick210 `
  --device auto `
  --epochs 10
```

可训练范围严格为：Language Block 2、CCD `224→192` adapter、融合 gate、融合后
LayerNorm 和 64 维 retrieval head，共 255,811 参数。它执行与服务器第四层相同的
CCD 合同和下游电子结构，但不更新前三层、Qwen、phase 或 router。checkpoint 仅按
train loss 选择，固定 gallery/test 不参与选轮。

输出位于：

```text
payload/quick210/04_language_global/offline_results/
├── best_offline_tail_state.pt
├── train_log.csv
├── metrics.json
└── ccd_inventory.json
```

该结果只能表述为“前三层仿真 + 第四层实测 + 第四层下游电子微调”，不能表述为
四层全部实测。

## 5. 服务器负责的工作

完整 Qwen 训练、理论振幅导出、四层逐层采集后的全模型微调仍在服务器完成。实验室
电脑只需回传 `ccd_captured`，不需要回传重建后的 1024×1024 振幅 BMP。

服务器 quick210 导出命令：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
MODULE=experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CKPT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
SESSION=$PROJECT/hardware_sessions/quick_language_global_run1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $PROJECT/configs/release/quick_last_stage_10x10.yaml \
  --checkpoint $CKPT \
  --session-dir $SESSION \
  --stage language_global \
  --phase export \
  --upstream-source simulation
```

正式打包命令：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_package \
  --checkpoint $CKPT \
  --phase-export-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/hardware_phase_export \
  --quick-session-dir $SESSION \
  --stage-a-run-dir $PROJECT/runs/caltech101_warmstart5_stage1_optical_calibration \
  --stage-b-run-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test \
  --output $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/qwen_caltech101_10cm_warmstart5_quick210_lab_bundle.zip \
  --overwrite
```

不要在正式移交时使用 `--omit-vendor-sdk`。打包完成后保留终端报告中的 ZIP
SHA-256 和同名 `.zip.json` sidecar；解压后的 `bundle_manifest.json` 可逐文件核对。

## 6. 四层逐层实验边界

完整四层顺序固定为：

```text
vision_expert → vision_global → language_expert → language_global
```

每完成一层采集，就在服务器按该层 CCD 微调其下游，再导出下一层输入。这个 quick210
包只预生成“前三层仿真、第四层实测”的最快路径；不能拿第四层 quick payload 代替
四层逐层 session。四张正式 phase BMP 已随包保存，便于核对版本和相位 SLM 加载。
