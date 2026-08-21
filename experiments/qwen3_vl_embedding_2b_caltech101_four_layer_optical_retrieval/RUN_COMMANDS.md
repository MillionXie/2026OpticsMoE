# Caltech101 四层光电联合训练与新 17 µm 光路命令

所有服务器命令均从仓库根目录执行。新配置不会覆盖旧的 16 µm 实验。

## 1. 新光路正式训练

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um.yaml --phase train
```

该配置从头联合训练电子 Mixer、四组 phase、两个 router、四组 CCD readout、四个融合门和最终 retrieval head。Qwen 主体保持冻结，不读取电子预训练 checkpoint。

关键训练量：

- logical pitch：17 µm；
- phase LR：`5e-4`；router LR：`2e-4`；
- router balance/importance loss：`0.02/0.005`；
- 输入、phase、CCD 的逻辑错位范围均为 `±12 px`；
- 训练输出：`experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um`。

### 强 phase 训练组（当前推荐）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml --phase train
```

它仍然从头联合训练，不读取上一组 checkpoint。相对于普通 17 µm 组只改变优化强度：

- phase LR：`5e-4 → 4e-3`；
- 光学融合门初值：`0.05 → 0.15`；
- 前 5 轮正常联合 warmup，之后每 3 轮安排一次 phase-only epoch；
- phase preview 每 5 轮保存一次，显示去掉每个平面圆周均值后的相对相位，标题保留绝对均值和 rad 标准差；
- 输出目录：`runs/caltech101_four_layer_moe4_joint_17um_strong_phase`。

强 phase 组导出时，把下面命令中的配置、run 目录替换为
`caltech101_four_layer_optical_joint_17um_strong_phase.yaml` 和
`caltech101_four_layer_moe4_joint_17um_strong_phase`。

## 2. 导出训练完成后的四组相位 BMP

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.export_phase_bmps --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/ema_best_train_loss_checkpoint.pt --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/hardware_phase_export
```

输出包括：

```text
hardware_phase_export/
├── compact_phase/     # 4 张 478×478 逻辑相位 PNG
├── phase_bmp/         # 4 张 1920×1200 原生相位 SLM BMP
└── phase_export_report.json
```

逻辑 phase 仍为 `478×478 @ 17 µm`。导出器根据 `17/8=2.125` 映射为约 `1016×1016 @ 8 µm`，沿用原来的纵向翻转，再放到配置中心 `(980,590)`。修改 YAML 的 `hardware.phase_mask.center_x/center_y` 后重新执行即可，不需要重新训练。

## 3. 生成两块 SLM 的对齐图案

```bash
python -m experiments.hardware_sdk.generators.dual_slm_alignment --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml
```

输出位置：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment/
├── amplitude_bmp/     # 1024×1024，17 µm，一像素对一像素
├── phase_bmp/         # 1920×1200，8 µm，保持纵向翻转
├── registration_preview/ # 理想聚焦叠加预览，不是传播仿真
└── alignment_manifest.json
```

重点配对：

```text
amplitude checker block 8  ↔ phase checker block 17
amplitude checker block 16 ↔ phase checker block 34
amplitude checker block 32 ↔ phase checker block 68
```

老师示例对应的新配对文件使用 `registration_checker` / `registration_checker_xy`
前缀。提供逻辑格宽 `64/80/96 px` 三档，每档包含 primary 与 complement 两次曝光：

```text
amplitude_registration_checker_c80_1024x1024.bmp
phase_registration_checker_xy_c80_p8_1920x1200.bmp

amplitude_registration_checker_c80_complement_1024x1024.bmp
phase_registration_checker_xy_c80_p8_1920x1200.bmp
```

调幅图是黑白大棋盘；调相图在完全相同的逻辑格边界内逐行交替放置 x/y 二值
`0/π` 光栅。对焦且对齐后，亮格里的横/纵光栅应被调幅边界整齐截断。primary 与
complement 都拍摄后，每个相位格至少会被照亮一次。建议先用 `c96` 粗调、`c80`
确认、`c64` 精调，并根据相机图像中的边界残差修改 phase center。

相位中心需要改变时，可以修改 YAML 的 `phase_slm.center_xy`，也可以直接覆盖：

```bash
python -m experiments.hardware_sdk.generators.dual_slm_alignment --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml --phase-center-x 980 --phase-center-y 590
```

## 4. 逐层实验导出与微调

当前服务器已经完成并可用的是 `_strong_phase` run；普通
`caltech101_four_layer_moe4_joint_17um` 目录不存在。配置 YAML 和 checkpoint 必须成对
使用，不能拿普通 YAML 加载 strong-phase checkpoint。

下文使用：

```text
SESSION=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_17um_strong_run1
CKPT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/ema_best_train_loss_checkpoint.pt
```

以第一层 Vision expert 为例，服务器生成478×478紧凑 amplitude、紧凑 phase，并直接生成完整 phase BMP：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_17um_strong_run1 --stage vision_expert --phase export
```

实验室电脑把 amplitude 一对一放入新 `1024×1024` 输入 SLM：

```bash
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir SESSION/01_vision_expert/compact_amplitude --output-dir SESSION/01_vision_expert/amplitude_to_play --slm-width 1024 --slm-height 1024 --scale-factor 1 --center-x 512 --center-y 512
```

本层完整相位文件已经位于：

```text
SESSION/01_vision_expert/phase_to_play/vision_expert.bmp
```

如果只传输了紧凑 phase，也可以在实验室重建：

```bash
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir SESSION/01_vision_expert/compact_phase --output-dir SESSION/01_vision_expert/phase_to_play --slm-width 1920 --slm-height 1200 --logical-pixel-pitch-um 17 --slm-pixel-pitch-um 8 --center-x 980 --center-y 590
```

采集后的 `478×478 uint8` CCD 使用相同 basename 放入本层 `ccd_captured/`，上传服务器后微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_17um_strong_run1 --stage vision_expert --phase finetune --epochs 20
```

后续 checkpoint 链：

```text
vision_global   ← checkpoints/after_vision_expert.pt
language_expert ← checkpoints/after_vision_global.pt
language_global ← checkpoints/after_language_expert.pt
```

必须按 `export → 实验采集 → finetune` 顺序逐层推进。后续层 amplitude 依赖前一层实测 CCD，不能在尚未采集前提前导出。

## 5. 旧 16 µm 光路

旧配置和旧整数放大命令仍然保留：

```text
configs/release/caltech101_four_layer_optical_joint.yaml
--scale-factor 2
```

## 6. 快速模式：只实测第四层 Language global

这个模式不需要采集前三层。Vision expert、Vision global 和 Language expert 使用已训练
checkpoint 的仿真输出，服务器直接生成第四层的理论振幅输入；实验中只播放第四层输入
和 `language_global.bmp`。采回 CCD 后，只有最后一个 Language 电子 Mixer、第四层 CCD
readout/output adapter、第四层融合门、输出归一化和64维检索头参与微调，四组 phase、
router、前三层及 Qwen 主体保持冻结。

服务器导出：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/quick_language_global_17um_strong --stage language_global --phase export --upstream-source simulation
```

把整个 `04_language_global` 目录传到实验室电脑，重建振幅 SLM BMP：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval\hardware_sessions\quick_language_global_17um_strong\04_language_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude
```

加载本层相位文件：

```text
$STAGE\phase_to_play\language_global.bmp
```

填写 `tucam_meadowlark_1024_windows.yaml` 中的相机 ROI 和正确 LUT 后，先做预检和3帧
试采：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --input-dir "$STAGE\amplitude_to_play" --output-dir "$STAGE\ccd_captured" --phase-mask "$STAGE\phase_to_play\language_global.bmp" --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --input-dir "$STAGE\amplitude_to_play" --output-dir "$STAGE\ccd_captured" --phase-mask "$STAGE\phase_to_play\language_global.bmp" --limit 3 --clear-output
```

确认3帧后，移除 `--limit 3` 并重新使用 `--clear-output` 采集全部2,855张。把同名
478×478 uint8 PNG 上传回服务器的 `04_language_global/ccd_captured/`，然后运行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um_strong_phase/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/quick_language_global_17um_strong --stage language_global --phase finetune --upstream-source simulation --epochs 20
```

结果位于：

```text
hardware_sessions/quick_language_global_17um_strong/
├── 04_language_global/finetune_metrics.json
└── checkpoints/after_language_global.pt
```

`--upstream-source simulation` 必须在导出和微调两条命令中同时保留；去掉它会回到
完整四层顺序实验，并要求前三层实测 CCD 文件存在。

## 7. Python 环境

服务器模型和实验室硬件代码的统一可复现环境已写入：

```text
environment_server_and_lab.yml
```

安装与激活：

```bash
conda env create -f experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/environment_server_and_lab.yml
conda activate opticsmoe-four-layer
```

实验室 Windows 必须另外安装64位 Meadowlark Blink Plus PCIe 驱动/runtime 和
TUCam/Mosaic 驱动；这些是系统驱动，不能由 conda/pip 安装。正式采集前运行
`--validate-only` 检查 DLL、LUT、ROI 和全部 BMP。
