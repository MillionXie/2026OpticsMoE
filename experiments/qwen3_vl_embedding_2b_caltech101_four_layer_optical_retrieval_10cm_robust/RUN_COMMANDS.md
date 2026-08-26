# 训练与四层硬件命令

所有服务器命令都从仓库根目录执行。实验室命令使用 Windows PowerShell，也从仓库根目录执行。

正式配置：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml
```

正式 checkpoint：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt
```

以下示例使用物理 GPU 4。

## 1. 可选：准备数据

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --phase prepare_data
```

## 2. Smoke测试

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/smoke/four_layer_optical_17um_10cm_robust_smoke.yaml --phase train
```

单元测试：

```bash
python -m pytest experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/tests -q
```

## 3. 从头联合训练

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --phase train
```

这条命令不加载旧 checkpoint。Qwen冻结，电子 Mixer、四组阶段phase、两个router、CCD readout、四个有下限的融合门和64维检索头从头联合训练。当前正式checkpoint architecture为独立518面阵错位语义的`..._v2`，不要用修复前`..._v1`恢复训练。

若进程中断，仅在确认 checkpoint 来自同一个新工程和同一配置后恢复：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/last_checkpoint.pt
```

### 3.1 只有1～2小时：随机均衡快速训练

快速配置每个epoch只训练15个均衡batch，即`15×(10类×3图)=450`次样本出现，
约为完整训练集的17%。采样器用绝对epoch重新设种子，所以各轮会重新洗牌并继续
覆盖不同样本，不是固定450张。25个短epoch保留逐轮仿真测试、EMA、phase-focus、
router loss、phase dropout、三路错位及k-space。按RTX 4090实测训练步和双评估
开销拆分后预计约93～99分钟。

从已有同架构`last_checkpoint.pt`继续（当前推荐）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust_fast_2h.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/last_checkpoint.pt
```

若没有任何checkpoint，也可以用同一快速配置从头联合训练；去掉
`--resume-checkpoint ...`即可。快速配置只改变附加epoch数和每轮优化步数，不改变
模型、loss、学习率、batch组成、10 cm传播或鲁棒性参数。

本次快速运行在绝对epoch 5～7观察到Language硬top-2连续只覆盖2/4专家，因而从
epoch 7 checkpoint开始使用以下“剩余20轮”配置。它加入训练态`noise_std=0.10`
和`0.05×router balance loss`；测试、BMP导出和实际硬件路由没有噪声。该配置还把
续训相对warmup设为0，使phase-focus继续落在绝对epoch 8、10、…、26：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust_fast_remaining20.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/last_checkpoint.pt
```

## 4. 固定checkpoint仿真评估

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt
```

## 5. 一次性导出四张相位BMP

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.export_phase_bmps --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/hardware_phase_export
```

输出：

```text
hardware_phase_export/
├── compact_phase/       # 四张478×478 PNG
├── phase_bmp/           # 四张1920×1200、8-bit灰度BMP
├── phase_preview.png
└── phase_export_report.json
```

## 6. 重新生成正常极性的双SLM标定图

输入棋盘格为正常振幅极性，白格255透光、黑格0遮光；相位光栅只位于白格。

```bash
python -m experiments.hardware_sdk.generators.dual_slm_alignment --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml
```

大块图案和±0.1相位倍率扫描：

```bash
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um_normal_scale_sweep.yaml
```

1、4、9个焦点的菲涅尔阵列（532 nm，5/10/15 cm）也必须按修正后的极性重新生成：

```bash
python -m experiments.hardware_sdk.generators.fresnel_phase_array --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_phase_array_17um_8um.yaml
```

修正后的硬件中，`0` 是遮光而不是照明，所以菲涅尔标定必须播放正常目录中的
`amplitude_uniform_white_1024x1024.bmp`（全255）。旧目录中的
`amplitude_all_zero_1024x1024.bmp` 会把光关掉，不能用于聚焦或ROI标定。

推荐使用独立目录中唯一的一对文件，避免与历史 primary/complement 文件错配：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment_normal_polarity/recommended_checker_grating_pair/
├── amplitude_checker_255open_c64_1024x1024.bmp
└── phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp
```

不要使用旧 `_inv` 或 `inverted` 目录。

服务器上可直接下载的三个已校验ZIP为：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment_normal_polarity/recommended_checker_grating_pair.zip
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_normal_large_blocks_k0p1.zip
experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_phase_array_532nm_17um_8um_normal_polarity.zip
```

## 7. 实验室每层通用采集命令

每一层先在PowerShell中把 `$STAGE` 指向该层目录。例如第一层：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\hardware_sessions\four_layer_10cm_robust_run1\01_vision_expert"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

重建正常极性的振幅 BMP：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
```

服务器的 `export` 已生成正确中心的 `phase_to_play/*.bmp`。如果实验室只收到 `compact_phase`，使用：

```powershell
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload phase --hardware-profile meadowlark_17um --center-x 980 --center-y 590
```

先填写 `tucam_meadowlark_1024_windows.yaml` 中真实 LUT、曝光和 `camera.device_roi_xywh`，再做只读预检：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
```

手动在相位 SLM 上加载 `$STAGE\phase_to_play\` 中唯一的 BMP。程序会核对文件及SHA256，但不会代替操作者控制相位 SLM。

首次采3张：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
```

确认极性、曝光、ROI、相位文件和basename都正确后，清除3张试拍并采集全部样本：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

把完整 `ccd_captured/*.png` 和 `acquisition_logs/` 上传回服务器同一层目录。上传后不要再次在该目录运行 `--clear-output`。

## 8. 正式四层顺序实验

正式 session：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1
```

### 8.1 第一层：Vision expert

服务器导出：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage vision_expert --phase export --upstream-source measured
```

实验室设置并执行第7节通用命令：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\hardware_sessions\four_layer_10cm_robust_run1\01_vision_expert"
```

上传 CCD 后，服务器微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage vision_expert --phase finetune --upstream-source measured --epochs 20
```

输出：`checkpoints/after_vision_expert.pt`。

### 8.2 第二层：Vision global

服务器导出，必须读取第一层微调 checkpoint 和第一层实测 CCD：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage vision_global --phase export --upstream-source measured
```

实验室：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\hardware_sessions\four_layer_10cm_robust_run1\02_vision_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

执行第7节重建、预检、3张试拍和全量采集。上传后微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage vision_global --phase finetune --upstream-source measured --epochs 20
```

输出：`checkpoints/after_vision_global.pt`。

### 8.3 第三层：Language expert

服务器导出：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1/checkpoints/after_vision_global.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage language_expert --phase export --upstream-source measured
```

实验室：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\hardware_sessions\four_layer_10cm_robust_run1\03_language_expert"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

执行第7节通用采集并上传 CCD，然后微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1/checkpoints/after_vision_global.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage language_expert --phase finetune --upstream-source measured --epochs 20
```

输出：`checkpoints/after_language_expert.pt`。

### 8.4 第四层：Language global

服务器导出：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1/checkpoints/after_language_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage language_global --phase export --upstream-source measured
```

实验室：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\hardware_sessions\four_layer_10cm_robust_run1\04_language_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

执行第7节通用采集并上传 CCD，然后最终微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1/checkpoints/after_language_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/four_layer_10cm_robust_run1 --stage language_global --phase finetune --upstream-source measured --epochs 20
```

最终结果：

```text
hardware_sessions/four_layer_10cm_robust_run1/
├── 04_language_global/finetune_metrics.json
└── checkpoints/after_language_global.pt
```

## 9. 最后一层快速验证

快速模式必须使用独立 session，避免与四层正式 CCD 混用：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/quick_language_global_10cm_robust_run1
```

服务器用前三层仿真直接导出第四层理论输入：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/quick_language_global_10cm_robust_run1 --stage language_global --phase export --upstream-source simulation
```

实验室：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust\hardware_sessions\quick_language_global_10cm_robust_run1\04_language_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

执行第7节的振幅重建、预检、3张试拍和全量采集。上传第四层 CCD 后：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/runs/caltech101_four_layer_moe4_joint_17um_10cm_robust/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/hardware_sessions/quick_language_global_10cm_robust_run1 --stage language_global --phase finetune --upstream-source simulation --epochs 20
```

快速结果：

```text
hardware_sessions/quick_language_global_10cm_robust_run1/
├── 04_language_global/finetune_metrics.json
└── checkpoints/after_language_global.pt
```

该结果只代表“前三层仿真+第四层实测”。
