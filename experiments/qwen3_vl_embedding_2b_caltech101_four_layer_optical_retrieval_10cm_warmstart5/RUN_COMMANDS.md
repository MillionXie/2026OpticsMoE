# 运行命令

> 当前新光路使用四点 homography 直接输出 canonical 478×478 CCD。凡是实测采集与
> 微调，必须使用文件名带 `canonical_ccd` 的配置；旧 `stage2_joint_sealed_test.yaml`
> 和 `quick_last_stage_10x10.yaml` 只保留封存仿真结果及 legacy CCD 合同。

## Sim-to-real CCD agreement（quick Language-global）

完整定义见 `SIM_TO_REAL_AGREEMENT.md`。以下命令全部从统一仓库根目录执行：

```powershell
Set-Location E:\code\guest\2026OpticsMoE
```

服务器导出固定探针、实际 phase、ideal fp32 与 transport-quantized 理论 CCD：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CKPT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
AGREE=$PROJECT/hardware_sessions/agreement_quick_language_global_run1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m \
  experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_export \
  --config $PROJECT/configs/release/agreement_quick_language_global.yaml \
  --checkpoint $CKPT \
  --session-dir $AGREE \
  --stages language_global \
  --upstream-source simulation
```

将整个 `agreement_quick_language_global_run1` 复制到实验室同一项目的
`hardware_sessions/`。实验室 PowerShell：

```powershell
$PROJECT = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5"
$STAGE = "$PROJECT\hardware_sessions\agreement_quick_language_global_run1\04_language_global"

python -m experiments.hardware_sdk.workflows.reconstruct_slm `
  --stage-dir $STAGE `
  --payload amplitude `
  --hardware-profile meadowlark_17um

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" `
  --validate-only

python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml `
  --stage-dir $STAGE `
  --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" `
  --clear-output
```

执行 capture 前，先按 `experiments/hardware_sdk/GEOMETRY_AND_BRIGHTNESS.md` 用四个
逻辑标注顶点生成 detector homography contract，并在相机 YAML 中设置
`detector_geometry.enabled=true`、`contract_file` 和 `expected_file_sha256`。homography
必须在 raw device ROI 上执行并直接保存 478×478 `canonical_model_xy`；不要先普通
resize，也不要在下游再次 flip。agreement evaluator 会拒绝 legacy axis-resize 数据。

实验室或服务器均可执行严格评估和绘图；它们不加载 Qwen：

```powershell
$SESSION = "$PROJECT\hardware_sessions\agreement_quick_language_global_run1"

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate `
  --session-dir $SESSION `
  --stages language_global

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_report `
  --evaluation-dir "$SESSION\agreement_evaluation" `
  --require-arial
```

若 calibration 的方向诊断最佳项不是 `identity`，不要让程序逐图自动选择翻转；应修正
session 的固定四点 contract 和逻辑 TL/TR/BR/BL 标注，把正确 contract SHA 写回相机
YAML，然后清空本次 stage 的 `ccd_captured` 并重新采集 canonical `ccd_captured` 后再评估。
不要在旧 CCD 上离线生成一个 `ccd_registered` 目录冒充重新采集，也不要在 evaluator 中
逐帧选择最佳翻转。

所有服务器命令均从仓库根目录运行。正式训练使用 GPU 4。

实验室端使用 x64 Windows、x64 Python 3.10/3.11 和匹配的相机/Meadowlark x64
驱动；振幅 LUT 按实际温度选 30C 或 70C。厂商示例只属于 SDK 参考，不是标定或
性能证据。

## 0. CPU 配置检查

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/tests
```

### 0.0 新光路标定资产、亮度与四点几何

```powershell
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep `
  --config experiments\hardware_sdk\generators\slm_patterns\configs\dual_slm_17um_8um_normal_scale_sweep.yaml

python -m experiments.hardware_sdk.generators.fresnel_square_aperture_array `
  --config experiments\hardware_sdk\generators\slm_patterns\configs\fresnel_square_aperture_array_17um_8um.yaml

python -m experiments.hardware_sdk.workflows.roi_calibration generate `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml

python -m experiments.hardware_sdk.workflows.roi_calibration exposure `
  --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml
```

双 SLM 无倍率对齐只用生成目录中的 `00_k1_ready_to_play`；菲涅尔先试 n1/n4/n9 的
`a64` 配对。n4 四个焦点本身对称，先用 k=1 非对称 large-block/quadrant 或已知数字 `3`
确认固定方向，再据此把 n4 四点按模型 TL/TR/BR/BL 填入模板并生成合同；n9 只做独立
几何验证：

```powershell
Copy-Item experiments\hardware_sdk\configs\detector_homography_478.example.yaml `
  experiments\hardware_sdk\artifacts\calibration\detector_homography_478.lab.yaml

python -m experiments.hardware_sdk.workflows.detector_homography fit `
  --config experiments\hardware_sdk\artifacts\calibration\detector_homography_478.lab.yaml `
  --output experiments\hardware_sdk\artifacts\calibration\detector_homography_478.contract.json
```

把输出的合同路径和 file SHA 写入 `tucam_meadowlark_1024_windows.yaml`，启用
`camera.detector_geometry`。详细说明见
`experiments/hardware_sdk/GEOMETRY_AND_BRIGHTNESS.md` 与
`experiments/hardware_sdk/generators/slm_patterns/V3_CALIBRATION_COMMANDS.md`。

### 0.1 封存 formal ZIP 的历史 v2 标定资产（当前新光路勿用）

本节只用于复现旧的封存 formal ZIP，不用于当前新光路。旧包只接受 v2 ROI 顶点阵列和
normal-polarity 推荐 checker/grating 对。当前文件已
生成；以下命令用于复现或打包前复核，不调用 GPU：

```bash
python -m experiments.hardware_sdk.generators.fresnel_roi_vertex_array \
  --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_roi_vertex_array_17um_8um.yaml

python -m experiments.hardware_sdk.generators.fresnel_roi_vertex_contract \
  --calibration-dir experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_roi_vertex_array_532nm_17um_8um_v2
```

必须输出 `status=passed` 且
`calibration_manifest_sha256=88dae667691dc823c09c41e355014d75efc40457e84450073323e6b69b748a2b`。
实验顺序是：checker/grating 双 SLM 像素级对齐 → n1 找焦面 → n4 直接确定 ROI
四顶点 → n9 检查全 ROI。详见 `CALIBRATION.md`。
ROI 未知时使用 `tucam_meadowlark_calibration_windows.yaml` 保存少量完整几何标定帧；
n4 确定 ROI 后才切回 `tucam_meadowlark_1024_windows.yaml` 做正式采集。

## 1. Stage A：只训练光支路

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage1_optical_calibration.yaml --phase train
```

必须看到：

```text
[warmstart] stage=optical_calibration mode=dual_source
```

并检查：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage1_optical_calibration/warmstart_initialization_report.json
```

报告应记录两个固定 SHA、每个模态电子/光 tensor 数量、gate 未从源 checkpoint 加载，以及最终 trainable 参数数。

## 2. Stage B：低学习率联合训练

Stage A 成功生成 EMA/train-best 后执行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_sealed_test.yaml --phase train
```

必须看到：

```text
[warmstart] stage=joint mode=stage_a_checkpoint
```

训练日志中的 `test_top1` 和 `ema_test_top1` 应为 `nan`；这是 sealed-test 正常行为，不是评估故障。

## 3. 唯一一次正式测试

提交配置、代码和预声明 checkpoint 规则后，只执行一次：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_sealed_test.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
```

禁止根据该 test 结果改用同一次运行的其他 epoch。

本次固定结果为 `Top-1=0.8100`、`Top-3=0.9300`、`MRR=0.876345`；完整
证据见 `FORMAL_RESULT.md`。该命令不得对同一次运行重复用于换 checkpoint。

## 4. 导出四张正式 phase BMP

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.export_phase_bmps --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_sealed_test.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/hardware_phase_export
```

## 5. 四层实测

本工程的独立硬件入口与 10 cm robust 协议参数相同。四层顺序仍为：

```text
vision_expert -> vision_global -> language_expert -> language_global
```

先固定变量：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
MODULE=experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CONFIG=$PROJECT/configs/release/stage2_joint_hardware_canonical_ccd.yaml
BASE_CKPT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
SESSION=$PROJECT/hardware_sessions/four_layer_run1
```

每一层都严格执行：服务器 export → 实验室播放/采集 → 原路径回传 CCD/log → 服务器
finetune。只有 finetune 成功后，才能用 `after_<stage>.pt` 导出下一层。

### 5.1 Vision expert

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $BASE_CKPT --session-dir $SESSION \
  --stage vision_expert --phase export --upstream-source measured
```

实验室处理 `01_vision_expert` 后回传，再运行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $BASE_CKPT --session-dir $SESSION \
  --stage vision_expert --phase finetune --upstream-source measured --epochs 20
```

### 5.2 Vision global

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_vision_expert.pt --session-dir $SESSION \
  --stage vision_global --phase export --upstream-source measured

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_vision_expert.pt --session-dir $SESSION \
  --stage vision_global --phase finetune --upstream-source measured --epochs 20
```

中间必须完成 `02_vision_global` 的实验室采集和回传，不能连续执行上面两条命令。

### 5.3 Language expert

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_vision_global.pt --session-dir $SESSION \
  --stage language_expert --phase export --upstream-source measured

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_vision_global.pt --session-dir $SESSION \
  --stage language_expert --phase finetune --upstream-source measured --epochs 20
```

两条命令之间完成 `03_language_expert` 的采集和回传。

### 5.4 Language global

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_language_expert.pt --session-dir $SESSION \
  --stage language_global --phase export --upstream-source measured

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m $MODULE.hardware_bridge \
  --config $CONFIG --checkpoint $SESSION/checkpoints/after_language_expert.pt --session-dir $SESSION \
  --stage language_global --phase finetune --upstream-source measured --epochs 20
```

两条命令之间完成 `04_language_global` 的采集和回传。最终硬件微调 checkpoint 是：

```text
$SESSION/checkpoints/after_language_global.pt
```

### 5.5 每一层在实验室电脑上的固定命令

把服务器生成的当前 stage 目录放在统一项目目录下，例如
`experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\four_layer_run1\01_vision_expert`，然后只用相对路径：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\four_layer_run1\01_vision_expert"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --limit 3 --clear-output
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --clear-output
```

先手动加载 `$STAGE\phase_to_play\<stage>.bmp`。试拍 3 张确认曝光/方向/ROI 后才全量
采集。回传 `$STAGE\ccd_captured\` 和 `$STAGE\acquisition_logs\`；不回传可重建的
`amplitude_to_play`。`device_roi_xywh` 四个值必须为 4 的倍数，设备侧不能直接设
478；采集工作流严格执行 raw device ROI → 一次 homography → 固定范围 uint8，输出
同 basename 的 canonical 478×478 灰度 PNG。相位 BMP 已围绕配置中心纵翻、播放不再
翻；CCD 已由 homography 统一方向，模型端横纵 flip 均为 false。`--validate-only` 只静态检查 runtime、文件和
ROI，不打开设备，不能代替连接、曝光和成像试拍。

## 6. 快速验证最后一层

quick 配置明确为每类 10 train + 10 test + 1 gallery，总计 210 张，不继承 3 gallery。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/quick_last_stage_10x10_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/quick_language_global_run1 --stage language_global --phase export --upstream-source simulation
```

实验室重建和采集：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\quick_language_global_run1\04_language_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --file-manifest "$STAGE\amplitude_to_play\reconstruction_manifest.csv" --clear-output
```

上传同 basename 的 478×478 uint8 灰度 CCD 后：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/quick_last_stage_10x10_canonical_ccd.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/quick_language_global_run1 --stage language_global --phase finetune --upstream-source simulation --epochs 10
```

## 7. 轻量离线第四层微调

新导出的 quick210 session 会同时写入 `offline_downstream/cache.pt`、
`downstream_state.pt` 和 `contract.json`。实验室电脑采完 210 张 CCD 后，不加载 Qwen
即可校验和微调约 255,811 个末层电子参数：

```powershell
$SESSION = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\quick_language_global_run1"

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir $SESSION `
  --device cpu `
  --validate-only

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.offline_quick_finetune `
  --session-dir $SESSION `
  --device auto `
  --epochs 10
```

## 8. 当前新光路：生成独立标定与 agreement 实验室 ZIP

当前入口不读取 checkpoint、不改动封存 formal ZIP，并包含 Fresnel v3、k=1 双 SLM、
四点 homography、32×3 亮度标定、采集和多指标 agreement 工具：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package --check-only

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package --overwrite

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package \
  --verify-zip experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/validation_bundles/qwen_warmstart5_optical_validation_lab_tools.zip
```

实验室当前应拿 `validation_bundles/qwen_warmstart5_optical_validation_lab_tools.zip`，
并以 `LAB_VALIDATION_BUNDLE.md` 为操作入口。

## 9. 封存历史 81% formal ZIP（当前新光路勿用）

以下命令仅用于逐字节复现旧的 checkpoint/quick210/Fresnel v2 封存包。它不是当前
Fresnel v3、canonical CCD 或 sim-to-real agreement 的实验室入口。

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
MODULE=experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
CKPT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
SESSION=$PROJECT/hardware_sessions/quick_language_global_run1
FRESNEL=experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_roi_vertex_array_532nm_17um_8um_v2
DUAL_SLM=experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment_normal_polarity/recommended_checker_grating_pair
FIXED_REPORT=$PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/fixed_simulation_report

python -m $MODULE.result_report \
  --root . \
  --baseline-json $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/metrics/evaluation_summary.json \
  --output-dir $FIXED_REPORT \
  --require-arial

python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_package \
  --checkpoint $CKPT \
  --phase-export-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/hardware_phase_export \
  --quick-session-dir $SESSION \
  --stage-a-run-dir $PROJECT/runs/caltech101_warmstart5_stage1_optical_calibration \
  --stage-b-run-dir $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test \
  --fresnel-calibration-dir $FRESNEL \
  --dual-slm-calibration-dir $DUAL_SLM \
  --fixed-simulation-report-dir $FIXED_REPORT \
  --output $PROJECT/runs/caltech101_warmstart5_stage2_joint_sealed_test/qwen_caltech101_10cm_warmstart5_quick210_lab_bundle.zip \
  --overwrite
```

打包器只接受封存 checkpoint SHA
`6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d`，并会拒绝架构
不为 `vision2_language2_moe4_10cm_warmstart5_stage_b_v1`、使用 test 选轮、固定指标/Arial
报告不一致、或 checkpoint/phase/transport/offline/标定 SHA 不一致的输入。
实验室完整步骤与服务器/实验室职责边界见 `LAB_BUNDLE.md`。
