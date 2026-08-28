# 当前命令入口（历史命令已移除）

实验室只看：

```text
experiments/lab_qwen/COMMANDS.md
```

该文件按顺序包含：需要修改的硬件/四点配置、双 SLM 对齐、Fresnel 距离与 ROI、
32 灰度×3 帧曝光标定、sim-to-real agreement、末层 quick210、四层逐层采集与微调、
结果绘图。旧 Fresnel v2/v3、旧极性、旧长路径和旧轻量 ZIP 均不再作为入口。

## 服务器重建完整 ZIP

在仓库根目录执行；`FOUR` 必须是已经导出第一层 `vision_expert` 的 session：

```bash
PROJECT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5
FORMAL=$PROJECT/lab_bundles/qwen_caltech101_10cm_warmstart5_quick210_lab_bundle.zip
FRESNEL=experiments/hardware_sdk/generators/slm_patterns/generated/fresnel_full_panel_532nm_17um_8um
DUAL=experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_normal_large_blocks_k0p1/00_k1_ready_to_play
AGREE=$PROJECT/hardware_sessions/agreement_quick_language_global_run1
FOUR=$PROJECT/hardware_sessions/four210_run1
OUTPUT=$PROJECT/lab_full_bundles/qwen_full_lab.zip

python -m experiments.hardware_sdk.generators.fresnel_full_panel_array \
  --config experiments/hardware_sdk/generators/slm_patterns/configs/fresnel_full_panel_17um_8um.yaml
python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep \
  --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um_normal_scale_sweep.yaml
python -m experiments.hardware_sdk.workflows.roi_calibration generate \
  --config experiments/lab_qwen/config/hardware.yaml

python -m experiments.lab_qwen.full_lab_package --create \
  --repo-root . --formal-zip $FORMAL --fresnel-dir $FRESNEL --dual-dir $DUAL \
  --agreement-session $AGREE --four-session $FOUR --output $OUTPUT --overwrite
python -m experiments.lab_qwen.full_lab_package --verify-zip $OUTPUT
sha256sum $OUTPUT
```

完整 ZIP 的实验室目标路径固定为短路径：

```text
E:\code\guest\2026OpticsMoE\experiments\lab_qwen
```
