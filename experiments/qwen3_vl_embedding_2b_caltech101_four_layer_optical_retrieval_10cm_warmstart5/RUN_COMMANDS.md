# 运行命令

所有服务器命令均从仓库根目录运行。正式训练使用 GPU 4。

## 0. CPU 配置检查

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/tests
```

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

## 4. 导出四张正式 phase BMP

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.export_phase_bmps --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_sealed_test.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/hardware_phase_export
```

## 5. 四层实测

本工程的独立硬件入口与 10 cm robust 协议参数相同。四层顺序仍为：

```text
vision_expert -> vision_global -> language_expert -> language_global
```

第一层导出示例：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/stage2_joint_sealed_test.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/four_layer_run1 --stage vision_expert --phase export --upstream-source measured
```

后续重建、采集、微调命令与 robust 工程一致，只把 Python module、config、checkpoint 和 session-dir 换成本工程路径。

## 6. 快速验证最后一层

quick 配置明确为每类 10 train + 10 test + 1 gallery，总计 210 张，不继承 3 gallery。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/quick_last_stage_10x10.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/quick_language_global_run1 --stage language_global --phase export --upstream-source simulation
```

实验室重建和采集：

```powershell
$STAGE = "experiments\qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5\hardware_sessions\quick_language_global_run1\04_language_global"
python -m experiments.hardware_sdk.workflows.reconstruct_slm --stage-dir $STAGE --payload amplitude --hardware-profile meadowlark_17um
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --validate-only
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_meadowlark_1024_windows.yaml --stage-dir $STAGE --clear-output
```

上传同 basename 的 478×478 uint8 灰度 CCD 后：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/configs/release/quick_last_stage_10x10.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/hardware_sessions/quick_language_global_run1 --stage language_global --phase finetune --upstream-source simulation --epochs 10
```
