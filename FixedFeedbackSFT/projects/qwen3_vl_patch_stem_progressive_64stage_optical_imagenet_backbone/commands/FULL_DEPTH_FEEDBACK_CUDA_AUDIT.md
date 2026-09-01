# P13 64/100-stage full-depth feedback CUDA audit

## 范围

该命令只验证深层 fixed-feedback 的工程可执行性，不启动正式训练：

```bash
P13_GPU=1 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/05_gpu_full_depth_feedback_cuda_audit_bs1.sh
```

它产生六个相互隔离的组合：

```text
64  x bp_current
64  x fa_source
64  x fa_random
100 x bp_current
100 x fa_source
100 x fa_random
```

每个组合重新实例化模型、严格迁移相同 P11 source，并显式设置
`alpha_mode=full_depth`，因此 carried 与 new 的每个 stage 都以 `alpha=1` 执行。随后用
相同合成 post-adapter 光场执行 warmup、独立梯度 audit 和计时 optimizer step。
`fa_random` 默认 base seed 为 `20260901`。

## 输出

```text
runs/p13_full_depth_feedback_cuda_audit_bs1/
  sweep_summary.json
  depth_064/feedback_bp_current/result.json
  depth_064/feedback_fa_source/result.json
  depth_064/feedback_fa_random/result.json
  depth_100/feedback_bp_current/result.json
  depth_100/feedback_fa_source/result.json
  depth_100/feedback_fa_random/result.json
```

每个 result 都记录：

- method、random seed、campaign/combination SHA；
- PyTorch/CUDA 版本与完整光学执行路径 implementation SHA；
- initial/final feedback manifest 及其 canonical JSON SHA-256；
- P11/stem SHA、migration、alpha 和参数预算；
- 全部 carried+new phase 梯度是否 present、finite、nonzero；
- 输入 amplitude 梯度是否 present、finite、nonzero，以确认梯度贯穿完整光学 body；
- peak allocated/reserved、step seconds 和 samples/s；
- GPU UUID，防止跨设备混表。

## 中断恢复

首次运行不允许覆盖已有文件。中断后使用：

```bash
P13_GPU=1 RESUME_EXISTING=1 \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/05_gpu_full_depth_feedback_cuda_audit_bs1.sh
```

恢复只复用 identity 完全一致且状态为 `passed_engineering` 的组合。配置、source、GPU、
depth、method 或 random seed 任一不同都会拒绝恢复。已有失败/OOM 组合会在同一 identity
下重新运行并原子替换，不会被当成通过结果。

## 解释边界

`passed_engineering` 只说明该组合在 `alpha=1` 下完成了少量合成光场反向步骤，全部 phase
和输入 amplitude 梯度通过工程检查。
它不说明 ImageNet 精度、优化收敛、深层 FA 等价于 BP，也不验证 100 个无源物理平面。
