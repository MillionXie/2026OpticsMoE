# P13 guarded 16→32→64→100 growth chain

该流程只允许相邻深度增长：

```text
P11 epoch-88 best → 16 → 32 → 64 → 100
```

16 层由正式 `06_launch_growth16_fa_source_20e.sh` 产生。后续配置不提前放置占位
checkpoint，也不能跳级生成；`render_progressive_growth_config.py` 必须实际读到上一深度
的 `checkpoints/best_full_depth.pt` 后才会创建配置。

## 每一级的固定 source

| target | required parent |
| ---: | --- |
| 32 | `runs/p13_growth16_fa_source_20e_gb192/checkpoints/best_full_depth.pt` |
| 64 | `runs/p13_growth32_fa_source_20e_gb192/checkpoints/best_full_depth.pt` |
| 100 | `runs/p13_growth64_fa_source_20e_gb192/checkpoints/best_full_depth.pt` |

renderer 会验证 checkpoint format/role、source depth、alpha=1、P13 signature、model report、
migration/initialization chain、feedback method/manifest SHA、epoch、config digest 和文件 SHA。
生成配置把这些 source identity 字段全部写入 `initialization`。再次执行 renderer 时只允许
验证完全相同的配置；source 或模板变化会硬拒绝，不覆盖旧配置。

## 渲染与启动

先生成或复核 32 层配置：

```bash
TARGET_DEPTH=32 \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/12_render_or_verify_progressive_growth.sh
```

正式启动：

```bash
TARGET_DEPTH=32 PHYSICAL_GPU_INDICES=0,1,3,4 P13_ACTION=fresh \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/13_launch_progressive_growth.sh
```

launcher 在 renderer 与训练启动外层持有目标 run 的 `flock`，GPU index 必须唯一且为非负整数，PID 原子更新。每次 fresh/resume 写入独立的 UTC/action/PID 分段日志，并用对应 `*.latest.log` 软链接指向最新段，因此恢复不会截断上一段历史。

同深度中断后只能使用 `P13_ACTION=resume`。32 层形成 `best_full_depth.pt` 后，依次用相同
两条命令设置 `TARGET_DEPTH=64`，再设置 `TARGET_DEPTH=100`。直接请求 64/100 而上一深度
best 不存在时会在渲染前退出。

## 等数据预算

每一级均为 20 epoch、4 GPU、effective global batch 192：

| target | batch/rank | accumulation | global batch |
| ---: | ---: | ---: | ---: |
| 32 | 12 | 4 | 192 |
| 64 | 6 | 8 | 192 |
| 100 | 4 | 12 | 192 |

这是可执行候选配置，不等于显存已经实测通过。启动正式深度前应先运行
`05_gpu_full_depth_feedback_cuda_audit_bs1.sh` 的 alpha=1 工程审计；如需调整 batch，必须
生成新的配置/output directory，并保持 effective global batch 与优化器 update 预算一致。
