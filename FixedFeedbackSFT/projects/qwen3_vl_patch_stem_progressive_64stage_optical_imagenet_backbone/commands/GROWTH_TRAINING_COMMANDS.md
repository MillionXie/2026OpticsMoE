# P13 ImageNet progressive-growth training commands

本文件只记录可复现命令；当前代码实现与测试阶段没有自动启动正式训练。

## 1. 训练状态机

每次调用必须显式选择一种模式：

- `--fresh`：只允许目标目录没有任何既有 manifest、initial phases、result、checkpoint、metric 或相应临时文件，并执行严格 source migration；
- `--resume`：只允许目标目录已有 `checkpoints/last.pt`，执行同深度 strict state load，不再调用 migration。

这两个路径不能自动互相回退。命令脚本对应使用 `P13_ACTION=fresh|resume` 或 `P11_ACTION=fresh|resume`。

P13 fresh 支持：

1. `p11_to_16`：同时读取 P11 `backbone.pt` 和 epoch-88 `best.pt`，硬锁官方两文件 SHA-256、config digest、P11 结构配置，并交叉核验非 head 权重、stem、epoch，再迁移完整 ImageNet readout；
2. `progressive_growth`：只接受 `16->32->64->100` 相邻增长，source 必须是本训练器生成的 `best_full_depth.pt`，并逐项匹配 renderer 锁定的 source SHA/depth/epoch/config/feedback method/feedback manifest SHA。

## 2. 先运行真实 ImageNet 全图 smoke

smoke 会在一张卡上读取真实 224x224 ImageNet batch，执行两次 micro-batch、一次 optimizer update、验证和 alpha=1 export。它不是合成 field 测试，也不是性能结果。

```bash
P13_GPU=4 P13_ACTION=fresh \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/05_gpu_smoke_full_image.sh
```

检查项：

- 双 P11 checkpoint 均锁定 epoch 88；
- `feedback.method=fa_source`，checkpoint 中存在 feedback manifest/hash；
- `new_phase`、`carried_phase` 梯度存在且 finite/nonzero；
- `optimizer_updates=1`，scheduler 只前进一步；
- `best_any.pt`、`best_full_depth.pt`、`last.pt` 和 `backbone_full_depth.pt` 均可读取；
- peak memory 为后续 batch=24/rank 配置留有余量。

若 smoke 已有旧目录，不允许删除或覆盖来“重跑”。应先保留旧结果并修改 config 的 `output_dir`，或明确使用：

```bash
P13_GPU=4 P13_ACTION=resume bash .../05_gpu_smoke_full_image.sh
```

## 3. P13 8->16、20 epoch 正式候选命令

正式 config 使用四张 GPU：每 rank batch=24、gradient accumulation=2，因此 effective global batch 为 `24*4*2=192`。alpha 在 epoch 1 从 0.02 启动，在 epoch 10 精确达到 1；只有 epoch 10 及以后才有资格写 `best_full_depth.pt` 和导出 backbone。

首次运行：

```bash
PHYSICAL_GPU_INDICES=0,1,3,4 P13_ACTION=fresh \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/06_launch_growth16_fa_source_20e.sh
```

同深度断点恢复：

```bash
PHYSICAL_GPU_INDICES=0,1,3,4 P13_ACTION=resume \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/06_launch_growth16_fa_source_20e.sh
```

状态与持续观察：

```bash
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/07_status_growth16.sh
INTERVAL_SECONDS=30 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/08_watch_growth16.sh
```

launcher 用物理 index 查询 GPU UUID 后设置 `CUDA_VISIBLE_DEVICES`，避免 index 重映射歧义；index 必须是唯一非负整数。`flock` 锁由训练进程继承并持有到退出，消除并发 launcher 的 check-then-write 竞态，PID 文件原子更新。

每次调用生成独立的 `*.UTC.action.launcherPID.log`，`*.latest.log` 软链接供 status/watch 使用。resume 会保留所有旧日志，不会再用 `>` 截断历史。

## 4. P11 epoch-88 matched continuation control

该控制不是从随机 P11 重训，也不是接 P11 的 `last.pt`：它同时读取正式 `backbone.pt` 和 `best.pt`，严格要求二者均为 epoch 88 且非 head tensor 完全相同，然后加载完整 epoch-88 model/head。

它重新初始化 optimizer/scheduler，运行相同 20 epoch 新预算；数据、augmentation、seed、loss、warmup/cosine、gradient clip 和 effective global batch 与 P13 一致。单卡 batch=96、accumulation=2，故 global batch 同样为 192；carried phase/electronic/head LR 与 P13 对应 carried group 相同。P13 新插入参数采用自己的 new-group LR，这是 growth 方法的一部分，会显式记录而不伪装成完全相同模型。

可以把第五张空卡同时用于该控制：

```bash
P11_CONTROL_GPU=5 P11_ACTION=fresh \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/09_launch_p11_matched_continue_20e.sh
```

恢复和观察：

```bash
P11_CONTROL_GPU=5 P11_ACTION=resume \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/09_launch_p11_matched_continue_20e.sh

bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/10_status_p11_matched.sh
INTERVAL_SECONDS=30 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/11_watch_p11_matched.sh
```

这样四卡跑 P13、一卡跑 P11，可以同时使用五张卡，并保持两组 optimizer update 所见的 global batch 都为 192。

## 5. checkpoint 语义

- `last.pt`：唯一 same-depth resume 输入；
- `best_any.pt`：包括 alpha<1 过渡期在内的最高验证精度，只用于分析；
- `best_full_depth.pt`：只在所有新增 alpha 精确等于 1 时更新，是下一深度 migration 的 source；
- `backbone_full_depth.pt`：只从 `best_full_depth.pt` 导出，不含临时 ImageNet readout。

每个训练 checkpoint 保存：完整 model/readout、model config/report、optimizer 五组、scheduler、AMP scaler、global optimizer update、每 rank Python/NumPy/Torch/CUDA RNG、initial phase hash、growth migration、alpha、phase motion，以及 feedback method/seed/full manifest/完整 manifest SHA/恢复哈希。

`load_state_dict` 后模型内部 runtime feedback 会先回到 BP，且非 tensor 的 source provenance 会进入显式“待外部恢复”状态；训练器随后从 checkpoint manifest 恢复 provenance、按 checkpoint 重新配置，并逐项核验完整 manifest 与 connector/source/transfer/seed hash。每个 connection 记录真实 `stage.feedback_mode`，逐层 guard 可发现单层退回 BP。任何不匹配都会终止。

`implementation_manifest` 还会进入 run manifest、每个 checkpoint、result 与 export：它逐文件锁定 P13/P11/optics/sampler/dataset 实现，并锁定 Python、Torch、Torchvision、CUDA build、cuDNN 和数值 flags。resume 必须与当前 dirty-worktree 文件及运行库完全一致。

train/validation loader 使用 split/rank 专属 generator；worker 创建不再消耗主训练 Torch RNG，因此重启后的 mixup/cutmix 轨迹不会因为 persistent workers 被重新创建而偏移。

## 6. batch 调整规则

`batch=24/rank` 只是 smoke 前的候选。若真实 full-image smoke 证明显存不足，可以降低 per-rank batch 并提高 accumulation，但必须同时满足：

1. P13 与 P11 control 的 effective global batch 相同；
2. 两个 config 的 `expected_effective_global_batch` 同步更新；
3. 形成新 config/output_dir，不修改已经开始的 run；
4. scheduler 仍按 optimizer update，而不是 micro-batch 计数。
