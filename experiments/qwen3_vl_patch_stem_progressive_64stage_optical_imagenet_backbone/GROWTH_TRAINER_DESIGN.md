# P13 正规 ImageNet growth trainer 实现记录

## 实现目标

本轮把 P13 从“迁移/显存工程原型”补成可执行的 progressive ImageNet 训练流程，但没有启动正式训练。训练器的首个正式候选是 P11 epoch-88 best 到 16 stage 的 20-epoch growth；后续沿用相同 checkpoint 合约支持 16→32→64→100。

## Fresh migration 与 resume 分离

CLI 强制选择 `--fresh` 或 `--resume`：

- fresh 只执行 `p11_to_16` 或 `progressive_growth` 严格迁移；目标目录只要已有 manifest、initial phases、result、checkpoint、metric 或相应临时文件，就在 dataset 构造前拒绝覆盖；
- resume 只读取同一 output 的 `last.pt`，严格核验 checkpoint format、config digest、完整 model config/depth、world size、逐 rank RNG 数量与当前实现哈希；不再次调用 migration；
- P11→16 同时读取正式 `backbone.pt` 和 `best.pt`，除 epoch=88 和 tensor/stem/head 交叉校验外，还硬锁两文件 SHA-256、官方 config digest 以及会影响 P11 重构的结构配置；
- 下一深度只接受 `best_full_depth.pt` 和 16→32→64→100 相邻转换，并要求 renderer 写入且训练器复核 source checkpoint SHA、depth、epoch、config digest、feedback method 与 feedback manifest SHA。

`migration_manifest` 是审计来源，不是模型 tensor，因此不会进入 `state_dict`。fresh 和 same-depth resume 都会从 `initialization_manifest` 显式安装该来源，并交叉核验 wrapper source/target depth、manifest target depth、模型持久化架构签名 `[13,1,2,depth]` 与 checkpoint 顶层副本；任一项不一致即拒绝训练。保存 checkpoint 时再次断言运行时来源没有漂移，所以 resume 后的 `model_report` 仍保留完整迁移链。

## Fixed-feedback 恢复保护

P13 stage 内的 runtime feedback buffer 不进入 state dict，模型 load hook 会回到 `bp_current`。训练器因此把以下内容写入每个 checkpoint：

- feedback method；
- fa-random base seed；
- 完整 feedback manifest；
- 完整 manifest 自身的 SHA-256；
- 包含 active phase 的 exact-resume SHA-256；
- 忽略 BP 动态 phase、但锁定 source/transfer/growth mapping/seed/mode 的 runtime-contract SHA-256。

resume 顺序固定为：加载 model state → 从完整 manifest 恢复非 tensor 的 feedback-source provenance → 按 checkpoint/config 显式调用 `configure_feedback` → 重新生成 manifest → 核验完整 manifest SHA 与两个 connector SHA。method、seed、source provenance 或任意 connector/source/transfer hash 不一致都会终止，不能静默以 BP 继续 FA run。每个 epoch 前后也检查 runtime contract。

每个 connection 还直接记录对应 `stage.feedback_mode`。训练器按全局方法逐 slot 要求 `bp`、`fa_pretrained` 或 `fa_random`；即便只把中间一层切回 BP、全局 `feedback_method` 没变，也会在 configure、epoch guard 或 checkpoint 前立即报错。

## 实现与运行库身份

启动时逐文件 SHA-256 锁定 P13 train/model/migration、P11 matched/separable/slim/base model、stem、P11 loader utilities、optics、sampler 和 ImageNet dataset/settings。另记录 Python、Torch、Torchvision、Torch CUDA build、cuDNN 与影响数值路径的 cuDNN/TF32 flags。完整清单及聚合 SHA 同时进入 `manifest.json`、所有训练 checkpoint、`result.json` 和 backbone export。

same-depth resume 要求当前逐文件哈希和运行库字段与 `last.pt` 完全相同；dirty worktree 改动也会被识别。每次保存 checkpoint 和最终 export 前会重新计算一次，运行中途改代码同样会终止。

## 优化器与更新语义

训练器要求模型提供并严格穷尽所有 trainable parameter：

1. `new_phase`；
2. `carried_phase`；
3. `new_electronic`；
4. `carried_electronic`；
5. `head`。

重复归组或漏掉任意 trainable parameter 都会报错。新增 phase 默认 LR `7e-3`，carried phase 为 `3.5e-3`，没有把光学 LR 缩得过低；每组参数量、LR 和 weight decay 写入 manifest/checkpoint。

支持 FP16/BF16 autocast；FP16 使用 GradScaler。gradient accumulation 的最后一个不完整窗口按实际 micro-batch 数归一化。DDP 非更新 micro-batch 在 forward+backward 外层使用 `no_sync()`。scheduler 只在 GradScaler 真正执行 optimizer update 后 step；显式保存并核验 `scheduler.last_epoch == global_optimizer_step`。

首次 optimizer update 会检查 new/carried phase 梯度是否全部 present、finite、nonzero。

## RNG 与数据连续性

checkpoint 保存每个 rank 的：

- Python `random`；
- NumPy RNG；
- Torch CPU RNG；
- 当前 CUDA device RNG。

resume 要求 world size 不变，并在 loader 第一次迭代前恢复对应 rank RNG。ImageNet dataset 的 augmentation 本身由 sample/view index 确定，epoch sampler 也显式 `set_epoch`，因此 worker 重启不会依赖未保存的持续 worker 随机状态。

train/validation DataLoader 现在各有按 split 与 rank 派生的独立 `torch.Generator`。新进程启动 worker 时产生 base seed 只推进该 loader generator，不再推进恢复后的主 Torch RNG；所以 uninterrupted persistent-worker 路径与 split resume 的下一次 mixup/cutmix `randperm` 轨迹一致。全部 rank 的 loader seeds 写入 manifest。

## Alpha 与 checkpoint 选择

- `best_any.pt`：包括 alpha<1 过渡阶段，只用于分析；
- `last.pt`：唯一 same-depth resume 入口；
- `best_full_depth.pt`：只允许全部新增 alpha 精确等于 1 时更新；
- `backbone_full_depth.pt`：只允许从 `best_full_depth.pt` 加载并再次核验 alpha=1 后导出。

若整个 run 没有产生 alpha=1 checkpoint，训练器明确拒绝 export。16-stage 20e config 在 epoch 1 使用 epsilon=0.02，在 epoch 10 达到 alpha=1。

## 五卡并行方案与 matched control

正式候选配置：

- P13：4 GPU × batch 24/rank × accumulation 2 = global batch 192；
- P11 control：1 GPU × batch 96 × accumulation 2 = global batch 192。

两组均运行 20 个新 epoch，使用相同 dataset、augmentation、seed、loss、warmup/cosine、gradient clip 和 carried/head LR。P11 control 同时核验并加载 epoch-88 `backbone.pt`/`best.pt`，但不沿用旧 optimizer/scheduler；二者都以新的公平 20-epoch 预算开始。P13 新插入参数使用显式 new-group LR，这是 growth 方法的一部分。

所有启动、恢复和 watch/status 命令见 [commands/GROWTH_TRAINING_COMMANDS.md](commands/GROWTH_TRAINING_COMMANDS.md)。

正式 launcher 使用 `flock` 持有整个训练进程生命周期的 run lock，GPU index 必须为唯一非负整数；PID 通过临时文件原子替换。每次 fresh/resume 创建带 UTC、action、launcher PID 的独立日志，并让 `*.latest.log` 软链接指向最新段，resume 不再截断历史。

## 自动验证

新增训练器测试覆盖：

- 五组参数无重复、无遗漏；真实 16-stage 模型为 8 carried + 8 new phase；
- accumulation=3、5 个 micro-batch 时只执行 2 次 optimizer/scheduler update，DDP `no_sync` 调用 3 次；
- scheduler update counter 对齐；
- fa-random mode/seed/manifest 恢复与篡改拒绝；
- Python/NumPy/Torch RNG round trip，以及 loader restart 不改变 split-resume mixup/cutmix 随机轨迹；
- dirty implementation/runtime manifest 不允许 resume；
- 单个 optical slot 静默切回 BP 会被拒绝；
- fresh 对首个 checkpoint 前遗留的 manifest/metric 也拒绝；
- alpha<1 不能进入 `best_full_depth`；
- P11 双 checkpoint 官方 SHA/config/epoch 与结构配置前置检查；
- CPU 端到端 fresh baseline→训练→best_any/best_full/last→alpha=1 export→same-depth resume，并断言 resume 没有重新调用 fresh migration。

最终本地命令：

```text
python -m pytest -q experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/tests
49 passed in 24.18s
```

同时通过 `py_compile`、三份 YAML load/validation 和 `git diff --check`。Windows 环境没有可用 bash，因此 shell `bash -n` 应在同步到服务器后作为启动前最后一道检查。

## 尚未验证

- 尚未执行真实 ImageNet full-image GPU smoke；
- batch=24/rank 是待 smoke 验证的候选，不是已锁定吞吐配置；
- 尚未报告 P13 ImageNet 精度、训练耗时或显存；
- P11 matched continuation 仍需与 P13 同时启动后才形成公平结果；
- 32/64/100 的正规训练 config 应在前一深度 `best_full_depth.pt` 产生后再生成，不能提前伪造 source。
