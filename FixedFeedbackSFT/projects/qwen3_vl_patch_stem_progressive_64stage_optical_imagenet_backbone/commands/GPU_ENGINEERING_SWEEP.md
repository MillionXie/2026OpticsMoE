# P13 GPU 工程 sweep（非正式训练）

这组命令只回答三个工程问题：真实 P11 source 能否严格迁移到指定深度、batch=1 时一次
forward/backward/SGD step 的显存与速度是多少、全部 phase 与输入 amplitude 是否都得到
finite 且非零梯度。
它使用合成的 `3 x 224 x 224` post-adapter 光场，不读取 ImageNet，也不输出准确率；
`passed_engineering` 不能写成 backbone 性能结论。

## 单卡顺序 sweep

```bash
P13_GPU=1 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03_gpu_engineering_sweep_bs1.sh
```

默认依次测 `16,32,64,100`，且每个深度分别独立运行 `bp_current`、`fa_source`、
`fa_random`。每个组合都从同一个正式 P11 `backbone.pt` 重新迁移，绝不继承前一个
method 的 optimizer step。若某个组合 OOM，会原子写入 `failed_oom`、释放模型和
CUDA cache，然后继续下一个组合。

## 四张卡并行分深度

```bash
P13_GPU=1 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03a_gpu_engineering_depth16_bs1.sh
P13_GPU=3 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03b_gpu_engineering_depth32_bs1.sh
P13_GPU=4 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03c_gpu_engineering_depth64_bs1.sh
P13_GPU=5 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03d_gpu_engineering_depth100_bs1.sh
```

四个 wrapper 使用互不覆盖的输出目录。GPU 编号只是示例，启动前仍应重新检查占用。

## 可覆盖参数

```bash
P13_GPU=1 \
ALPHA_MODE=epsilon_probe \
ALPHA_EPSILON=0.02 \
FEEDBACK_METHODS=bp_current,fa_source,fa_random \
FEEDBACK_RANDOM_SEED=20260901 \
WARMUP_STEPS=2 \
MEASUREMENT_STEPS=5 \
PHASE_LEARNING_RATE=0.01 \
ELECTRONIC_LEARNING_RATE=0.001 \
P11_CHECKPOINT=/absolute/path/to/p11/backbone.pt \
STEM_CHECKPOINT=/absolute/path/to/qwen3_vl_static_stem_224.pt \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03c_gpu_engineering_depth64_bs1.sh
```

默认开启 non-reentrant per-stage activation checkpoint。普通 `03` sweep 默认
`ALPHA_MODE=epsilon_probe`、`epsilon=0.01`，只用于小 alpha 下的工程梯度/资源探针，不能
称为训练稳定性证据。`ALPHA_MODE=full_depth` 会忽略 epsilon 并将全部新增层设为精确 1。

## 输出契约

每个组合输出
`depth_NNN/feedback_METHOD/result.json`，汇总输出 `sweep_summary.json`，均采用
同目录临时文件加 `os.replace` 原子写入。单深度 wrapper 的 root 目录名称已包含深度。

默认拒绝覆盖任何已有 summary/result。中断后只能显式设置 `RESUME_EXISTING=1`；恢复时
会校验 source/stem SHA、GPU UUID、全部 sweep 配置、depth、method 和 random seed 的
campaign/combination SHA。已通过且 identity 完全一致的组合会复用，失败组合会重跑；
任何不一致都是硬错误，不能把不同 method 或 GPU 的资源结果混入同一个目录。
campaign 还锁定 PyTorch/CUDA 版本以及 sweep、P13、P11、stem 和底层 optics 实现文件
的组合 SHA，代码改动后不能复用旧吞吐结果。

重点检查：

- `source.p11_checkpoint_sha256` 与 `migration.source_checkpoint_sha256`；
- `device.gpu_uuid`，避免仅凭逻辑 GPU 序号误认设备；
- `parameters.optical_phase_parameters` 与光学占比；
- `alpha.mode`、`configured_new_stage_alpha`、`all_stages_exactly_one` 和完整 alpha report；
- `combination.combination_sha256`、`feedback.method`、random seed；
- `feedback.initial_manifest_sha256` 与 `final_manifest_sha256`；
- `checks.every_phase_gradient_present`、`finite`、`nonzero` 三项均为 `true`；
- 三项 `input_amplitude_gradient_*` 检查均为 `true`；
- `measurement.peak_allocated_bytes`、`peak_reserved_bytes`、`mean_step_seconds` 与
  `samples_per_second`；
- `status` 只能按工程含义解释：`passed_engineering`、`failed_checks`、
  `failed_oom` 或 `failed_error`。

这个 sweep 不保存训练 checkpoint，不延续语义预训练，不实现 1-D FFT，也不启动正式
ImageNet 训练。

梯度健康度通过一次独立、未计时的 forward/backward/optimizer audit step 检查；
吞吐计时不包含逐 phase 的诊断规约，避免把审计开销误写成训练 step 时间。

## 64/100 层 full-depth feedback 专项审计

```bash
P13_GPU=1 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/05_gpu_full_depth_feedback_cuda_audit_bs1.sh
```

该命令固定 `depth=64,100`、batch=1、`ALPHA_MODE=full_depth`，并对三种 feedback
method 分别审计。详见
[FULL_DEPTH_FEEDBACK_CUDA_AUDIT.md](FULL_DEPTH_FEEDBACK_CUDA_AUDIT.md)。
