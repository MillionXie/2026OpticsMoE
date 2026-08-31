# P13 GPU 工程 sweep（非正式训练）

这组命令只回答三个工程问题：真实 P11 source 能否严格迁移到指定深度、batch=1 时一次
forward/backward/SGD step 的显存与速度是多少、每个新增 phase 是否都得到 finite 且非零梯度。
它使用合成的 `3 x 224 x 224` post-adapter 光场，不读取 ImageNet，也不输出准确率；
`passed_engineering` 不能写成 backbone 性能结论。

## 单卡顺序 sweep

```bash
P13_GPU=1 bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03_gpu_engineering_sweep_bs1.sh
```

默认依次测 `16,32,64,100`，每个深度独立从同一个正式 P11 `backbone.pt` 迁移。
若某个深度 OOM，会原子写入 `failed_oom`、释放模型和 CUDA cache，然后继续下一个深度。

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
ALPHA_EPSILON=0.02 \
WARMUP_STEPS=2 \
MEASUREMENT_STEPS=5 \
PHASE_LEARNING_RATE=0.01 \
ELECTRONIC_LEARNING_RATE=0.001 \
P11_CHECKPOINT=/absolute/path/to/p11/backbone.pt \
STEM_CHECKPOINT=/absolute/path/to/qwen3_vl_static_stem_224.pt \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03c_gpu_engineering_depth64_bs1.sh
```

默认开启 non-reentrant per-stage activation checkpoint。新增层 alpha 必须严格大于 0；默认
`epsilon=0.01`，否则新增 phase 在 exact bypass 下没有梯度。

## 输出契约

每个深度输出 `depth_NNN/result.json`，汇总输出 `sweep_summary.json`，均采用同目录临时文件
加 `os.replace` 原子写入。单深度 wrapper 的 root 目录名称已包含深度。

重点检查：

- `source.p11_checkpoint_sha256` 与 `migration.source_checkpoint_sha256`；
- `device.gpu_uuid`，避免仅凭逻辑 GPU 序号误认设备；
- `parameters.optical_phase_parameters` 与光学占比；
- `alpha.configured_epsilon` 和完整 alpha report；
- `checks.every_new_phase_gradient_present`、`finite`、`nonzero` 三项均为 `true`；
- `measurement.peak_allocated_bytes`、`peak_reserved_bytes`、`mean_step_seconds` 与
  `samples_per_second`；
- `status` 只能按工程含义解释：`passed_engineering`、`failed_checks`、
  `failed_oom` 或 `failed_error`。

这个 sweep 不保存训练 checkpoint，不延续语义预训练，不实现 1-D FFT，也不启动正式
ImageNet 训练。

梯度健康度通过一次独立、未计时的 forward/backward/optimizer audit step 检查；
吞吐计时不包含逐 phase 的诊断规约，避免把审计开销误写成训练 step 时间。
