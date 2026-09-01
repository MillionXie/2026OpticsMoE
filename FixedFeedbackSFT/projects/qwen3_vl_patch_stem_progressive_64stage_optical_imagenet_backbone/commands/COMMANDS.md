# P13 engineering and guarded training commands

本目录同时包含自动测试、严格迁移/工程审计，以及正式 ImageNet 训练 launcher。
launcher 的存在不代表训练已经运行；正式状态必须以对应 run 的 checkpoint/result 为准。

## 1. 本地/服务器代码验收

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/01_test.sh
```

这组测试不读取真实 ImageNet；它构造合成 stem 和符合正式字段契约的 P11
export，验证 schedule、预算、严格迁移、alpha=0 feature 等价、alpha>0 phase
梯度及 ramp/state reload。

## 2. 从正式 P11 export 构造 64-stage 初始化

先只读确认源文件：

```bash
sha256sum FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt
sha256sum FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt
```

再执行迁移：

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/02_build_migrated_64stage_prototype.sh
```

若实际路径不同，必须显式覆盖，不能改成模糊 glob：

```bash
P11_CHECKPOINT=/absolute/path/to/backbone.pt \
STEM_CHECKPOINT=/absolute/path/to/qwen3_vl_static_stem_224.pt \
OUTPUT_DIRECTORY=/absolute/path/to/p13_migrated_64stage_initialization \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/02_build_migrated_64stage_prototype.sh
```

成功产物是 `p13_migrated_initialization.pt` 和 `manifest.json`。应检查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("FixedFeedbackSFT/runs/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/p13_migrated_64stage_initialization/manifest.json")
payload = json.loads(path.read_text())
print(json.dumps({
    "source_sha256": payload["migration"]["source_checkpoint_sha256"],
    "phase_hash_equal": payload["migration"]["source_phase_sequence_sha256"] == payload["migration"]["target_anchor_phase_sequence_sha256"],
    "num_stages": payload["model"]["num_stages"],
    "optical_phase_parameters": payload["model"]["optical_phase_parameters"],
    "electronic_backbone_parameters": payload["model"]["electronic_backbone_parameters"],
    "formal_training_started": payload["formal_training_started"],
}, indent=2))
PY
```

期望核心值：`phase_hash_equal=true`、`num_stages=64`、
`optical_phase_parameters=9633792`、`electronic_backbone_parameters=965176`、
`formal_training_started=false`。

## 3. 两种不同的深度路径

`02` 原型命令仍支持把同一个 P11 source 独立迁移到 `16|32|64|100`，只用于架构与资源
审计。正式训练采用另一条已实现的严格链：`P11 epoch-88 -> 16 -> 32 -> 64 -> 100`。
每一步完整迁移上一深度的 stage、adapter 和 ImageNet readout；新增 stage 以 alpha=0
保持函数不变，再由正式训练 schedule 拉到 1。两条路径不能混写为同一个实验结果。

## 4. GPU 工程 sweep（batch=1，非正式训练）

16/32/64/100 层的单卡顺序 sweep：

```bash
P13_GPU=1 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/03_gpu_engineering_sweep_bs1.sh
```

也可以用 `03a`、`03b`、`03c`、`03d` 四个 wrapper 在不同 GPU 上分别测
16/32/64/100 层。它们只使用合成 post-adapter 光场做 warmup 和少量
forward/backward/optimizer step，并严格读取正式 P11 source；不读取 ImageNet，
不保存训练 checkpoint，也不能解释为性能实验。完整参数、输出字段与 OOM 行为见
[GPU_ENGINEERING_SWEEP.md](GPU_ENGINEERING_SWEEP.md)。

## 5. 64/100 层 full-depth feedback CUDA audit

```bash
P13_GPU=1 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/05_gpu_full_depth_feedback_cuda_audit_bs1.sh
```

该命令只做合成光场工程审计：64/100 层在精确 `alpha=1` 下分别运行 `bp_current`、
`fa_source`、`fa_random`，记录逐组合 manifest hash、全部 carried+new phase 及输入
amplitude 梯度健康、峰值显存和 step time。
不读取 ImageNet、不保存训练 checkpoint、不构成性能实验。中断恢复和结果目录契约见
[FULL_DEPTH_FEEDBACK_CUDA_AUDIT.md](FULL_DEPTH_FEEDBACK_CUDA_AUDIT.md)。

## 6. 正式 16 层训练与逐级 guarded growth

P11→16 的正式 smoke、训练、状态命令以及 P11 matched continuation control 见
[GROWTH_TRAINING_COMMANDS.md](GROWTH_TRAINING_COMMANDS.md)。只有 16 层产生正式
`best_full_depth.pt` 后，才能渲染 32 层配置；64 与 100 层同理：

```bash
TARGET_DEPTH=32 \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/12_render_or_verify_progressive_growth.sh

TARGET_DEPTH=32 PHYSICAL_GPU_INDICES=0,1,3,4 P13_ACTION=fresh \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/13_launch_progressive_growth.sh
```

将 `TARGET_DEPTH` 依次改为 `64`、`100`。脚本内部固定上一深度 run，不接受占位 source；
每次渲染和启动都会重新检查 `best_full_depth` 的 role、depth、alpha、architecture、
migration、feedback manifest 及文件 SHA。配置写入 `configs/generated/`；若已有配置与
当前 source identity 不一致会拒绝覆盖。四卡 per-rank batch/accumulation 分别为
`12/4`、`6/8`、`4/12`，三档 effective global batch 均为 192。
完整 guard 与逐级操作见 [PROGRESSIVE_GROWTH_CHAIN.md](PROGRESSIVE_GROWTH_CHAIN.md)。
