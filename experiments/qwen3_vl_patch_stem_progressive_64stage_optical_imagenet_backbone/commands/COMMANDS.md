# P13 prototype commands

本目录当前只有自动测试和严格迁移初始化命令，**没有正式训练 launcher**。

## 1. 本地/服务器代码验收

```bash
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/01_test.sh
```

这组测试不读取真实 ImageNet；它构造合成 stem 和符合正式字段契约的 P11
export，验证 schedule、预算、严格迁移、alpha=0 feature 等价、alpha>0 phase
梯度及 ramp/state reload。

## 2. 从正式 P11 export 构造 64-stage 初始化

先只读确认源文件：

```bash
sha256sum experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt
sha256sum experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt
```

再执行迁移：

```bash
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/02_build_migrated_64stage_prototype.sh
```

若实际路径不同，必须显式覆盖，不能改成模糊 glob：

```bash
P11_CHECKPOINT=/absolute/path/to/backbone.pt \
STEM_CHECKPOINT=/absolute/path/to/qwen3_vl_static_stem_224.pt \
OUTPUT_DIRECTORY=/absolute/path/to/p13_migrated_64stage_initialization \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/02_build_migrated_64stage_prototype.sh
```

成功产物是 `p13_migrated_initialization.pt` 和 `manifest.json`。应检查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/runs/p13_migrated_64stage_initialization/manifest.json")
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

## 3. 其他深度

迁移模块支持 `--num-stages 16|32|64|100`。每个深度都从同一个 P11 source
独立迁移；当前没有实现 16→32→64 checkpoint 级继续增生。不要把原型迁移
命令描述成训练，也不要在没有 GPU memory smoke 的情况下添加正式 launch。
