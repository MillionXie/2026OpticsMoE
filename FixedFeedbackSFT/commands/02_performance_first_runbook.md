# Performance-first backbone runbook

状态：研究接口约定，性能实验代码尚未创建；以下命令在对应模块实现前不要执行。

建议新实验模块名：

```text
FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone
```

必须实现统一 CLI：

```bash
python -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/main.yaml \
  --phase prepare_data

python -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/main.yaml \
  --phase smoke

python -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/main.yaml \
  --phase train --seed 1234

python -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/main.yaml \
  --phase evaluate_optical_dependence --seed 1234

python -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/main.yaml \
  --phase compare
```

实验自己的 `commands/` 目录必须提供：

```text
01_prepare_data.sh
02_smoke.sh
03_train_bp_backbone.sh
04_evaluate_optical_dependence.sh
05_aggregate.sh
COMMANDS.md
```

`evaluate_optical_dependence` 至少输出：

- normal accuracy；
- optical-off accuracy；
- phase-random accuracy；
- phase-shuffled accuracy；
- normalized optical dependence；
- residual weight min/mean/max；
- optical/skip activation energy ratio；
- electronic parameter count 和 MACs。

正式服务器脚本仍应接受 `PHYSICAL_GPU_INDEX` 或明确的 GPU UUID，不把服务器密码、
个人路径凭据或 test-selected 超参数写入脚本。
