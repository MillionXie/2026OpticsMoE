# 服务器命令

在仓库任意位置均可调用脚本。GPU 使用 `nvidia-smi` 中的物理序号，并在脚本内转换成 UUID，避免 CUDA 可见序号错位。

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/01_prepare_data.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/02_smoke_test.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/04_evaluate_a01.sh
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/05_aggregate_a01.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/06_train_a02_pool16.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/07_pretrain_a03_cifar100.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/08_finetune_a04_cifar10.sh
PHYSICAL_GPU_INDEX=1 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/09_refine_a05_from_a01.sh
```

长任务后台启动示例：

```bash
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main_rgb8
nohup env PHYSICAL_GPU_INDEX=3 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main_rgb8/train.log 2>&1 &
```

默认会从 `latest.pt` 自动续训。只有明确希望丢弃同配置断点时才直接在 Python 命令追加 `--force`。

`08_finetune_a04_cifar10.sh` 依赖 A03 的 `best.pt`，必须等 `07_pretrain_a03_cifar100.sh` 正常结束后启动。

`09_refine_a05_from_a01.sh` 依赖 A01 的 `best.pt`，用于低学习率连续精修；它不改变架构。

A05 只有在明确重置该输出目录的训练状态时使用 `FORCE_RESTART=1`，否则默认从自己的 `latest.pt` 续训：

```bash
FORCE_RESTART=1 PHYSICAL_GPU_INDEX=1 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/09_refine_a05_from_a01.sh
```
