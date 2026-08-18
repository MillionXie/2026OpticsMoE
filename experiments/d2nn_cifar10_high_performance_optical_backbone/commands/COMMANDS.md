# 服务器命令

在仓库任意位置均可调用脚本。GPU 使用 `nvidia-smi` 中的物理序号，并在脚本内转换成 UUID，避免 CUDA 可见序号错位。

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/01_prepare_data.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/02_smoke_test.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/04_evaluate_a01.sh
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/05_aggregate_a01.sh
```

长任务后台启动示例：

```bash
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main_rgb8
nohup env PHYSICAL_GPU_INDEX=3 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main_rgb8/train.log 2>&1 &
```

默认会从 `latest.pt` 自动续训。只有明确希望丢弃同配置断点时才直接在 Python 命令追加 `--force`。
