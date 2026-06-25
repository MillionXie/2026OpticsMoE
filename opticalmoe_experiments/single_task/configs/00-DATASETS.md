# Single-Task 数据集说明

本目录下所有光学实验默认把输入变成单通道振幅图：

```text
image -> grayscale -> resize to input_size x input_size -> tensor in [0,1]
```

当前 fair134 光学 MoE 默认：

```text
input_size = 134
canvas_size = 1000
expert_size = 134
expert_pitch = 200
```

## 数据集概览

| 配置前缀 | torchvision 名称 | 原始尺寸 | 通道 | 类别数 | 官方 train/test | 当前默认处理 |
| --- | --- | --- | --- | ---: | --- | --- |
| `mnist` | MNIST | 28x28 | 灰度 | 10 | 60000 / 10000 | resize 到 134x134 |
| `fashionmnist` | FashionMNIST | 28x28 | 灰度 | 10 | 60000 / 10000 | resize 到 134x134 |
| `kmnist` | KMNIST | 28x28 | 灰度 | 10 | 60000 / 10000 | resize 到 134x134 |
| `emnist_letters` | EMNIST split=letters | 28x28 | 灰度 | 26 | 124800 / 20800 | 修正方向，标签 1-26 转成 0-25，resize 到 134x134 |
| `cifar10_gray` | CIFAR10 | 32x32 | RGB | 10 | 50000 / 10000 | 转灰度，再 resize 到 134x134 |

## 难度建议

MNIST 对当前模型可能太简单，容易很快接近高精度，不一定能看出 MoE 的优势。

建议实验顺序：

1. `mnist`: 只做 smoke 和 sanity check。
2. `fashionmnist`: 比 MNIST 更适合作为 10 类基础对比。
3. `kmnist`: 仍是 10 类，但字形更复杂。
4. `emnist_letters`: 26 类，更能区分模型容量和读出能力。
5. `cifar10_gray`: 灰度后仍较难，可作为更强压力测试。

## train/val/test 划分

默认：

```yaml
val_split: 0.1
sampling_protocol:
  enabled: false
  total_size: null
  train_test_ratio: [4, 1]
  class_balanced: true
  seed_offset: 0
```

含义：

- 使用官方 train split 作为训练池。
- 从训练池里切出 `val_split` 比例作为 validation。
- 官方 test split 保持为最终 test。

例如 MNIST：

```text
官方 train = 60000
val_split = 0.1
实际 train ~= 54000
val ~= 6000
test = 10000
```

## sampling_protocol 怎么用

如果数据太多、训练太慢，或者想做小样本公平比较，可以打开：

```yaml
sampling_protocol:
  enabled: true
  total_size: 5000
  train_test_ratio:
    - 4
    - 1
  class_balanced: true
  seed_offset: 0
```

含义：

```text
total_size = train_pool + test
train_test_ratio = [4,1]
train_pool = 4000
test = 1000
再从 train_pool 中按 val_split 切 validation
```

如果 `val_split: 0.1`，则大约：

```text
train = 3600
val = 400
test = 1000
```

这个抽样是尽量 class-balanced 的，适合比较不同模型。

`seed_offset` 用于在同一个全局 seed 下得到另一个可复现抽样。

## max_*_samples 怎么用

如果你想直接指定每个 split 的上限，而不是从 `total_size` 推导，可以写：

```yaml
max_train_samples: 5000
max_val_samples: 1000
max_test_samples: 1000
```

优先级是：

```text
--smoke_test 最高
max_train/val/test_samples 只截断对应 split
sampling_protocol 控制总规模
enabled=false 使用完整官方数据集
```

每次训练启动时都会打印实际 train/val/test 样本数，并保存到
`loader_summary.json`。

## batch 和 epoch 的关系

不是“每个 batch 把整个数据集过一遍”。

实际逻辑是：

```text
一个 epoch = train_loader 中所有 batch 依次跑完一遍
一个 batch = 数据集中的一小批样本
```

例如：

```text
train samples = 54000
batch_size = 64
steps per epoch ~= ceil(54000 / 64) = 844
```

所以一个 epoch 会做大约 844 次参数更新，每次更新只看 64 张图。

如果使用：

```yaml
sampling_protocol:
  enabled: true
  total_size: 5000
```

则训练集变小，每个 epoch 的 step 数也会明显减少。

## num_workers 怎么设置

正式配置默认：

```yaml
num_workers: 16
pin_memory: auto
persistent_workers: true
prefetch_factor: 4
```

这适合 Linux 服务器作为起点。你已经观察到 `num_workers` 从 0 调到 16
后速度明显变快，所以后续正式训练建议先用 16。

如果 CPU 或内存压力太大，可以改成：

```yaml
num_workers: 8
```

或：

```yaml
num_workers: 4
```

Windows、本地调试、排查 dataloader 问题时建议：

```yaml
num_workers: 0
persistent_workers: false
prefetch_factor: null
```

含义是 DataLoader 不开额外进程，最稳定，但通常更慢。

所有 `--smoke_test` 会自动强制：

```yaml
num_workers: 0
persistent_workers: false
prefetch_factor: null
```

避免 Windows/CI/快速检查时多进程 DataLoader 干扰定位。

字段含义：

- `pin_memory: auto`: CUDA 可用时自动启用 pinned memory，加快 CPU 到 GPU 传输。
- `persistent_workers: true`: `num_workers > 0` 时 worker 在 epoch 间保持活跃。
- `prefetch_factor: 4`: 每个 worker 预取 4 个 batch。`num_workers=16` 时最多约 64 个 batch 被提前准备。
- `num_workers=0`: 不传 `persistent_workers` 和 `prefetch_factor` 给 DataLoader。

## grayscale 是否能省算力

对于 MNIST / Fashion-MNIST / KMNIST / EMNIST，这些数据集原本就是灰度。
设成：

```yaml
grayscale: false
```

不会明显省算力。当前 `PILToFloatTensorNoNumpy` 最终仍会 `convert("L")`，
输出 `[1,H,W]` 单通道张量。

对于 CIFAR10，当前光学模型也是单通道振幅输入，`cifar10_gray` 配置应保持
灰度。若未来要做 RGB 光学输入，需要单独设计 RGB pipeline，不建议靠简单
改 `grayscale:false` 来实现。

## 如何去掉电子读出层

对于 MoE / D2NN 光学配置，默认：

```yaml
readout:
  type: mlp
```

这是探测器能量之后的电子 MLP。若要尽量做“无电子后处理”的对照，改成：

```yaml
readout:
  type: optical_only
  normalize_detector_energy: true
  logit_scale: 10.0
  input_norm: none
  norm_affine: false
  hidden_dim: 64
  hidden_layers: 1
  activation: gelu
  dropout: 0.0
```

其中 `hidden_dim / hidden_layers / activation` 在 `optical_only` 下会被忽略。

注意：

- `optical_only` 没有可训练电子层。
- detector energy 求和仍然存在，因为分类必须把 CCD/探测器区域能量读出来。
- `logit_scale` 只是把 detector energy 放大成 cross entropy 更容易优化的 logits。

LeNet-5 是纯电子 baseline，不适用这个设置。

