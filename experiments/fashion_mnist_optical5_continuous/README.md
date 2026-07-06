# Fashion-MNIST Optical5 Continuous Mask Control

本实验用于观察简单 Fashion-MNIST 十分类任务中，五层连续光传播网络能否学习出更有结构的 phase masks。

```text
28×28 grayscale Fashion-MNIST
 -> resize 224×224
 -> RMS normalization
 -> interpolate to 256×256
 -> load complex field once
 -> phase/amplitude mask + angular spectrum propagation × 5
 -> one final square-law detection |E|²
 -> 2×5 fixed class-region detector
 -> two-convolution readout + MLP
 -> ten logits
```

五层之间始终传递 complex field，不进行中间探测、归一化、ReLU 或重新加载。逐层 intensity 图只用于 diagnostics，不参与下一层 forward。

## Dropout

所有正式配置明确设置：

```json
"phase_dropout": {
  "enabled": false
}
```

当 `enabled=false` 时，代码不会生成 dropout mask，`last_phase_dropout_mask` 始终为 `null`，完整 phase modulation 每次都会使用。训练 history 和 phase statistics 会记录这一状态。

## 为什么提供两种初始化

- `fashion_mnist_uniform.json`：与 BDD 连续光模型一致，phase 从 `[0,2π]` 随机均匀初始化；epoch 0 就会呈颗粒状。
- `fashion_mnist_zeros.json`：所有 phase 从 0 开始，用于观察颗粒是否由训练主动形成。

两者 architecture、数据、loss、readout 和 dropout 设置完全相同，只有 `phase_init` 不同。

## Loss

```text
L = classification CE
  + 1.0 × class-region CE
  + 0.1 × detector concentration loss
  + 0.0 × phase total variation
```

`phase_smoothness_weight=0.0`，因此默认没有人为平滑 mask。该项只作为后续显式实验开关，不能与默认 baseline 混为一谈。

## Mask 诊断

每个保存周期输出：

- raw phase；
- wrapped phase `[0,2π]`；
- `cos(phase)`，避免 0/2π 周期边界造成假断裂；
- 五层逐层 light fields；
- 十个 detector 区域能量。

`metrics/phase_statistics.csv` 每层每 epoch 保存 raw std、更新 RMS、circular variance、total variation、最后 batch gradient norm 和 dropout mask 是否存在。仅凭 wrapped phase 图片不能判断 mask 是否训练。

