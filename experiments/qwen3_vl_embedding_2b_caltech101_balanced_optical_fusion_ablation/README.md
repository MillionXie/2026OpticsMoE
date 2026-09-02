# Caltech101 光电残差贡献消融

本工程不修改 `warmstart5` 正式模型，而是以其固定 **81.00% Top-1** 的 EMA
checkpoint 为唯一初始化，比较更可解释的光电融合方式。

## 为什么旧 alpha 不能叫“光贡献率”

旧式融合为：

```text
F = E + alpha * O
```

电子分支恒为 1，光分支才有系数；而且 `E`、`O` 的 RMS 可以不同。所以旧
`alpha=0.055` 只是“光学残差系数”，不是 5.5% 的实际贡献率。

## 新融合

每个样本分别在全部有效 token × 192 channel 上计算 RMS，padding 不进入统计：

```text
rE = stopgrad(RMS(E))
rO = stopgrad(RMS(O))
En = E / rE
On = O / rO
M  = (1-alpha) * En + alpha * On
F  = rE * M / stopgrad(RMS(M))
```

它有四个性质：

1. 电系数明确为 `1-alpha`，光系数明确为 `alpha`；
2. 两个分支先处于相同 RMS，光不会仅因数值小而被电淹没；
3. 最终输出重新回到原电子分支的 RMS，尽量保持下游网络的工作尺度；
4. `alpha=0` 时在代数上严格退化为 `F=E`。

RMS 统计使用 `detach`：网络不能仅靠放大/缩小分支范数来绕过 alpha。没有加入
branch-specific affine、额外 LayerNorm 或逐 token 归一化，因此 token 之间原有的
强弱关系保留。

每层记录：alpha、电/光原始 RMS、`rO/rE`、匹配后 RMS 比、融合后/电子 RMS 比和
电光 cosine。Vision/Language 各两层，共四个 alpha。

## 对照组

| 组别 | alpha 范围 | 初始值 | 用途 |
|---|---:|---:|---|
| free | (0.01, 0.95) | 0.055 | 自由学习参考 |
| low | (0.05, 0.49) | 0.055 | 强制电贡献更大 |
| high | (0.51, 0.95) | 0.55 | 强制光贡献更大 |
| electronic-only | 无光传播 | — | 重新训练后的电子容量上限 |

除此之外，必须对同一个最佳 hybrid checkpoint 直接做 `remove_optical`，不重新训练。
这是回答“把光拿走，准确率下降多少”的首要因果对照；电子-only 重新训练是另一个
问题，不能代替它。

## 公平性

- 四组固定从 warmstart5 epoch 8 EMA checkpoint 开始；SHA-256 被锁定；
- 所有非 gate tensor 严格同名、同形状加载；四个 gate 按本组范围重置；
- optimizer 状态不继承；数据、batch、loss、增强、30 epoch 完全一致；
- `optimizer_steps_per_epoch=null` 使用 `ceil(N/(P*K))` 个自然等效 batch；由于
  `P=10` 覆盖全部类别、每类固定 `K=3`，这是类别平衡 PK 抽样，不等于每张训练图
  在单个 epoch 恰好出现一次（大类会跨 epoch 逐步覆盖，小类会重复抽样）；
- 按用户指定，不设 validation：每 5 epoch 测一次 test，并保存最高 test Top-1
  的 live/EMA checkpoint。该结果属于 test-selected，不应再宣称为无偏泛化估计。

具体命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
