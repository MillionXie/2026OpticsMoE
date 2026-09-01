# P13 全深度光学 fixed-feedback 设计

## 目标

P13 支持 16/32/64/100 个 OEO stage。本实现为目标深度中的**每一个 stage 输入光学算子**定义一条独立反馈连接，不把 P11 的 8 个 phase/connector 循环复用到新增层。

- depth=16：16 条光学输入连接，其中 15 条是层间连接；
- depth=32：32 条光学输入连接，其中 31 条是层间连接；
- depth=64：64 条光学输入连接，其中 63 条是层间连接；
- depth=100：100 条光学输入连接，其中 99 条是层间连接。

第 0 条连接从 adapter 输出进入 stage 0，其余第 `i` 条连接从 stage `i-1` 的输出进入 stage `i`。保留第 0 条是因为它决定 adapter 收到的误差信号，也能避免训练口径里隐藏一条未审计的光学连接。

## 三种反馈模式

### `bp_current`

每层使用当前 forward phase 对应的精确光学 Jacobian。它是深层模型的标准 BP 对照。

### `fa_source`

每层 forward 始终使用当前 phase；当前 phase 的局部梯度也始终精确。只有传给该 stage 输入 amplitude 的复数光场连接器固定为 source phase。

严格迁移 P11 后，系统立即捕获完整目标深度的 source：

- 8 个 anchor 使用各自迁移后的 P11 phase；
- 每一个新增 stage 使用它自己按 P13 seed schedule 初始化的 phase；
- 不插值、不平铺、不循环复用 8 层反馈。

若捕获后 phase 尚未漂移，则每一条 source connector 与 current-BP connector 相同，因此全模型梯度在数学上相同；单测同时比较输入、全部 phase 和全部被使用电子参数的梯度。

### `fa_random`

每个 stage 使用独立派生的 63-bit PRNG seed：

```text
seed_i = SHA256(format, base_seed, target_depth, stage_index, optical_axis)
```

该派生与模型初始化 RNG 的消费顺序无关，也不会因为插入其他 connector 而改变既有层的随机流。随机 phase 在 `[0,2pi)` 均匀采样。

对 stage `i`，被替换的复线性连接器可写为：

```text
B_i(phi) = H_i diag(exp(j phi))
```

其中 `H_i` 是该层固定的 token-axis 或 channel-axis 传播算子。任意 phase 的对角调制矩阵都是酉矩阵，因此把 connector 视为复线性算子时，`B_i(phi_random)` 与 `B_i(phi_source)` 具有完全相同的奇异值谱与 Frobenius 范数。实际网络输入给该算子的 amplitude 是实数，后面还接有强度探测与非线性；因此我们只把它作为严格的逐层复算子/能量尺度控制，不声称任意下游实 Jacobian 的完整奇异值谱也必然相同。

## 哪些梯度被固定，哪些仍是精确 BP

固定的只有每个 stage 光学复场输出到其输入 amplitude 的连接器。以下内容仍走当前 forward 图上的精确自动微分：

- 当前 phase 的局部导数；
- CCD square-law、归一化和非线性；
- 光电残差融合；
- 8 个 anchor 的 width-96 mixer；
- adapter、临时 readout/head；
- 新增层的 outer depth blend `x + alpha * (Stage(x)-x)`。

因此准确表述是：**在 exact-electronic scaffold 下固定全深度 inter-stage optical connector**，而不是“整个网络完全不用 BP”。实现没有把光学层近似成 identity STE；自定义 autograd 明确重算当前局部 phase 的精确梯度，并用指定的冻结光学算子替代传向上一层 amplitude 的 connector。这本身就是受控的 feedback-alignment surrogate gradient，不能表述成全网络的真实 BP。

当新增层 `alpha=0` 时它是严格 Python bypass，phase 按定义没有梯度；当 `alpha>0` 时才要求每个新增 phase 的梯度存在、有限且非零。

## checkpoint 与恢复契约

`feedback_source_phases` 是 `[depth,3,224,224]` 的冻结、持久化 buffer。它是恢复 `fa_source` 所必需的实验状态，不计入可训练参数。

每个 stage 内实际启用的 `feedback_phase` 是 runtime-only、`persistent=False`、`requires_grad=False`：

- 不写入 `state_dict`，避免随机/旧 runtime connector 被误当成模型权重；
- `load_state_dict` 后模型强制回到 `bp_current`；
- resume controller 必须显式再次调用 `configure_feedback("fa_source")` 或 `configure_feedback("fa_random", random_seed=...)`。

正式 checkpoint 外层需要保存 `feedback_manifest()`。迁移原型已经保存该字段。manifest 逐连接记录：stage/axis/anchor 身份、source 与 active phase SHA-256、传播 transfer-function SHA-256、冻结状态、随机子流 seed 和范数匹配语义。

持久化 source 会增加 checkpoint buffer 体积：64 层约 36.8 MiB，100 层约 57.4 MiB（float32，仅 phase source；不增加可训练参数）。这是恢复任意训练后 deep source connector 所需的信息，而不是额外电子计算。

## 使用顺序

```python
model = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
migrate_strict_p11_checkpoint(model, p11_checkpoint)

# exact BP control
model.configure_feedback("bp_current")

# fixed full-depth migrated source
model.configure_feedback("fa_source")

# norm-matched, per-stage independently seeded random feedback
model.configure_feedback("fa_random", random_seed=20260901)
```

未来若先完成 64/100 层 source 预训练，应在 source checkpoint 选定后调用 `capture_feedback_source(...)` 冻结当时的完整深层 phase，再开始下游 `fa_source`。不能继续沿用迁移起点的随机新增层作为“预训练 source”。

## 当前验证范围与尚未成立的结论

自动测试覆盖：

- 16/32/64/100 四种深度均为每层建立独立可审计 connector；
- random 子流逐层唯一、同 base seed 可复现、不同 base seed 改变全部连接；
- random phase 为单位模调制；
- source 捕获点的 `fa_source` 与 `bp_current` forward/gradient 一致；
- 16 层 `alpha>0` 时，`fa_source` 和 `fa_random` 的全部新增 phase 梯度均 finite/nonzero；
- state dict 保存 source、不保存 runtime feedback，reload 后显式恢复反馈模式。

本轮没有启动正式训练，因此尚不能宣称：

- 64/100 层能够在目标 GPU 上以可接受吞吐训练；
- deep FA 的 ImageNet 精度等于或接近 BP；
- 随机反馈在深度增加后仍保持优化稳定；
- 逻辑 100-stage OEO 已等价于无源 100 个物理相位平面的硬件。

下一步应先把 engineering sweep 扩展为 `bp_current/fa_source/fa_random` 三模式的 16/32/64/100 梯度、显存和耗时审计，再决定 64 层主实验的 alpha ramp、batch 和累积步数。
