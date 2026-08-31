# P13：P11 兼容的渐进式 64 层光学 backbone 原型

## 1. 当前状态与边界

P13 是一个**只完成本地架构与迁移验证、尚未启动正式训练**的独立原型。
首要配置将 P11 的 8 个光学 stage 扩展为 64 个，使可训练相位从
`1,204,224` 增至 `9,633,792`（约 9.63M）。同一实现还支持
16/32/100 层，便于做深度与参数规模消融。

P13 没有宣称新的 ImageNet 精度。目前唯一成立的结论是：结构预算、P11
严格迁移、alpha=0 的特征函数保持以及 alpha>0 后新增相位获得梯度均由
自动测试覆盖。正式性能、吞吐、显存和物理扰动结论必须等待后续实验。

## 2. 架构

固定前端与 P11 相同：

1. 冻结的 Qwen3-VL Patch/Position Stem 输出 `196 x 1024` token；
2. 唯一可训练 adapter 将其变为 `196 x 224`；
3. pad 成三路 `3 x 224 x 224` 光学场；
4. token-axis、channel-axis 光学 stage 交替；
5. 临时 ImageNet readout 只用于预训练，不计入 reusable backbone 预算。

64 层由 32 个 token/channel 光学宏块组成。每个 stage 仍是 P11 的 OEO
语义：相位调制、单轴角谱传播、平方律探测、归一化/非线性、电子重载与
受约束光电融合。只有从 P11 迁移的 8 个 anchor stage 保留 width-96
`SlimSpatialTokenMixerSkip`；其余 56 个 stage 的电子 skip 是无参数 identity。
这里的 identity processor 仍沿用 P11 的无参数 RMS 数值归一化，并非完全没有
电子运算；它不含可训练 transform。每个新增 stage 只多出一个原有的受约束
光/skip 融合标量，不复制 mixer。

### Anchor 排布

四个原 P11 token/channel pair 保持顺序与相邻关系，并均匀分布到目标深度：

| 深度 | P11 anchor stage（0-based） |
| ---: | --- |
| 16 | `0,1,4,5,10,11,14,15` |
| 32 | `0,1,10,11,20,21,30,31` |
| 64 | `0,1,20,21,42,43,62,63` |
| 100 | `0,1,32,33,66,67,98,99` |

这意味着 alpha=0 时，P13 实际执行的 8 个非 identity stage 与 P11 完全同序，
不是简单复制同一个 phase，也不是把 8 层反馈算子循环套用到深层网络。

## 3. Function-preserving depth gate

每个新增 stage 外部使用固定调度门：

```text
y = x + alpha * (Stage(x) - x)
```

- anchor 的 `alpha` 永远为 1；
- 新增层 `alpha=0` 时走 Python 级直接 bypass，既严格等于 identity，也避免
  无意义 FFT；
- `alpha=0` 同时意味着新增 phase 没有梯度，所以训练第 1 个增长 epoch 从
  `epsilon` 开始；
- `apply_depth_ramp(epoch)` 将所有新增层从 epsilon 线性拉到 1；默认
  `epsilon=0.01`、10 个增长 epoch；
- alpha 是持久化 buffer 和强制 schedule 状态，不是新增可训练电子参数。

GPU forward 使用同步的 Python alpha 副本判断 bypass，不会为每个新增 layer
执行 `device -> host` 同步；加载 state dict 时会同步恢复该副本。

## 4. P11 -> P13 严格迁移

`migration.py` 只接受正式 P11 `backbone.pt` 格式，并验证：

- `p11_separable_architecture_signature == [11,1,2,4]`；
- `model_report.optical_mixer_variant == separable_token_channel_axis`；
- source 恰好为 8 stage；
- P11 与 P13 的冻结 Qwen stem SHA-256 完全一致；
- reusable state 没有 ImageNet task head，且除 head 外无 missing/unexpected key；
- 8 个 source `raw_phase` 与迁移后 8 个 anchor `raw_phase`（含顺序、dtype、
  shape）的 SHA-256 相同。

迁移内容包括冻结 stem buffers、1024->224 adapter 和 8 个完整 P11
光学/mixer stage。新增 phase 保持目标 seed 的确定性初始化。P11 的临时
ImageNet head 按 backbone export 契约不迁移，因此 alpha=0 保证的是
**最终光学 feature 逐元素等价**，不是随机新 head 的 logits 等价。

迁移器生成：

- `p13_migrated_initialization.pt`：不含临时 task head 的 reusable 初始化；
- `manifest.json`：source 路径与 SHA、stem SHA、anchor mapping、phase hash、
  完整参数报告和 `formal_training_started=false`。

## 5. 参数预算

每层有 `3 x 224 x 224 = 150,528` 个相位。电子 backbone 由固定 adapter、
8 个唯一 mixer 及每层一个融合标量组成：

| 深度 | 光学 phase | 电子 backbone（不含临时 head） | 光学占 backbone |
| ---: | ---: | ---: | ---: |
| 16 | 2,408,448 | 965,128 | 71.3915% |
| 32 | 4,816,896 | 965,144 | 83.3079% |
| **64** | **9,633,792** | **965,176** | **90.8937%** |
| 100 | 15,052,800 | 965,212 | 93.9742% |

64 层新增的 56 个 identity skip 一共包含 0 个 transform 参数；与 P11 相比，
电子 backbone 只增加 56 个融合标量。临时 ImageNet head 和冻结 Qwen stem
均不进入上表的 reusable backbone 占比。

## 6. 已实现的工程保护

- 可选 `activation_checkpointing=true` 使用 non-reentrant、逐 stage 重计算；
- checkpoint closure 显式绑定当前 stage，避免 backward 重计算时发生 Python
  late-binding 错层；
- phase、adapter、残差电子与 head 参数均有独立迭代器；
- `parameter_report()` 输出深度、axis schedule、anchor mapping、phase/电子预算、
  alpha 状态与 migration manifest；
- `backbone_state_dict()` 明确排除临时 ImageNet head。

逐 stage checkpoint 只是 stage 内部重计算，**尚不能视为已解决 64/100 层显存**。
正式训练前必须在目标 GPU 实测峰值显存与吞吐，并根据结果增加 2--4 stage
segment checkpoint 或其他分段策略。

## 7. 自动验收

测试覆盖：

1. 16/32/64/100 四档 anchor schedule 与 token/channel parity；
2. 64 层 9.63M phase、8 个唯一 mixer、0 参数 identity skip、965,176
   电子 backbone 预算；
3. 用正式 export 结构构造 P11 source，严格迁移到 64 层后在 eval 模式下验证
   alpha=0 feature **bitwise equality**；
4. 16 层 alpha>0 全链反传时，每个新增 phase 的梯度均 finite 且 norm>0；
5. epsilon/ramp 端点与 state-dict reload 后 Python alpha 同步。

执行命令见 [commands/COMMANDS.md](commands/COMMANDS.md)。命令只运行测试或
构造迁移初始化，不含正式训练 launcher。

## 8. 正式训练前仍需完成

- 在空闲 GPU 做 16/32/64 的真实 batch forward/backward、峰值显存和 samples/s
  审计，再锁 batch、checkpoint segment 与累积步数；
- 先做 alpha=0 的真实 P11 checkpoint feature audit，再开始 epsilon/ramp；
- 对比 8/16/32/64 深度，至少包含相同训练预算与 phase-parameter-matched 控制；
- 单独消融 progressive ramp 对比直接 alpha=1，避免把“更深”与“更易训练”混为一谈；
- 当前 `AxisAngularSpectrumPropagator` 的物理传递函数是单轴的，但实现仍使用
  `fft2/ifft2`；真正的 1-D FFT 优化尚未实现；
- 64 个逻辑 OEO stage 在 alpha=1 时表示 64 次相位/传播/探测/重载，并不自动
  等价于已搭建的 64 个无源物理平面。外部 depth blend 在 alpha<1 时也是训练期
  电子路径；部署主张应以 alpha=1 为准；
- fixed-feedback 训练必须先获得完整深层 source connector，不能循环复用 P11
  的 8 个反馈 phase。这不属于当前初始化原型。

工程 sweep 模块已经实现，但尚未在服务器执行。它会针对 16/32/64/100 层严格迁移
真实 P11 source，以 batch=1 合成光场测量 forward/backward/SGD step 的峰值显存、
吞吐和所有新增 phase 的梯度健康度；OOM 会记录后清理并继续下一深度。该结果只用于
锁定可运行配置，不能作为 ImageNet 或 backbone 性能结果。命令见
[commands/GPU_ENGINEERING_SWEEP.md](commands/GPU_ENGINEERING_SWEEP.md)。
