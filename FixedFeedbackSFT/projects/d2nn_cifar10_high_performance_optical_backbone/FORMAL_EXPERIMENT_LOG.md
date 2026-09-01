# 固定反馈四组实验记录

更新日期：2026-08-18

## 唯一正式方法

主表只允许 NoFT、BP、FA-pretrained、FA-random 四组。head-only、optical-off、phase-random 和 phase-shuffle 只作为内部机制诊断，不是额外方法组。

## 共同 source 与起点

- source optical operator：A03 CIFAR-100 best checkpoint；
- source SHA-256：`f632c57cf851805090686cda81d4b4a0efc07b02c91dc0e0b63c00912247becc`；
- source CIFAR-100 test：32.13%，normalized optical dependence 87.92%；
- 先冻结全部 optical stages，只训练一个共同 CIFAR-10 head；
- head-warmup checkpoint 的模型权重和 SHA-256 必须被四组共同引用；
- 每个 seed 内，BP/FA-pretrained/FA-random 使用相同 batch order、augmentation seed、优化器、LR、epoch 和 validation policy。

NoFT 直接评估共同起点。三个微调组的电子头和 residual 均使用普通 BP；区别只在 optical stage 之间的反向误差连接：

- BP：当前相位算子；
- FA-pretrained：冻结 A03 source phase；
- FA-random：冻结、seed-matched 的随机 phase。

所有方法的前向都使用当前可训练相位。FA 不允许在前向中使用冻结 phase；本层 phase gradient 仍由当前局部算子精确计算，只有传向前一 stage 的误差信号使用固定连接。

## Pilot P01

状态：2026-08-18 三个 seed 全部完成，正式汇总已生成。

- 配置：`configs/formal_pilot.yaml`
- head warm-up：10 epoch，seed 4242；
- downstream：seed 2026、2027、2028，各 20 epoch；
- phase LR：5e-4；electronic/residual LR：3e-4；
- checkpoint：validation best，且 epoch-0 共同起点受到保护；
- 预期用途：验证高性能 RGB 八层骨干上是否仍出现 FA-pretrained 接近 BP、且优于 FA-random 的现象。

必须回填：共同 checkpoint SHA-256、head-warmup best、四组 test accuracy、光学依赖、epoch-0 分层 gradient cosine、endpoint phase drift、BP-FA gap 和是否进入多 seed。

已完成的共同起点与 NoFT：

- head warm-up selected epoch：10；validation accuracy：54.10%；
- common checkpoint SHA-256：`deceeec8dfad0026904b921f4d8e20f352bf1ac33735ef2bc14ad1787030c31a`；
- NoFT test accuracy：55.53%；optical-off：16.08%；phase-random：14.14%；phase-shuffle：10.40%；
- NoFT normalized optical dependence：86.65%。

这说明共同起点已经具有可展示的下游性能，并且该性能不是电子 bypass 单独提供的。四种方法均引用这个完全相同的 checkpoint；后三种方法在每个 seed 内采用完全相同的数据顺序与预算。

### P01 seed 2026 结果

| 方法 | selected epoch | validation | test | optical-off | normalized optical dependence | phase RMS drift |
|---|---:|---:|---:|---:|---:|---:|
| NoFT | 0 | 54.10% | 55.53% | 16.08% | 86.65% | 0.0000 rad |
| BP | 20 | 57.04% | 58.44% | 15.15% | 89.37% | 0.0680 rad |
| FA-pretrained | 20 | 57.04% | 58.44% | 15.16% | 89.35% | 0.0682 rad |
| FA-random | 15 | 56.38% | 57.39% | 15.95% | 87.44% | 0.0955 rad |

直接差值：BP - NoFT = +2.91 pp；FA-pretrained - NoFT = +2.91 pp；FA-pretrained - FA-random = +1.05 pp；BP - FA-pretrained = 0.00 pp。

机制检查：FA-pretrained 在 epoch 0 的八层 gradient cosine 均约为 1；FA-random 从第一层到末层为 `[0.0428, 0.0623, 0.2471, 0.0826, 0.4010, 0.4279, 0.7140, 1.0000]`。随机反馈的末层为 1 是设计预期，因为末层不经过更后的固定 connector。

seed 2026 的初步结果触发了原协议的重复验证；seed 2027、2028 没有修改 common checkpoint 或任何超参数。

### P01 三 seed 正式汇总

下表为 test Top-1 的三 seed 均值 ± 样本标准差；光学依赖和相位漂移列为三 seed 均值。

| 方法 | seed 2026 | seed 2027 | seed 2028 | test mean ± std | optical dependence | phase RMS drift |
|---|---:|---:|---:|---:|---:|---:|
| NoFT | 55.53% | 55.53% | 55.53% | 55.53% ± 0.00 pp | 86.65% | 0.0000 rad |
| BP | 58.44% | 58.27% | 58.38% | 58.36% ± 0.09 pp | 89.27% | 0.0676 rad |
| FA-pretrained | 58.44% | 58.28% | 58.44% | 58.39% ± 0.09 pp | 89.28% | 0.0680 rad |
| FA-random | 57.39% | 57.66% | 57.40% | 57.48% ± 0.15 pp | 88.08% | 0.0987 rad |

配对 test 差值：

- BP - NoFT：`[+2.91, +2.74, +2.85]` pp，均值 `+2.83 pp`；
- FA-pretrained - NoFT：`[+2.91, +2.75, +2.91]` pp，均值 `+2.86 pp`；
- FA-pretrained - BP：`[0.00, +0.01, +0.06]` pp，均值 `+0.02 pp`；
- FA-pretrained - FA-random：`[+1.05, +0.62, +1.04]` pp，均值 `+0.90 pp`，三个 seed 方向一致。

P01 结论：在当前高性能、强光学依赖但小相位更新的 regime 中，预训练固定反馈复现了 BP 的性能和更新轨迹，而随机固定反馈稳定较差。该结论不应外推到大相位漂移：BP/FA-pretrained 的平均 phase RMS drift 只有约 0.068 rad，operator coherence 约 0.9977。下一轮不再增加方法组，而是在同样四组内提高训练时长/phase LR，并把残差光学权重下限提高，检验结论在更高光学处理比例和更强更新下是否仍成立。

后续 BP 可行性 A07 已验证上述方向：`main_min=0.50`、50 epoch 时 test Top-1 60.25%，平均光学权重 51.73%，normalized optical dependence 97.71%。因此下一轮四组协议应直接固定 A07 设置，不再筛选更多结构；目标是判断 FA-pretrained 是否仍贴近 BP 且优于 FA-random，而不是继续增加对比方法。

## 工程验证 S01

在 P01 长跑前，用 `configs/formal_smoke.yaml` 对真实 A03 source 执行一次 head warm-up、四种方法各两个训练 batch、四种光学消融和完整性受检的比较汇总。S01 只验证数据、checkpoint、固定反馈 autograd 与文件输出链路，不用于科学结论。

状态：2026-08-18 服务器通过。

- 服务器单元测试：8 passed；
- 真实 A03 checkpoint 的旧格式兼容与 SHA-256 校验通过；
- smoke common checkpoint SHA-256：`a7d5c436ab33ebe15435b27cae1fe16e5d7be86f6083d8ec1970f9f052d0874b`；
- 四组均生成 `best.pt`、`result.json`，比较器生成 `comparison.json` 和 `runs.csv`；
- FA-pretrained 的 epoch-0 八层 gradient cosine 均约为 1，符合“冻结算子等于初始前向算子”预期；
- FA-random 的前七层 gradient cosine 为 `[0.1023, 0.1749, 0.1194, 0.4647, 0.2537, 0.5784, 0.7256]`，末层为 1，符合“只替换跨层 error connector，末层仍由相同 head 误差直接驱动”的预期。

结论：S01 只覆盖极少样本，accuracy 不作解释；工程链路允许进入 P01。

## P02：A13 强骨干上的唯一四组（已启动）

启动门槛：A13 的 seed 1234/2026/2027 三 seed 复验必须先通过
`OPTIMIZATION_LOG.md` 预注册性能、光学依赖和 gate 条件。确认性 seed 2028 无论好坏都报告，
但不改变启动门槛。

若门槛通过，P02 固定以下内容，不再按方法单独调参：

- 唯一四组仍为 NoFT、BP、FA-pretrained、FA-random；
- A03 source checkpoint 与 SHA-256 保持不变；
- A13 的 low-resolution electronic residual、312,336 residual electronic parameters、
  104,330 参数 MLP head、`main_min=0.50` 全部锁定；
- 共同 head warm-up 为 seed 4242、10 epochs；stage 全冻结，因此新电子残差在共同起点仍为
  零初始化，不允许某个反馈方法单独获得预训练电子旁路；
- downstream seeds 为 2026/2027/2028，每组 50 epochs，LR 和 cosine schedule 与 A13 相同；
- 所有电子模块始终用普通 BP，四组只改变光学 stage 之间的 backward connector；
- 除原有 normal/optical-off/random/shuffle 外，正式结果增加 electronic-skip-off 和
  long-skip-off，排除电子旁路解释。

配置为 `configs/formal_a13_high_performance.yaml`；共同起点、单个 method×seed 和最终比较只能
分别使用 `commands/28_prepare_common_a13_formal.sh`、`commands/29_run_a13_formal_method.sh`、
`commands/30_compare_a13_formal.sh`。在 A13 复验结束前只准备代码和测试，不启动 P02。

2026-08-19 启动记录：A13 预注册三 seed test 为 72.34% ± 0.17 pp，全部通过性能、光学依赖
和 gate 门槛；确认性第四 seed 也通过，因此触发 P02。共同 head warm-up selected epoch 10，
validation 50.42%，checkpoint SHA-256
`b64570eec65141037cd512a1626e44e879500a73cf7df817e892fd3455b728f0`。NoFT 三 seed test
均为 51.18%，optical-off 11.59%，normalized optical dependence 96.14%；
electronic-skip-off 仍为 51.18%，证明共同起点的新电子残差保持零初始化。BP、FA-pretrained、
FA-random 的 seed 2026 已分别在 GPU 4/5/2 启动，结果待统一回填。

资源调度记录：发现 GPU 3 计算利用率为 0%、剩余显存足够后，使用同一正式入口在 GPU 3
并行启动 FA-random seed 2027。为保证长任务连续执行并留下可复现入口，新增
`commands/31_queue_a13_formal_method.sh`；它只等待当前 launcher 结束并逐项调用 command 29，
不修改配置或训练逻辑。GPU 4 已排队 BP seeds 2027/2028，GPU 5 已排队 FA-pretrained seeds
2027/2028，GPU 2 已排队 FA-random seed 2028。至此九个可训练的 method×seed 均已运行或排队，
NoFT 三 seed 已完成。

最终汇总由 `commands/32_wait_and_compare_a13_formal.sh` 在 CPU 侧等待固定的四方法乘三 seed
共 12 个结果，齐全后自动调用 command 30。comparison 输出除 test mean/std 外，还包括所有
光学/电子消融、光学依赖、相位漂移、门控、梯度对齐统计和预注册方法之间的逐 seed 配对差值。

P02 最终结果（2026/2027/2028 三个配对 seeds，均值 ± sample std）：

| 方法 | test | optical-off | electronic-skip-off | optical dependence | phase RMS drift |
|---|---:|---:|---:|---:|---:|
| NoFT | 51.18% ± 0.00 pp | 11.59% | 51.18% | 96.14% | 0.000 rad |
| BP | **72.30% ± 0.54 pp** | 13.70% | 32.06% | 94.06% | 0.304 rad |
| FA-pretrained | **71.44% ± 0.42 pp** | 12.76% | 32.87% | 95.51% | 0.295 rad |
| FA-random | 63.29% ± 0.91 pp | 15.01% | 16.37% | 90.59% | 0.480 rad |

逐 seed 配对差值：`BP - FA-pretrained = 0.86 ± 0.26 pp`；
`FA-pretrained - FA-random = 8.15 ± 1.32 pp`。FA-pretrained 的 epoch-0 分层 gradient cosine
均值/最小值约为 1.000/1.000；FA-random 三 seed 的均值为 0.224，逐运行最小层的平均值为
-0.203。FA-random 相位更新更大但性能更低，说明优势不能只用“更大更新量”解释。所有方法的
最小光学 gate 均保持不低于 0.50008。P02 支持在冻结 A13 高性能骨干上，预训练固定反馈接近
BP 且显著优于随机固定反馈；它仍然只覆盖理想数字光学环境。

下一阶段 P03 冻结上述 checkpoint，仅在推理光路注入部署非理想因素。预注册设计见
`DEPLOYMENT_ROBUSTNESS_PLAN.md`；P03-S 先在 validation split 筛选误差工作区间，不修改模型。

P03 已完成三 training seeds × 三 deployment seeds 的 test 确认。相位 0.15 rad、横向失配
0.125 pixel、探测器 5% RMS 和联合工作点下，FA-pretrained 相对 FA-random 的配对优势分别
为 4.69/3.35/7.89/4.20 pp；对应 BP 相对 FA-pretrained 的差距为 2.03/2.56/1.08/2.00 pp。
0.25 pixel 失配下三种高性能模型严重退化且 FA-pretrained - FA-random 反转为 -7.36 pp，定义为
装调失效边界而非有效部署工作点。完整协议、筛选过程、分层统计和限制见
`DEPLOYMENT_ROBUSTNESS_PLAN.md`。

## P04：部署偏移后的继续训练（2026-08-20 启动）

P03 不包含部署后反向传播。P04 修复 `detach phase_override` 不能继续训练且会绕过 FA 的问题，
新增可微部署位移：前向使用固定偏移后的当前光学算子；BP-current 使用当前精确 Jacobian；
FA-pretrained 只把跨层 error connector 固定为部署前最后一次训练的算子，局部相位梯度仍精确；
FA-random 使用同形状固定随机 connector。四组均从 P02 BP seed 2026 的同一个 endpoint 开始，
避免不同预部署 checkpoint 的性能和电子依赖混淆适配结论。

首轮为 validation-only：global/layerwise 两种几何分别测试 0.125/0.25 pixel，适配 10 epochs，
GPU 4/5 由 command 39 并行执行。正式协议、代码动作、指标和进入多 seed/test 的门槛见
`P04_DEPLOYMENT_ADAPTATION_PLAN.md`。
