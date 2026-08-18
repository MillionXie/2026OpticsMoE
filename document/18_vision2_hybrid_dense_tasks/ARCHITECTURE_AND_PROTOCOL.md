# 架构、训练协议与数据流

## 三项稠密任务的共同主干

三项实验的实际前向路径相同：

```text
224×224 image
  → frozen Qwen3-VL patch/position stem
  → spatial tokens, hidden 1024
  → LayerNorm + Linear(1024→192)
  → [2D Mixer block 1 || MoE4 expert optical stage]
  → electronic_1 + sigmoid(gate_1) × optical_delta_1
  → [2D Mixer block 2 || global optical stage]
  → LayerNorm(electronic_2 + sigmoid(gate_2) × optical_delta_2)
  → restore 192-channel spatial feature map
  → task-specific progressive decoder
```

关键事实：

- Qwen3-VL 的 patch/position stem 冻结；原生 24 个 vision Transformer block 均不执行。
- DeepStack 关闭，Qwen merger 输出也不供 dense head 使用。
- 1024→192 adapter、两个电子 Mixer、两级光路、CCD readout、融合门控和任务 head 从 epoch 1 联合训练。
- 不使用 attention，不使用 teacher/KD，也没有电子预训练 checkpoint 依赖。
- 电子 Mixer 的 token mixing 为 `3×3 depthwise Conv2D + pointwise projection`，再接通道 MLP；两层宽度均为 192，MLP expansion 为 2。
- 每级融合都是残差式增量：电子输出保持主路径，光学 readout 形成 `optical_delta`，再由独立 sigmoid gate 控制幅度。

## 有效光路

每个任务实际使用的是 Vision2 MoE4 双融合，而不是继承配置中可能出现的旧 MoE16 字段：

- 第一级：4 个 `224×224` expert phase mask，按 `2×2` 排布，router 为 top-2；
- 第二级：一个 `478×478` global phase mask；
- 有效区域 `478×478`，仿真 canvas `518×518`；
- 波长 `532 nm`，像素间距 `16 μm`，传播距离为 `0.10 m`；
- k-space 扰动开启，`theta_max=0.65°`；
- phase block dropout `p=0.05`，block size `8`；
- 训练扰动：输入/phase/CCD 最大偏移 8 px，CCD gain 0.5–2.0，offset fraction 0.03，read-noise fraction 0.01；
- CCD 先按单帧均值做相对强度归一化、截断并 `log1p` 压缩，没有虚构背景帧或背景扣除；
- CCD operating-point loss 权重为 `0.02`。

相位学习率在三项任务中均为 `1e-4`。电子 Mixer 为 `1e-4`，router/readout 为 `5e-5`，任务 head 为 `3e-4`。

## Caltech101-10 四级检索架构（仿真）

Caltech101 不是稠密 decoder 任务，也不与前三项共用完全相同的前向。其冻结证据对应如下四级检索网络：

```text
image
  → frozen Qwen3-VL vision encoder → [Nv,1024]
  → Linear/LayerNorm 1024→192
  → [Vision Mixer2D block 1 || Vision MoE4 expert optics]
  → electronic_1 + sigmoid(gv1) × optical_delta_1
  → [Vision Mixer2D block 2 || Vision global optics]
  → electronic_2 + sigmoid(gv2) × optical_delta_2
  → Linear 192→1024 → frozen Qwen main merger
  → frozen Qwen language hidden [Nl,2048]
  → Linear/LayerNorm 2048→192
  → [causal Mixer1D block 1 || Language MoE4 expert optics]
  → electronic_1 + sigmoid(gl1) × optical_delta_1
  → [causal Mixer1D block 2 || Language global optics]
  → electronic_2 + sigmoid(gl2) × optical_delta_2
  → mean+max pooling [384]
  → LayerNorm → Linear 384→64 → L2 normalization
```

关键口径：

- Qwen3-VL-Embedding-2B 使用预训练权重并冻结，DeepStack 关闭；四级 hybrid 模块和 64-D retrieval readout 不依赖电子预训练 checkpoint，从随机初始化开始训练。
- Vision 电子支路为两层 `3×3 depthwise Conv2D + pointwise + channel MLP`；Language 电子支路为两层 causal `kernel=5 depthwise Conv1D + pointwise + channel MLP`；均不使用 attention。
- 光学阶段顺序为 Vision MoE4 expert、Vision global、Language MoE4 expert、Language global。expert 有效单元为 `224×224`，MoE4 的 `2×2` 有效区域为 `478×478`，传播 canvas 为 `518×518`。
- 每一级光学 readout 与同宽 192-D 电子输出经独立 sigmoid gate 做残差融合。两个 MoE4 router 为 top-2；不添加 router balance/importance loss。
- 模型报告记录可训练参数共 `2,683,709`，包括两模态 adapter/Mixer、router、四级光学/readout、融合门和最终 64-D head；冻结的 Qwen 参数不计入。
- 本次只整理 simulation。硬件 session、真实 CCD、SLM 重建和逐层硬件微调均不在本证据包内。

最重要的限制是：该 run 保存的 resolved config 中 `phase_learning_rate=0.0`。四组 phase 参数收到非零梯度，但 60 个 epoch 的 `phase_delta_run_rms_rad` 始终为 0，所以实际学习的是电子模块、router、CCD readout、融合门和 embedding head，phase mask 保持初始化值。当前源码 release YAML 已改为 `2e-5`，但那是后续配置，不能回写到本次历史结果。

## 任务 decoder

| 任务 | Decoder | 输入与输出 | 可训练 head 参数 |
|---|---|---|---:|
| SALICON | progressive saliency density decoder | 192 通道空间特征 → 224×224 单通道 saliency logits | 85,412 |
| ISIC 2016 | progressive lesion boundary decoder | 192 通道空间特征 → 224×224 单通道 lesion logits | 85,845 |
| LSP | progressive pose heatmap decoder | 192 通道空间特征 → 14×56×56 joint heatmaps | 133,425 |

SALICON/ISIC head 先做 `192→128` 投影和两层 depthwise residual block，再逐级上采样为 `96→64→32→16` 通道。ISIC 比 SALICON 多一层 boundary refinement。LSP 使用 `192→160` 投影、两层 residual block、`128→96` progressive upsampling，最后预测 14 个关节点热图。

总可训练参数分别为：SALICON `1,218,211`、ISIC `1,218,644`、LSP `1,266,224`。冻结的 Qwen checkpoint 参数不计入这些数字。

注意：模型报告中的 `router_parameters: 0` 是参数分类统计的遗留问题；实际 trainable list 中存在 `4×196` gate weight 和 4 个 bias，即 788 个 router 参数。后续写参数表时应以 trainable parameter list 为准。

## Loss

### SALICON

```text
KLD
+ 0.5 × (1 − CC)
+ 0.25 × (1 − SIM)
− 0.1 × NSS
+ 0.02 × CCD operating-point loss
```

`map_kd_weight=0`，router balance/importance 和 phase-DC loss 均为 0。

### ISIC 2016

```text
1.0 × BCE
+ 1.0 × Dice loss
+ 0.75 × soft-IoU loss
+ 0.25 × boundary loss
+ 0.02 × CCD operating-point loss
```

router balance/importance 和 phase-DC loss 均为 0。

### LSP

```text
1.0 × masked heatmap loss
+ 0.1 × coordinate loss
+ 0.02 × CCD operating-point loss
```

teacher distillation、router balance/importance、router-response consistency 和 phase-DC loss 均为 0。

### Caltech101-10 retrieval

```text
1.0 × supervised contrastive retrieval loss
+ 1.0 × episodic gallery/prototype retrieval CE
+ 0.02 × CCD operating-point loss
```

teacher/KD、relational KD、teacher-gallery CE、router balance/importance/response consistency 和 phase-DC 均为 0。temperature 为 `0.07`，gallery temperature 为 `0.15`；gallery prototype 对 embedding 主干停止梯度。训练使用 cosine schedule、5% warm-up、gradient clip norm 1.0 和 EMA decay `0.995`。

## 数据与选模协议

### SALICON

- train：官方 train2014，10,000 张；
- validation：官方 val2014，5,000 张，image ID 与 train 不重叠；
- 训练 60 epoch，checkpoint 按 validation CC 最大选择；
- 私有 test 没有公开标注，本次没有报告。

### ISIC 2016

- 官方 train 900，官方 test 379，无重叠；
- 训练 100 epoch，`evaluate_test_each_epoch=false`；
- checkpoint 只按 training loss 最小选择；
- 训练结束后加载 epoch 96 checkpoint，在官方 test 上运行一次正式评估；
- 因此 history 内的逐轮 `test_* = 0` 仅是占位字段。

### LSP

- train：LSPET 9,428 + LSP 前 1,000，共 10,428；
- test：LSP 后 1,000；
- 训练 150 epoch，checkpoint 按 training loss 最小选择；
- 实现仍在每个 epoch 计算 test，因此所有 test 曲线都只能作为 monitored-test 诊断；
- 正式引用 epoch 148，不能根据 test 峰值改选 epoch 149。

### Caltech101-10

- 类别：`airplanes`、`Motorbikes`、`Faces`、`Leopards`、`accordion`、`grand_piano`、`scorpion`、`sunflower`、`watch`、`yin_yang`；
- seeded per-category shuffle（seed 42）后划分互不重叠的 gallery、train 和 test；
- train 2,625 张；gallery 每类 3 张、共 30 张；test query 每类 20 张、共 200 张。P×K sampler 为凑齐 batch 使 history 的 `samples=2630`，不代表多出 5 张唯一训练图；
- gallery 同类 3 张 embedding 取均值形成 class prototype，query 与 10 个 prototype 做相似性检索；
- 训练 60 epoch，checkpoint 按 training total loss 最小选择，得到 epoch 54；
- test query 在每个 epoch 被监控，因此曲线和峰值只作诊断；正式值必须引用不看 test 的 epoch 54 选模规则；
- epoch 54 raw Top-1/Top-3/MRR 为 `0.88500/0.97500/0.92676`，EMA 为 `0.89000/0.97500/0.93009`。

## 需要继续核验的架构问题

- LSP 的 normalized per-sample router entropy 很快趋近 0，说明单样本路由概率高度尖锐；这不等价于“所有样本都选同一个专家”。是否发生全局 expert collapse，必须另外统计每个 expert 的跨样本占用率。
- 目前 router regularization 权重为 0，这是有意保留的当前设置。任何后续调整都应作为独立消融，不要覆盖本次 evidence。
- 三项结果来自不同任务 head，不能仅凭总参数相近就宣称 head 完全参数匹配。
- Caltech101 的 phase LR 为 0 是本次 run 的事实，不是推荐设置；以后训练可学习 phase 的结果应建立新 evidence variant，不得覆盖本快照。
- Caltech101 epoch 54 的 normalized router entropy 已很低（Vision `0.00175`、Language `0.02484`），最大 expert importance 分别为 `0.99969` 和 `0.99540`。虽然两个 router 的四个 expert 都至少被选中过，仍显示一个 expert 几乎总是主导；不能据此声称 MoE4 已实现负载均衡。
