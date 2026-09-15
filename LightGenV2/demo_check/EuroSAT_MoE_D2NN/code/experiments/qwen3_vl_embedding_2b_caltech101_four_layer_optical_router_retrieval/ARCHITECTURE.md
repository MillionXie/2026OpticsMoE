# 四层光电检索中的电子 Router 与光学 Router

## 1. 这次实验回答什么问题

本工程建立在
`qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5`
之上，只研究两个问题：

1. 原来的电子 Router 能否换成一张相位权重和一次 CCD 曝光完成的光学 Router；
2. 四个专家究竟应该激活 1 个、2 个，还是全部 4 个。

除 Router 及其幅度权重口径外，Qwen 前端、两层 Vision 光电融合、两层
Language 光电融合、64 维检索头、Caltech101 十类划分和损失主体均保持一致。

这里的“光学 Router”准确地说是：

> 光路负责把一个 224×224 特征场变成四个区域的 CCD 能量；电子端只读取四个数，
> 再执行 softmax 和 top-k。

因此它不是完全不含电子运算的全光 Router，但已经去掉了原电子 Router 的
`14×14 → Linear(196,4)` 分类网络。

## 2. 共同的物理几何不会改变

| 项目 | 固定值 |
|---|---:|
| 波长 | 532 nm |
| 逻辑像素 | 17 μm |
| 传播距离 | 10 cm |
| 数值传播画布 | 518×518 |
| CCD 有效 ROI | 478×478 |
| 单个专家 | 224×224 |
| 专家排列 | 2×2，共 4 个 |
| 专家间距 pitch | 254 像素 |
| 专家之间空隙 | 30 像素 |
| k 空间截止角 | 0.65° |

在 518×518 数值画布中，478×478 有效区域四周各有 20 像素保护带。四个
224×224 专家按行优先排列：左上、右上、左下、右下。改变 top-k 只会令未选
专家的输入振幅为零，不会改变这套版面、ROI、传播距离或 SLM 中心。

## 3. 原电子 Router 到底是什么

Vision 和 Language 各有一个独立 Router。每个 Router 只在该模态的 Expert
阶段计算一次；同一模态随后的 Global 阶段复用完全相同的专家编号和权重。

其输入已经是光输入适配器产生的非负场：

```text
latent feature [B,N,192]
→ Linear 192→224
→ LayerNorm(224)
→ Softplus
→ 补齐 token 行
→ RMS 标准化
→ amplitude field [B,224,224]
```

电子 Router 的内部结构为：

```text
[B,224,224]
→ AdaptiveAvgPool2d(14,14)
→ flatten
→ [B,196]
→ non-affine LayerNorm(196)
→ Linear(196,4)
→ logits [B,4]
→ softmax(logits / 2.0)
→ top-k + 所选概率重新归一化
→ routing weights [B,4]
```

`Linear(196,4)` 每个模态有 `196×4+4=788` 个参数，Vision 和 Language 合计
1,576 个。warmstart5 中这两个 Router 不是随机的：它们先从 robust checkpoint
载入，随后在 Stage A 和 Stage B 都进入了优化器。当前 checkpoint 中它们已经
训练过，只是概率整体仍然接近均匀分布；“接近均匀”不等于“没有训练”，hard
top-k 仍会依据细小的相对差异选择专家。

接近均匀与下列设置有关：温度为 2、router balance loss、router importance
loss，以及当前任务本身并不强迫单个样本形成非常尖锐的专家概率。后续比较必须
同时报告概率熵、专家负载和 top-k margin，不能只看最终选择编号。

## 4. 光学 Router 的一次计算

光学 Router 接收与电子 Router 完全相同的 `[B,224,224]` 非负振幅场。

### 4.1 输入、相位和传播

```text
输入振幅 [B,224,224]
→ 放在 518×518 画布正中心
→ 与一张可训练 224×224 phase-only Router mask 共面相乘
→ 532 nm、17 μm、10 cm angular-spectrum propagation
→ 从 518×518 中截取原来的 478×478 CCD ROI
→ router CCD [B,478,478]
```

中心 224×224 输入在 518 画布中的范围为 `[147,371)`；换算到 478 ROI 的局部
坐标为 `[127,351)`。导出硬件相位时，224×224 Router mask 被居中放进一张
478×478 逻辑相位图，再沿用正式工程相同的 17 μm→8 μm 栅格映射、SLM 中心和
翻转合同。

相位使用：

```text
phase = 2π × sigmoid(raw_phase)
```

初始化不是毫无方向的常数相位，而是面向四个探测区的四束相位全息初值，随后由
检索损失和 Router 辅助损失继续训练。

### 4.2 四个固定探测区

完整 CCD ROI 始终是 478×478。Router 只在其中积分四个固定的 59×59 小区域，
没有裁掉或改小正式 CCD ROI。

每个坐标区间均采用 Python 的半开区间 `[start,end)`：

| 专家 | x 区间 | y 区间 | 区域大小 |
|---|---:|---:|---:|
| 0，左上 | `[164,223)` | `[164,223)` | 59×59 |
| 1，右上 | `[255,314)` | `[164,223)` | 59×59 |
| 2，左下 | `[164,223)` | `[255,314)` | 59×59 |
| 3，右下 | `[255,314)` | `[255,314)` | 59×59 |

四区中心相对 CCD 中心为 ±45.5 像素；最远的对角方向约为 0.627°，仍位于正式
0.65° 径向 k 空间截止角内。四区顺序与 MoE4 专家的行优先顺序完全一致。

对四区分别求原始强度和，得到：

```text
router CCD [B,478,478]
→ four detector energies [B,4]
→ 四个能量减去样本内均值
→ 除以四个能量的样本内 RMS（无仿射参数）
→ softmax(logits / temperature)
→ top-k
→ selected weights [B,4]
```

该四数标准化同时消除统一乘性增益和四区共同偏置；它不是 CCD 图像的背景扣除，
也不会修改保存的原始 CCD。配置还保留 `log_energy_fraction` 作为可选敏感性对照，
但五份正式 release 配置固定采用 `standardized_region_energy`。
程序还记录“四个探测区捕获的能量 / 整张 CCD 能量”，并通过 capture loss 防止
相位把绝大多数光送到四区之外。

## 5. 为什么必须采用时分流程

当前只有一套振幅 SLM、相位 SLM 和 CCD。只有 CCD 先测出 Router 的四个强度，
电子端才能知道应当把下一次振幅输入复制到哪些专家。因此在不增加光学器件的前提
下，不可能用同一次曝光既完成条件路由，又完成被选专家的正式传播。

一张样本的物理顺序由原来的四次 CCD 采集变为六次：

```text
1. Vision Router
   中心 Vision 振幅 + Vision Router phase
   → CCD四区 → softmax/top-k

2. Vision Expert
   按上述结果生成2×2 routed amplitude + Vision Expert phase
   → CCD → 电子读出并与 Vision Mixer Block 1 融合

3. Vision Global
   融合结果重新编码；复用同一 Vision routing + Vision Global phase
   → CCD → 电子读出并与 Vision Mixer Block 2 融合

   → 冻结的 Qwen main merger，把196个Vision token合成49个2048维图像token

4. Language Router
   中心 Language 振幅 + Language Router phase
   → CCD四区 → softmax/top-k

5. Language Expert
   2×2 routed amplitude + Language Expert phase
   → CCD → 电子读出并与 Language Mixer Block 1 融合

6. Language Global
   复用同一 Language routing + Language Global phase
   → CCD → 电子读出并与 Language Mixer Block 2 融合

   → mean/max pooling → 384→64 → L2 normalization → cosine retrieval
```

新增的两次 Router 曝光不是新的特征处理层，所以模型仍称“四层光电网络”；更准确
的硬件说法是“四个特征 CCD 边界 + 两个 Router CCD 探针”。

## 6. Top-k=1、2、4 分别意味着什么

### k=1

只有一个专家有输入。优点是选择最明确、串扰小；缺点是一次错路由就没有第二专家
兜底，通常最脆弱。此外，普通 hard top-1 在重新归一化后所选权重恒等于 1，任务
损失无法通过这个常数权重训练 Router，因此正式训练必须采用 STE。

### k=2

保留两个专家，可在专门化和容错之间折中。这是 warmstart5 原先采用的设置，但本次
实验不会先验断言它一定最好，而是与 k=1、k=4 在同一功率合同下比较。

### k=4

四个专家全部有输入，不再有离散选择错误，但专家专门化可能变弱，且零级衍射、
SLM 漏光和跨区域串扰会同时影响所有分支。

四个专家在 SLM 上是空间并行的，因此减少激活专家数量不会减少 10 cm 自由空间传播
时间。top-k 主要影响光功率分配、专家专门化、容错和串扰，而不是传播 FLOPs。

## 7. 两种比较口径必须分开

### 7.1 Legacy anchor：复现旧模型语义

`electronic_legacy_topk2_anchor.yaml` 保持 warmstart5 的原始定义：

```text
top_k = 2
expert amplitude scale a_e = sparse routing weight w_e
```

它用于回答“新工程有没有正确复现旧的电子 Router”，不能拿来独自证明哪一个 k 更
好。因为在 amplitude-domain 中，总专家输入功率满足：

```text
P ∝ Σ a_e² = Σ w_e²
```

概率接近均匀时，k=1、2、4 的功率大约分别是 1、1/2、1/4。直接比较会把专家数和
光功率两个变量混在一起。

### 7.2 正式消融：power_l2 + STE

正式 k=1/2/4 消融统一使用：

```text
sparse weights: w
amplitude scale: a = w / sqrt(Σ w² + eps)
```

因此每个样本都有：

```text
Σ a² = 1
```

也就是三组实验的总专家输入功率相同，仅改变激活专家数量和相对分配。工程内配置名
使用 `electronic_power_topk1/2/4`；其中 `power` 指的就是这项 L2 功率守恒。

正式消融还使用 straight-through estimator（STE）：

- 前向严格使用 hard top-k，未选专家振幅为零，和真实播放一致；
- 反向使用 dense softmax surrogate 给 Router 梯度；
- 尤其避免 k=1 因前向权重恒为 1 而彻底失去任务梯度。

Legacy anchor 与 power_l2 正式消融属于两个不同问题，结果表中必须分栏，不能把旧
anchor 的 Top-2 数值与 power_l2 的 k=1/4 当成同口径排名。

## 8. 光 Router 与电子 Router 的匹配系统比较

光/电 Router 的主比较都采用：

- 同一个 warmstart5 起点；
- 相同 Caltech101 划分和随机种子；
- 相同 Qwen、电子 Mixer、四张 feature phase、CCD readout 和 64 维头；
- 相同 top-k=2；
- 相同 power_l2 振幅合同与 STE；
- 相同训练步数和检索主损失。

核心变化是四个 Router logits 的来源：

```text
电子：Pool14×14 → LN196 → Linear196→4
光学：中心振幅 → Router phase → 10cm → CCD四区能量
```

这不是“只替换一个算子、其他训练细节完全相同”的严格单因素实验。光学 Router
还必须使用确定性的四束相位初值、独立的相位学习率、位移/phase-dropout 扰动和
四区 capture loss；其参数量与物理曝光次数也不同。因此应把
`electronic_power_topk2` 对 `optical_power_topk2` 表述为等数据、等步数、等功率合同的
**系统级比较**，不能把差异全部归因于“光优于或劣于一个 Linear 层”。另外，公共
warmstart5 body 曾与旧电子 Router 共同训练；本工程虽重置两种 Router，却不能让
公共 body 变成完全 backend-neutral，这也是结论边界。

除 Top-1、Top-3、MRR 外，还必须报告：

- 四专家平均负载与 Router entropy；
- 第 k 名与第 k+1 名的 margin；
- 光/电 Router top-1 agreement 与 top-k Jaccard；
- 四区概率的 JS/KL 差异；
- Router 四区能量捕获率；
- 位移、增益和噪声扰动下的 route stability；
- 参数量，以及光 Router 新增两次 SLM/CCD 时序的实际延迟。

## 9. 参数与硬件代价

电子 Router 每个模态 788 参数。当前光 Router 每个模态使用一张 224×224 相位，
即 50,176 个可训练相位参数；Vision 和 Language 共 100,352 个。导出时虽然载体是
478×478，相位的可训练区域仍是居中的 224×224，外围只是固定填充。

光 Router 不增加镜片、SLM、CCD或光程，但每张样本多两次相位切换、两次振幅播放和
两次 CCD 采集。因此应同时给出“网络精度”和“端到端硬件延迟”，不能只以光传播本身
近乎瞬时为由忽略 SLM/CCD 时序。

## 10. 结果解释边界

- 光 Router 在仿真中可通过反向传播训练；换成实测 CCD 后，普通离线微调不能穿过
  真实光路反向更新物理相位，只能更新下游电子参数或使用额外的物理梯度方法。
- 四区响应不要求主观上成为四个完美光点；Router 的判据是固定区域积分、margin 和
  扰动稳定性。
- CCD 图和探测区坐标都应处于 canonical 478×478 模型方向。不能一边使用已经完成
  homography/翻转的图像，一边再次按 raw sensor 方向解释四区。
- sealed test 只在训练结束后显式执行一次；各 epoch 不以 test 结果选 checkpoint。
