# 架构审核与设计说明

## 1. 审核结论

新工程满足当前正式实验的主要要求：10 cm传播、17 µm逻辑输入、8 µm相位导出、正常振幅极性、四个有硬下限的光学融合门、较强 phase 更新、phase dropout、k-space约束、±16 pixel错位和逐层硬件微调。

它与旧工程 checkpoint 不兼容。原因不是 Qwen 主体变化，而是融合门由普通 sigmoid 改成了带0.10硬下限的参数化，训练超参数和鲁棒性分布也发生了变化。当前独立错位物理语义的checkpoint architecture为`..._v2`；修复前`..._v1`也会被加载器拒绝。正式 run 应从头训练。

## 2. 主干与形状

Qwen3-VL-Embedding-2B 负责产生冻结的视觉和语言 hidden states。新训练参数只属于紧凑替代模块和检索头。

```text
Vision hidden [Nv,1024]
  → Linear(1024,192) + LayerNorm
  → [Nv,192]
  → Block 1: 2D depthwise Mixer || Vision MoE4 expert optics
  → E1 + alpha_v1 * O1
  → Block 2: 2D depthwise Mixer || Vision global optics
  → LayerNorm(E2 + alpha_v2 * O2)
  → Linear(192,1024) + residual
  → Qwen main merger

Language hidden [Nl,2048]
  → Linear(2048,192) + LayerNorm
  → [Nl,192]
  → Block 1: causal depthwise Conv1D Mixer || Language MoE4 expert optics
  → E1 + alpha_l1 * O1
  → Block 2: causal depthwise Conv1D Mixer || Language global optics
  → LayerNorm(E2 + alpha_l2 * O2)
  → token mean+max pooling [384]
  → electronic retrieval readout
  → 64-D L2-normalized embedding
```

Vision 使用二维 Mixer，因为视觉 token 有明确的二维网格；Language 保持 causal 1D Mixer，避免错误地给语言 token 构造二维邻接关系。DeepStack 关闭。

## 3. 四个光学阶段

| 顺序 | 阶段 | 相位结构 | 对应电子支路 |
|---:|---|---|---|
| 1 | `vision_expert` | 2×2 MoE4 expert相位拼图 | Vision Mixer 1 |
| 2 | `vision_global` | 单张global相位 | Vision Mixer 2 |
| 3 | `language_expert` | 2×2 MoE4 expert相位拼图 | Language Mixer 1 |
| 4 | `language_global` | 单张global相位 | Language Mixer 2 |

一个 expert BMP 是四个独立 expert phase 参数在478×478有效区域内的物理拼图。因此硬件播放4张阶段相位 BMP，但模型内部共有 `4+1+4+1=10` 个独立 raw-phase 张量。

每个光学阶段执行：

```text
192维token
→ 非负电子编码
→ 224×224输入场
→ MoE4 fan-out或global全局场
→ 478×478有效相位
→ 518×518角谱传播，z=0.10 m
→ 478×478 CCD强度
→ 帧均值归一化、相对强度clip、log1p
→ 478→224 readout
→ 192维光学delta
```

不做背景扣除。实际 CCD 的绝对增益变化通过逐帧均值尺度归一化处理；这不会虚构不存在的暗场。

## 4. 光电融合门

四个阶段分别拥有独立标量门：

```text
alpha = minimum + (1 - minimum) * sigmoid(raw_gate)
minimum = 0.10
initial alpha = 0.20
fused = electronic + alpha * optical_delta
```

这保证电子网络不能把光学残差系数压到0。由于两条支路的张量范数仍由各自网络决定，`alpha>=0.10` 不是严格的光学功率占比；汇报时应使用“光学残差系数硬下限”这一表述。

## 5. phase训练

正式配置使用：

- `phase = 2π × sigmoid(raw_phase)`；
- `small_normal` 初始化，raw标准差0.02；
- phase LR `6e-3`；
- 80 epochs、cosine学习率计划；
- 前5个epoch联合 warmup，之后每2个epoch安排一次 phase-only epoch；
- 每4个epoch输出 phase preview；
- 训练态显式开启 block phase dropout，概率0.08、块大小8。

phase-only epoch 仍使用任务损失反向传播，只冻结电子、router和readout的优化器组。推理、BMP导出和实测 CCD 评估时 phase dropout 关闭。

`6e-3` 高于旧 strong-phase run 的 `4e-3`，属于有意增强相位运动的新实验。训练时必须同时观察：

- `phase_std_rad` 和相对初始化的 phase RMS；
- `phase_grad_rms`；
- raw sigmoid 饱和比例；
- 四组 phase 是否都有梯度；
- 固定 checkpoint 的检索 Top-1，而不是只看 preview 是否“花”。

更大的 phase 变化不自动意味着更高的检索准确率。

## 6. 鲁棒性措施

### 6.1 几何错位

每个expert/global物理阶段都会在完整518×518逻辑面上，独立采样三类±16 pixel整数位移：

- 输入复光场先在518面上平移，越界补0；
- phase modulation map独立平移，越界补单位相位调制，之后才与输入场相乘；
- 传播得到完整518强度后先平移，再裁出478×478 CCD ROI。该顺序允许原ROI外、518计算窗内的光进入偏移ROI，不能用“先裁478再零填平移”代替。

一个逻辑像素是17 µm，因此每个绝对位移的最大单轴幅度为272 µm。三次抽样彼此独立，所以输入场与phase mask在同一轴上的最坏相对错位是±32逻辑像素，即±544 µm；这不是把单项配置从±16改成±32。同一次抽样在一个batch内共享；expert和global阶段还会分别重新抽样。不包含旋转、非整数亚像素移动或4F倍率变化。

### 6.2 强度与噪声

仿真 CCD 训练扰动包括：

- 增益随机范围0.4～2.5；
- 相对offset上限0.05；
- 相对读出噪声0.015；
- 非负截断；
- 帧均值尺度归一化、relative clip=12、log1p。

### 6.3 k-space

角谱传播开启径向 k-space 截止，`theta_max=0.65°`，用于抑制数值模型中难以被实际有限孔径传输的高空间频率。

### 6.4 router

Vision和Language各有独立电子 top-k router，均为4专家选2。联合训练和硬件下游微调都加入：

```text
0.05 × router balance loss
+ 0.005 × router importance loss
```

正式配置在训练态给router logits加入标准差0.10的高斯探索噪声，避免接近均匀的
soft概率经过确定性top-k时长期饿死某些物理专家。`eval()`、固定checkpoint评估、
BMP导出和实验室推理均自动关闭该噪声，仍使用确定性top-2。

## 7. 检索损失

无 teacher KD。主体任务保持精简：

```text
1.0 × supervised contrastive loss
+ 1.0 × episodic prototype retrieval CE
+ 0.02 × CCD operating-point loss
+ 0.05 × router balance loss
+ 0.005 × router importance loss
```

硬件逐层微调使用 supervised contrastive、episodic prototype 以及 router balance/importance。测得的当前层 CCD 会截断该层 phase 的梯度，因此只更新当前实测边界之后仍有意义的电子模块、后续仿真模块和最终检索头。

代码还会建立参数级“采集合同”：凡是决定已播放振幅或已采CCD的adapter、router、当前phase、上游readout、Block-1及其融合门都必须冻结；若trainable集合与该合同相交会直接报错。特别是采完`language_expert`后，Language `input_adapter/input_norm`保持冻结，避免微调后的理论输入与已播放BMP不再对应。

## 8. 硬件坐标合同

```text
逻辑 amplitude: 478×478 @ 17 µm
→ 1:1 放入 1024×1024 @ 17 µm

逻辑 phase: 478×478 @ 17 µm
→ 先按配置纵向翻转
→ 依据像素中心物理坐标 nearest 栅格化到约1016×1016 @ 8 µm
→ 放入 1920×1200，默认中心(980,590)
```

振幅值是直接硬件命令：255代表白色/透光，0代表黑色/遮光，不能再执行历史版本的黑白反相。

## 9. 结果解释限制

- 完整四层实测结果必须包含四层全部 CCD，并按顺序微调。
- `--upstream-source simulation` 的最后一层快速模式只验证第四层实测影响。
- 融合门0.10是系数下限，不是物理能量占比。
- k-space和±16平移提高的是对已建模扰动的鲁棒性，不能覆盖未建模的严重旋转、倍率误差、散斑漂移或相机非线性。
- phase SLM中心 `(980,590)` 属于导出标定量，改变中心只需重新导出，不需要重新训练。
