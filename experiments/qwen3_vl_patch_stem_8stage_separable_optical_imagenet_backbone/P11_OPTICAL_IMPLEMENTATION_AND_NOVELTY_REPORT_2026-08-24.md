# P11 光路实现、相关工作与论文创新性评估

更新日期：2026-08-24

对象：P11 `qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone`

目的：回答 P11 如何由真实光路实现、哪些部分可能构成论文创新、哪些表述会被现有工作否定，以及下一步应如何验证。

## 0. 结论先行

P11 **可以设计成真实光路**，但不能直接用普通二维自由空间传播照搬。其代码要求：

- token stage 只沿 token 轴衍射，feature 轴保持理想成像；
- channel stage 只沿 feature 轴衍射，token 轴保持理想成像；
- 每层经过相机探测、电子归一化、轻量电子残差和门控后，再把非负振幅重载到下一层。

因此它准确的硬件定位是：

> **带轴选择性衍射主支路和轻量电子残差的多级 OEO 混合光电 backbone**，而不是八片相位板连续级联的全光网络。

最建议的第一代原型不是一次铺开八套光路，而是做一个可复用核心：

```text
电子固定路由/转置
        ↓
三 bank 振幅加载 → 学习相位面 → 4f 系统 + 单轴频域相位 H_y → 相机
        ↑                                             ↓
        └──── 归一化 + Slim Mixer + 光电门控 + 下一层重载 ────┘
```

token stage 直接使用该核心；channel stage 在现有 OEO 边界对数据和相位做一次**固定转置**，再复用同一个单轴核心。八层就是八次曝光/重载。这个方案与当前代码最接近、器件最少，也最容易先证明算子确实可实现。

创新性方面，本次原始文献检索没有发现与 P11 **精确相同**的公开结构：把预训练视觉 stem 输出的 `token × hidden-feature` 张量直接映射为二维光场，并交替用 token 轴和 feature 轴的可训练相位衍射完成两类 mixing。但是，一维傅里叶光学、正交轴处理、光学 token/空间处理、光学通道混合、光学 Transformer 和空间—通道分解都已有先例。因此：

- P11 仅有数值模型时，创新强度属于“有想法、证据不足”；
- 完成同预算四组、真实轴选择性光路、实测数字孪生和下游迁移后，可以成为架构与系统创新；
- 若再与本课题的预训练固定反馈、部署漂移和低成本校准结合，论文主线会明显强于单独讲 P11。

建议名称为：

> **Semantics-Aligned Separable Diffractive Token–Channel Mixer**
>
> 语义轴对齐可分离衍射 Token–Channel Mixer（简称 OTCM）

更保守的论文表述是 `MLP-Mixer-inspired axial diffractive backbone`，不能直接称为 `optical MLP-Mixer`。

## 1. P11 当前到底计算了什么

以下内容来自对 [P11 model.py](model.py)、[P11 架构说明](ARCHITECTURE.md)、父实验 P09、Qwen stem 和公共光学算子的逐项源码审计。当前本地和服务器对应源码已经核对一致。

### 1.1 从 RGB 图像到二维光场

```text
RGB, 224×224
  → 冻结 Qwen Patch/Position Stem
  → 196 tokens × 1024 hidden
  → 电子 TokenAdapter: LN + Linear(1024→224) + Softplus + RMSNorm
  → 196 × 224
  → token 轴补 28 行零
  → 224 × 224
  → 复制为 3 个 latent optical banks
```

需要特别澄清：

1. 三个 bank **不是 RGB 三波段**。RGB 已经在冻结 Qwen patch convolution 中融合；三个 bank 是相同初值、之后由独立相位掩模逐渐分化的三条潜在光路。
2. 没有加载 Qwen Transformer、attention 或语言模型；只使用冻结的 Patch/Position Stem。
3. `Softplus` 使首层输入非负，后续每层也经过 `ReLU`，所以当前版本不需要解决一般 signed hidden-state 的光场编码问题。
4. 模型中的数值表示下一层的**目标场振幅**。如果实验设备的 LUT 控制的是光强而非场振幅，需要做平方根和器件响应标定，不能把数组值直接当灰度加载。

### 1.2 token 与 feature 的物理坐标

P11 将一个 bank 的二维光场定义为：

```text
纵轴 y：224 个 token modes（前 196 个有效，后 28 个 padding）
横轴 x：224 个 learned feature modes
```

Qwen 的 196 个 token 原生采用 2×2 block-major 顺序。只在 token 光学层之前，P11 用固定置换把它恢复为 14×14 图像的 row-major 顺序；层后再逆置换回 Qwen 顺序。该置换发生在已经存在的相机读出/重载边界，不是学习得到的电子矩阵乘。

这里也有两个必须如实报告的限制：

- token 轴实际是长度 196 的一维 raster 序列，不是真正的二维 14×14 局部卷积；
- 28 个 padding mode 在经过衍射后会被激活，可能充当额外隐状态，目前还没有专门消融。

### 1.3 八层轴选择性光学算子

八层顺序固定为：

```text
[Token-axis → Channel-axis] × 4 macroblocks
```

每层、每个 bank 有一张独立的 `224×224` 可训练相位，因而：

```text
8 layers × 3 banks × 224 × 224 = 1,204,224 optical phase parameters
```

每层相位为：

```text
φ_l = 2π · sigmoid(raw_phase_l)
```

令输入非负振幅为 `A_l`，则一个光学支路为：

```text
U_l = A_l · exp(i φ_l)
U'_l = F_axis^{-1}(H_axis · F_axis(U_l))
I_l = |U'_l|²
O_l = RMSNorm(ReLU(Standardize(I_l)))
```

其中：

- token stage 的 `H_T(f_y)` 只依赖 `f_y`，沿 feature/x 方向恒定；
- channel stage 的 `H_C(f_x)` 只依赖 `f_x`，沿 token/y 方向恒定；
- 当前参数为波长 532 nm、有效像素间距 16 μm、有效传播距离 50 mm。

虽然代码写成 `FFT2 → H_axis → IFFT2`，但它严格等价于只在一个轴做一维传播、另一个轴做恒等 relay。普通自由空间的传递函数同时依赖 `f_x` 和 `f_y`，因此普通 50 mm 自由传播**不能**直接实现 P11。

逐线看，token stage 对每一个 feature 列独立执行：

```text
u'[:, c] = P_T · diag(exp(i φ[:, c])) · u[:, c]
```

channel stage 对每一个 token 行独立执行：

```text
u'[t, :] = P_C · diag(exp(i φ[t, :])) · u[t, :]
```

`P_T/P_C` 是固定的轴选择性传播。因为二维相位允许每一列/行拥有不同相位轮廓，P11 也不等价于标准 MLP-Mixer 的共享 MLP；单张相位面加固定传播更不能无条件实现任意稠密矩阵。

### 1.4 每层保留了哪些电子计算

每个 stage 的电子旁路仍是 width-96 `Slim Spatial Token Mixer`：

```text
224 → 96 adapter
  → true 14×14 depthwise 3×3 spatial mixer + 第一次门控残差
  → 96→192→96 channel MLP + 第二次门控残差
  → 96 → 224 adapter
```

每层的三个 bank 共享该层电子 mixer，但八层之间不共享。八层电子残差合计 733,472 个可训练参数。最后按：

```text
A_{l+1} = α_l · O_l + (1 - α_l) · E_l
α_l = 0.5 + 0.5 · sigmoid(g_l) ≥ 0.5
```

融合后再重载。`α≥0.5` 只表示两个已归一化分支的数值融合系数，不表示光学能耗、MAC、延迟或硬件工作量超过 50%。

因此不能宣称：

- P11 是 fully optical backbone；
- 所有 token/channel mixing 都由光完成；
- 55.51% 光学参数占比等于 55.51% 光学计算或能效占比。

更准确的描述是：

> 光学主支路承担受物理约束的轴向全局 mixing，电子旁路提供受限容量的局部空间和通道修正。

## 2. 三种可实现光路

### 2.1 方案 A：可复用单轴 4f 核心 + 固定数字转置（首选）

硬件结构：

```text
532 nm laser
  → 扩束 / 偏振控制
  → amplitude SLM 或 DMD：加载 3-bank 输入振幅
  → phase-only SLM：加载当前层 3 张 learned phase
  → 4f lens 1
  → Fourier plane phase filter：H_y(f_y)，沿 f_x 恒定
  → 4f lens 2
  → sCMOS camera：测量 |U'|²
  → ADC / PC 或 FPGA：标准化、ReLU、RMS、电子旁路、门控
  → 下一层振幅重载
```

token stage：

```text
Qwen order → 固定 row-major permutation → 单轴核心 → inverse permutation
```

channel stage：

```text
场和相位固定转置 → 同一个单轴核心 → 输出转置回来
```

优点：

- 一个固定的 `H_y` 核心即可复用八层；
- 数学上最接近当前 `F_axis^{-1} H_axis F_axis`；
- 转置只是 OEO 边界的地址重排，不引入大规模电子 TF；
- 最适合先做一层、一个 macroblock，再逐步扩到八次循环。

代价：

- 八层需要八次相机曝光和八次振幅重载；
- 实际速度受 SLM/DMD、相机和 ADC/DAC 限制，而不是受光传播时间限制；
- 论文必须把全部 OEO 开销纳入延迟和能耗统计。

### 2.2 方案 B：两套正交 4f/柱面核心

搭建一套 y-axis/token 核心和一套旋转 90° 的 x-axis/channel 核心，避免电子转置：

```text
Token core:   y 方向传播/FT，x 方向 1:1 imaging
Channel core: x 方向传播/FT，y 方向 1:1 imaging
```

可以用：

- 2D 4f 系统，在 Fourier plane 分别放置只依赖 `f_y` 或 `f_x` 的相位滤波器；或
- 柱面/像散 relay，使活动轴的 ABCD 矩阵近似自由传播 `B≈z_eff`，非活动轴近似成像 `B≈0`。

该方案光学故事更直接，但对准、旋转、倍率匹配和正交轴泄漏都更难。一个孤立柱面透镜并不足以保证“一个轴 50 mm 传播、另一个轴严格恒等”，需要完整的各向异性 relay 设计。

### 2.3 方案 C：八层完全展开的流水光路

理论上可以把八个 learned-phase stage、八个探测/重载节点完全铺开，以提高吞吐。但当前每层还有电子残差、全局统计、门控和重载，真正展开需要大量相机、调制器和同步电路，成本与校准难度很高。

当前不建议直接做。只有单核心循环证明精度、轴泄漏和多次重载误差都可控后，才值得评估两级或八级展开。

## 3. 推荐原型的具体落地方式

### 3.1 三个 bank

优先采用同一 SLM 上的**空分 tile**：

```text
[bank 1, 224×224]  guard  [bank 2, 224×224]  guard  [bank 3, 224×224]
```

以 16 μm 有效采样计算，每个 tile 宽约 3.584 mm，三个 tile 合计约 10.75 mm，另加 guard band。实验中要通过 relay magnification 把 SLM 原生 pixel pitch 映射到模型的 16 μm 有效 pitch。

不建议第一版使用 RGB/波分复用，因为不同波长会改变传播传递函数、相位 LUT 和色散，反而引入不必要变量。

### 3.2 learned phase 与固定传播相位

需要区分两个相位面：

1. 空间面的 `φ_l(y,x)`：每层可学习，三 bank 不同；
2. Fourier plane 的 `H_y`：固定轴向传播相位，所有 token/channel stage 在转置复用方案中相同。

`z_eff=50 mm` 在 4f 实现中是写入 `H_y` 的**有效传播相位**，不要求两个平面真的相距 50 mm。Fourier plane 的物理坐标与空间频率由焦距、波长和像素间距共同决定，必须按实验器件重新标定。

如果只做固定波长和固定 z，`H_y` 可以最终制作为固定 DOE；原型期建议先用可编程 phase SLM，以方便校准。

### 3.3 相机读出与重载

当前软件 stage 的操作不是简单 `camera intensity → 下一层`，而是：

```text
I → 每 bank 全平面 mean/variance 标准化 → ReLU → RMSNorm
  → 与电子 Slim Mixer 按 gate 合并
  → 把合并后的非负数组作为下一层目标振幅加载
```

实验时必须建立：

- camera raw counts → linear optical intensity 的 dark/flat-field/gamma 标定；
- 目标场振幅 → amplitude SLM/DMD 灰度的 LUT；
- 曝光、饱和、零级光和有效动态范围控制；
- 三个 tile 分别统计、分别归一化，不能跨 bank 混算。

### 3.4 最小器件清单

| 模块 | 原型建议 | 作用 |
|---|---|---|
| 光源 | 532 nm 相干激光、扩束和偏振器 | 匹配当前模拟波长 |
| 输入/重载 | amplitude SLM 或 DMD | 加载三个非负振幅 tile |
| 学习相位 | phase-only SLM | 逐层切换 `φ_1…φ_8` |
| 传播核心 | 两片透镜的 4f relay | 实现 FT/filter/IFT |
| 频域相位 | 第二块 phase SLM 或固定 DOE | 加载 `H_y(f_y)` |
| 探测 | 线性响应 sCMOS | 获取每层强度 |
| 控制 | PC/FPGA + ADC/DAC | 固定路由、归一化、电子残差、gate、重载 |
| 可选校准旁路 | 小功率参考臂/离轴全息 | 只在标定时测复场和 `H_actual` |

正常推理仍然只需要强度探测；参考臂用于算子标定，不必成为每次推理的一部分。

## 4. 相关工作给出的创新边界

### 4.1 算法结构先例

- [MLP-Mixer](https://research.google/pubs/mlp-mixer-an-all-mlp-architecture-for-vision/) 已经定义交替 token mixing 与 channel mixing。P11 只能说受其启发，不能说发明了这种分解。
- [FNet](https://arxiv.org/abs/2105.03824) 和 [Adaptive Frequency Filters](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Adaptive_Frequency_Filters_As_Efficient_Global_Token_Mixers_ICCV_2023_paper.html) 已表明傅里叶变换/频域滤波可以充当电子 token mixer。
- [MetaFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.html) 说明 token mixer 不必是 attention；因此“用非 attention 算子构建 backbone”本身不是创新。

### 4.2 衍射、1D/轴向和光学张量处理先例

- [D²NN](https://doi.org/10.1126/science.aat8084) 奠定了学习相位面和衍射传播执行光学推理的基本范式。
- [任意线性变换的多层衍射实现](https://www.nature.com/articles/s41377-021-00623-5) 表明足够多的衍射层可逼近一般复线性映射，因此“衍射可以混合 token/feature”本身不新。
- [片上 IDNN](https://www.nature.com/articles/s41467-022-28702-0) 已通过 ODFT/OIDFT 实现一维序列和卷积计算，不能声称首个一维衍射/FFT 网络。
- 1989 年的[一维成像与正交傅里叶变换](https://opg.optica.org/abstract.cfm?uri=ao-28-22-4731) 已展示一个方向成像、正交方向进行一维 FT 的光学原语。
- [Direct tensor processing with coherent light](https://www.nature.com/articles/s41566-025-01799-7) 更直接地使用柱面镜组实现 row-wise imaging/column-wise FT 和随后正交变换，并部署 CNN/ViT tensor operations。这是“正交轴光学处理”不能作为首创点的最强先例。

### 4.3 光学 Transformer 与空间—通道混合先例

- [Optical Transformers](https://openreview.net/forum?id=Xxw0edFFQC) 和 [Lightening-Transformer](https://arxiv.org/abs/2305.19533) 已研究用光学/光子矩阵乘执行 Transformer 运算，P11 不是首个 optical Transformer。
- [Monet 多通道光学视觉架构](https://www.nature.com/articles/s41377-022-00945-y) 已把 inter-channel 干涉与 intra-channel 衍射组合起来，宽泛的“光学通道/空间协同”已有先例。
- [MDR-HDONN](https://arxiv.org/abs/2411.05748) 是很接近的竞争工作：自由空间衍射负责 depthwise spatial processing，集成 photonic tensor core 负责前后 channel mixing。P11 的可区分点是 feature mixing 也尝试由正交轴相位衍射完成，而不是另设通用光子 MVM。
- 2026 年的 [Multi-channel Optical Vision Model](https://arxiv.org/abs/2606.10253) 已包含百万级相位参数、光学 visual tokens 和每层 channel mixing；其 channel mixing 仍是电子加权和。P11 不能声称首个百万参数光学视觉模型或首个 optical-token front end，但可强调尝试把 feature mixing 映射到轴向衍射。

### 4.4 本次检索的严格结论

截至 2026-08-24，在以上原始论文、出版社页面和作者官方页面中，**没有检索到 P11 精确组合的公开结构**：

```text
pretrained vision stem
  → token-by-hidden-feature optical field
  → [token-coordinate phase diffraction
     → feature-coordinate phase diffraction] × L
```

但“没有检索到”不等于世界首个。本次没有完成专利、Web of Science/Scopus 全库、非英文数据库和完整引文网络的新颖性检索。正式论文只能使用带完整限定语的 `to our knowledge` 或 `among the first investigations`。

## 5. P11 能否算论文创新点

### 5.1 单独以当前 P11 投稿：创新性中等、证据偏弱

当前 P11 的候选贡献是：

1. 将预训练视觉 token 表显式映射为 `token × feature` 的物理二维光场；
2. 用交替的轴选择性相位衍射分别承担 token 和 feature 主支路 mixing；
3. 在每层光学融合系数受下界约束、电子残差预算受控的条件下做大规模 ImageNet backbone 训练。

但是现在只有理想仿真、单 seed、且 P11 因显卡让用被停止在 15/90 epochs。匹配 epoch 15 时：

| 模型 | ImageNet-1K Top-1 | Top-5 |
|---|---:|---:|
| P09 普通二维各向同性光学 | 36.430% | 61.064% |
| P11 token/channel 轴向光学 | 37.392% | 62.168% |
| P11 - P09 | +0.962 pp | +1.104 pp |

这只能作为“值得继续”的早期信号，不能证明结构创新有效：训练未完成、只有一个随机种子，而且电子 gate 与优化轨迹也会共同影响结果。

### 5.2 加入真实光路和数字孪生：可以成为明确的架构/系统创新

更可守住的贡献表述是：

> We investigate a semantics-aligned axial diffractive vision backbone that maps tokens and hidden features onto orthogonal optical coordinates and alternates phase-only token-coordinate and feature-coordinate mixing through calibrated OEO stages.

建议用 `investigate`、`among the first`，而不是无条件 `the first`。

真正让它成为论文创新的证据应包括：

- 一个实测 token stage 和一个实测 channel stage；
- 证明活动轴发生预期混合、正交轴泄漏可测且可控；
- nominal simulator → measured hardware → calibrated digital twin → hardware-aware tuning 的完整恢复曲线；
- 同参数/同电子预算的四组架构比较；
- ImageNet 之外至少一个下游感知任务。

### 5.3 与 FA 主线合并：最强的完整论文故事

本项目更适合的中心思想不是“又做了一个光学分类器”，而是：

> **Optics-native separable token–channel backbone with reusable pretrained optical feedback for efficient downstream adaptation under hardware drift.**

对应中文：

> 构建面向光学物理的可分离 token–channel 通用视觉骨干，并研究在部署漂移下复用预训练固定光学反馈算子进行低成本下游适配。

这条主线把三个问题连起来：

1. forward：怎样设计一个光学占主导、能在 ImageNet 学到通用表征的算子；
2. backward：下游微调时，是否可以复用预训练终点的固定反馈关系；
3. deployment：光路错位后，固定反馈的有效工作区、失效边界和校准成本是什么。

P11 因此不是孤立创新，而是把原来的 FA 结论扩展到更大规模、更具物理结构的 source operator。

## 6. 建议只做的四组核心架构实验

遵守“主比较只保留四组”的要求，建议全部使用相同训练 recipe、相同 seed、相同八层/三 bank/相位参数量、相同 width-96 电子旁路和相同门控下界：

| 组 | 光学算子 | 回答的问题 |
|---|---|---|
| 1. P09 isotropic-2D | 八层普通二维 angular spectrum | 现有受控基线 |
| 2. P11 T→C alternating | `[token-axis → channel-axis] × 4` | 语义轴分解是否优于普通二维传播 |
| 3. Token-only | 八层都沿 token 轴 | 提升是否只来自 token mixing |
| 4. Channel-only | 八层都沿 feature 轴 | 两个轴是否都必要 |

主表最终应完成 90 epochs，并至少做 3 seeds。`C→T` 顺序可作为短程附录，不进入主表。

对每个已训练 checkpoint 还可以做不新增训练的四种推理诊断：normal、optical-off、learned-phase→random-phase、electronic-transform-off。这些是机制诊断，不是额外的方法组。

## 7. 真实光路必须补的实验

### 7.1 算子级

分别对 `H_y` 和转置后的 channel 算子输入针孔、单线、正弦条纹和随机场，测量：

- 实测传递函数 `H_actual`、PSF/MTF；
- 活动轴输出 NRMSE/cosine；
- 正交轴泄漏；
- 三个 bank tile 串扰；
- 多次重复的漂移和方差。

建议把正交轴泄漏定义为：

```text
ε_axis = ||T_actual - (T_y ⊗ I_x)||_F / ||T_actual||_F
```

第一阶段可把“单层归一化后 feature cosine > 0.98、泄漏低于 -20 dB”作为工程目标，但这只是建议的 go/no-go 线，不是现有实验结论。

### 7.2 单 stage 与 macroblock

用数百个真实 stem 输出依次验证：

1. amplitude loader；
2. learned phase；
3. token stage；
4. channel stage；
5. `T→C` 一个完整 macroblock；
6. 重复 2/4/8 次后的误差累积。

每一级同时报告 raw intensity NRMSE、归一化后 cosine/Pearson、SNR、光效率和十次重复标准差。只报最终分类准确率无法定位失配来源。

### 7.3 四阶段 sim-to-hardware 恢复曲线

主硬件表建议固定为四列：

| 阶段 | 含义 |
|---|---|
| ideal simulator | 当前理想模型 |
| uncalibrated hardware | 直接部署到名义光路 |
| calibrated digital twin | 用实测几何、LUT、`H_actual` 替换名义模型 |
| hardware-aware tuned | 基于数字孪生/闭环做少量适配 |

这条恢复曲线本身比只展示一个硬件准确率更有学术价值，因为它能量化“物理可实现”和“可校准”的区别。

### 7.4 非理想因素

至少覆盖：

- 亚像素平移、旋转、倍率误差和固定翻转；
- 柱面轴角误差及 cross-axis leakage；
- z/f 偏差、离焦、有限 NA、光阑和像差；
- SLM phase LUT、量化、填充因子、像素串扰、零级和面板曲率；
- camera shot/read/dark/PRNU、ADC 位宽、饱和和曝光漂移；
- 激光功率、相位、偏振、散斑和照明不均；
- 八次 OEO 中的误差累积。

当前仿真 FFT 隐含周期边界，真实有限孔径更接近零边界；这一点也应进入数字孪生。

## 8. 审稿人最可能追问的问题

### 8.1 “这不就是 MLP-Mixer 的光学实现吗？”

不是严格等价。标准 MLP-Mixer 使用共享的两层 MLP 和非线性；P11 是 line-specific phase modulation、固定受限传播和平方律读出。论文应说 `MLP-Mixer-inspired`，并比较表达能力、参数共享和物理约束。

### 8.2 “一维/正交轴傅里叶光学早就有了，新在哪里？”

承认物理原语不新。创新落在：预训练 token 表的物理坐标映射、两个语义轴的结构化交替、matched-budget backbone 效果、真实 OEO 映射及其与固定反馈微调的结合。

### 8.3 “feature 维为什么具有物理邻接关系？”

224 个 feature 本身没有天然空间邻接。当前可辩护点是 1024→224 adapter 可训练，它能够共同学习适合一维物理传播的 feature basis；但必须用 channel-only、phase 破坏和迁移实验验证，不能只凭直觉。

### 8.4 “token stage 为什么把 14×14 拉成一维？”

它是全局一维 token mixer，不是二维局部视觉卷积。下一版可以研究真正的 axial-token 结构：在 14×14 网格上分别做 row-token 和 column-token 光学 mixing，再处理 feature 轴。该方案更保持空间拓扑，但会改变当前受控 P11，不应在 P11 尚未完成时混入主比较。

### 8.5 “光学占比超过 50% 是否意味着更快、更省电？”

不能。当前 55.51% 是可训练 backbone 参数的数量比例；八次相机、ADC、电子归一化、Slim Mixer、gate、DAC/SLM 重载可能主导真实能耗和延迟。只有测量系统 wall time、吞吐、器件功耗和转换开销后才能讨论硬件优势。

## 9. 推荐推进顺序

### 阶段 1：先把数值结论做完整

1. 找到稳定双卡资源，完成 P11 剩余 90-epoch 训练；
2. 完成 P09、P11 的 matched 90-epoch 主比较；
3. 只再训练 token-only 和 channel-only，形成唯一四组；
4. 做四个零训练推理诊断，确认性能确实依赖学习相位和光学支路；
5. 根据预先约定的 ImageNet validation 和光学依赖指标选定 source。

### 阶段 2：搭一套可复用单轴核心

1. 先不接 Qwen 和分类头，只验证 `H_y`；
2. 加 learned phase，测单 stage；
3. 用固定转置复现 channel stage；
4. 连接一个 `T→C` macroblock；
5. 建立 measured digital twin，再尝试八次循环。

### 阶段 3：把 backbone 和 FA 主线接起来

1. ImageNet source 冻结后选择分类之外的一个简单下游任务；
2. 在同一 P11 source 上只做 NoFT、BP-current、FA-pretrained、FA-random 四组；
3. 比较 ideal、stale nominal operator、measured current operator 和 calibration 后的结果；
4. 报告 accuracy、逐层 gradient alignment、phase drift、校准次数、传输状态量和 wall time。

### 阶段 4：决定论文定位

- 若 P11 只在理想仿真中小幅领先：把它作为 FA 主论文中的 backbone 设计，不单独宣称硬件架构突破；
- 若实测 macroblock 与数字模型高度一致、P11 matched-budget 稳定领先且能迁移：可把 OTCM 作为第一创新点，FA 作为第二创新点；
- 若硬件误差较大但标定/固定反馈能低成本恢复：把“可校准的轴向光学 backbone + 固定反馈适配”作为更有特色的系统论文主线。

## 10. 最终建议

1. **值得继续做 P11**。epoch 15 的受控早期领先说明它至少比随机设想更值得完成，但证据尚不足以定论。
2. **先搭可复用单轴 4f 核心，不要一开始铺八层**。利用 OEO 边界固定转置复用同一硬件，是当前最合理的工程方案。
3. **创新必须写窄、写准**。主张“token-by-feature 语义轴映射与交替轴向衍射”，不要主张首个光学 token mixer、首个光学 Transformer、首个一维光学网络。
4. **P11 单独不是最强故事**。最有潜力的论文主线是 OTCM backbone、真实硬件漂移和预训练固定反馈三者结合。
5. **当前最大的科学风险不是参数量，而是物理对应是否成立**：正交轴恒等 relay、feature 物理邻接、28 个 padding mode、八次 OEO 累积误差都必须用实测或明确消融回答。
6. **下一步不应继续扩散新架构**。先用四组把“为什么 token/channel 都需要”回答清楚，再进入硬件和下游任务。

## 11. 主要参考文献

1. Tolstikhin et al., [MLP-Mixer: An All-MLP Architecture for Vision](https://research.google/pubs/mlp-mixer-an-all-mlp-architecture-for-vision/), NeurIPS 2021.
2. Lee-Thorp et al., [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824), 2021.
3. Lin et al., [All-optical machine learning using diffractive deep neural networks](https://doi.org/10.1126/science.aat8084), Science 2018.
4. Kulce et al., [All-optical synthesis of an arbitrary linear transformation using diffractive surfaces](https://www.nature.com/articles/s41377-021-00623-5), Light: Science & Applications 2021.
5. Zhu et al., [Space-efficient optical computing with an integrated chip diffractive neural network](https://www.nature.com/articles/s41467-022-28702-0), Nature Communications 2022.
6. Yan et al., [A multichannel optical computing architecture for advanced machine vision](https://www.nature.com/articles/s41377-022-00945-y), Light: Science & Applications 2022.
7. Anderson et al., [Optical Transformers](https://openreview.net/forum?id=Xxw0edFFQC), TMLR 2024.
8. Yin et al., [Multi-Dimensional Reconfigurable, Physically Composable Hybrid Diffractive Optical Neural Network](https://arxiv.org/abs/2411.05748), DAC 2025.
9. Wang et al., [Direct tensor processing with coherent light](https://www.nature.com/articles/s41566-025-01799-7), Nature Photonics 2025/2026.
10. Huang et al., [Adaptive Frequency Filters As Efficient Global Token Mixers](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Adaptive_Frequency_Filters_As_Efficient_Global_Token_Mixers_ICCV_2023_paper.html), ICCV 2023.
11. Yu et al., [MetaFormer Is Actually What You Need for Vision](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.html), CVPR 2022.
12. Li et al., [Multi-channel Optical Vision Model](https://arxiv.org/abs/2606.10253), 2026.

---

本报告将“当前代码事实”“建议硬件实现”和“未来增强方案”分开表述。P11 当前 ImageNet 预训练使用正常 exact BP；固定反馈仍是后续下游微调问题，不能把当前 P11 预训练结果提前写成 FA 结论。
