# Warmstart5 完整数据流与网络结构

## 1. 先定义本文的形状符号

| 符号 | 含义 | 本工程典型值 |
|---|---|---:|
| `B` | batch size | 训练配置为 30 |
| `Nv` | 每张图的视觉 patch token 数 | `14×14=196` |
| `L` | prompt 中有效 Language token 数 | 每样本不同，必须 `≤224` |
| `Dv` | Qwen Vision hidden | 1024 |
| `Dl` | Qwen Language hidden | 2048 |
| `D` | 本工程光电 Mixer 宽度 | 192 |
| `R` | 光学 token-row 宽度 | 224 |
| `A` | CCD 有效区域 | `478×478` |
| `C` | 仿真传播 canvas | `518×518` |
| `De` | 最终检索 embedding | 64 |

下面以单张图为例，batch 维只需在最前面增加 `B`。

## 2. 输入与 Qwen Processor

### 2.1 数据输入

Caltech101 图像先由数据集代码读取为 RGB，并以训练增强或无增强方式得到 `224×224`
图像。训练增强包括随机裁剪、亮度/对比度扰动与旋转；正式 gallery/query 评估不使用
这些随机增强。

输入并不是只把图片交给视觉网络。Qwen processor 同时构造一段 chat：

```text
system: Represent this image for image-to-image object-category retrieval.
user:   <image>
```

因此 processor 输出至少包括：

- `pixel_values`：图像 patch 输入；
- `image_grid_thw`：图像 token 的二维网格信息；
- `input_ids`：instruction、控制符与 image placeholder；
- `attention_mask`：Language 有效 token 位置。

本工程禁止静默截断。视觉 token 数和 Language token 数分别超过 224 时直接报错。

## 3. 冻结 Qwen 的 Vision 前端

### 3.1 Patch embedding

Qwen3-VL-Embedding-2B 的视觉配置为：

```text
patch_size = 16
temporal_patch_size = 2
vision hidden = 1024
vision depth = 24
```

静态图像经 processor 的时间维包装后，Qwen 使用一个
`Conv3D(3→1024, kernel=stride=(2,16,16))` 做 patch embedding。空间上：

```text
224×224 RGB
→ 14×14 patches
→ [196,1024]
```

随后加上冻结的视觉位置 embedding。Qwen 也准备 RoPE，但由于本学生路径不执行
原生视觉 attention，RoPE 不会进入原生 attention 计算；二维位置关系主要由下游的
真实 14×14 2D Mixer 网格保留。

### 3.2 原 24 个 Vision Transformer block 去哪里了

学生模式会把 `visual.blocks` 全部替换：

- 第 0 个槽位触发完整的两级 Vision 光电模块；
- 中间槽位是 identity bypass；
- 最后一个槽位把缓存的 Vision 光电输出交给 Qwen 主 merger；
- 原 24 个 Vision attention/FFN block 均不执行。

所以“Qwen 冻结”表示保留预训练前端和接口，不表示完整冻结 Vision Transformer
仍逐层运行。

## 4. Vision 两级光电模块

输入为 `[B,196,1024]`。

### 4.1 降维到光电公共宽度

```text
[B,196,1024]
→ Linear(1024,192)
→ LayerNorm(192)
→ Xv = [B,196,192]
```

这是可训练电子 adapter。192 是工程设计的紧凑 latent 宽度，不是 pooling 得到的，
也不是只留下 192 个 token；token 数仍是 196。

### 4.2 Vision Block 1：2D 电子 Mixer 与 MoE4 expert 光路并行

电子分支：

```text
Xv [B,196,192]
→ 恢复为 [B,192,14,14] 的真实二维网格
→ LayerNorm
→ 3×3 depthwise Conv2D（groups=192，通道数不变）
→ GELU → pointwise Linear(192,192) → Dropout(0.1)
→ 可学习 sigmoid 残差相加
→ LayerNorm → MLP 192→384→192（GELU + Dropout）
→ 第二个可学习 sigmoid 残差相加
→ Ev1 [B,196,192]
```

光学分支：

```text
Xv [B,196,192]
→ Linear(192,224) → LayerNorm(224) → Softplus
→ 补齐 token 行，形成 [B,224,224] 非负振幅场
→ RMS 归一到 0.5
→ MoE4 router 选择 top-2 expert
→ 2×2 expert phase mask + 10 cm 衍射
→ CCD [B,478,478]
→ 统一 CCD readout
→ Ov1 [B,196,192]
```

融合：

```text
Fv1 = Ev1 + alpha_v1 × Ov1
alpha_v1 = 0.05 + 0.95 × sigmoid(g_v1)
Fv1 shape = [B,196,192]
```

两条分支从同一个 `Xv` 出发，只有在 `Fv1` 处合并；光路内部不调用电子 Mixer。

### 4.3 Vision Block 2：Global 光路

`Fv1` 同时进入第二个电子 Mixer，以及重新编码的 global 光路：

```text
电子：Fv1 → 第二个同结构 2D Mixer → Ev2 [B,196,192]

光学：Fv1 → Linear(192,224)+LN+Softplus → [B,224,224]
     → 按同一 router 权重重新铺到 2×2 有效区
     → 单张 478×478 global phase mask + 10 cm 衍射
     → CCD readout → Ov2 [B,196,192]

Zv = LayerNorm(Ev2 + alpha_v2 × Ov2)  # [B,196,192]
```

注意：Block 1 的 CCD 不会作为复振幅直接传到 Block 2。Block 1 的光学结果先被电子
readout、与电子分支融合，然后 `Fv1` 再被编码为下一次振幅 SLM 输入。这正是一次
“光—电—光”闭环。

### 4.4 恢复 Qwen Vision 接口

为了让冻结的 Qwen 主 merger 接受数据：

```text
Zv [B,196,192]
→ Linear(192,1024)
→ 乘可学习外层 residual gate
→ 加回进入本模块前的 [B,196,1024]
→ [B,196,1024]
```

这一步只恢复 hidden 维度，不改变 token 数。

## 5. 冻结 Qwen 主 merger：Vision 与 Language 的真正连接点

本工程关闭 3 路辅助 DeepStack，但保留主 merger。Qwen 主 merger 做：

```text
每个 1024-D token 做 LayerNorm
→ 每个相邻 2×2 token 拼接为 4096-D
→ Linear(4096,4096) → GELU → Linear(4096,2048)
```

因此：

```text
[196,1024] = [14,14,1024]
→ [7,7,2048]
→ [49,2048]
```

这 49 个 image token 替换 prompt 中对应的 image placeholder embedding。文本 token
由冻结的 `Embedding(151936,2048)` 查表得到。最终得到统一的多模态序列：

```text
H0 = [B,L,2048], L≤224
```

`H0` 中既有 instruction/control token，也有 49 个主 merger image token。这里不是
把一个“Vision 192 维向量”与一个“Language 192 维向量”拼接；Vision 始终保持
token 序列，经过 merger 后成为 Language 序列的一部分。

## 6. Language 两级光电模块

原 Qwen 有 28 个 Language decoder layer。学生模式中：

- layer 0 放 Language Block 1；
- layer 1 放 Language Block 2；
- layer 2–27 全部 bypass；
- 原生 Language self-attention 和 FFN 均不执行；
- native pre-attention 关闭；
- 最后 Qwen RMSNorm 为完成原接口仍会执行，但检索 head 不读取其输出。

### 6.1 降维

```text
H0 [B,L,2048]
→ Linear(2048,192)
→ LayerNorm(192)
→ Xl [B,L,192]
```

### 6.2 Language Block 1

电子分支使用 causal 1D Mixer：

```text
LayerNorm
→ 左侧补 4 个位置
→ kernel=5 depthwise Conv1D（groups=192，输出长度仍为 L）
→ GELU → Linear(192,192) → Dropout
→ gated residual
→ LayerNorm → MLP 192→384→192
→ gated residual
→ El1 [B,L,192]
```

只有左侧 padding，因此第 t 个 token 不读取未来 token。光学 expert 分支与 Vision
Block 1 使用同样的 `192→224→224×224→MoE4→CCD→192` 流程，但拥有独立的
router、phase、CCD readout 和 adapter：

```text
Fl1 = El1 + alpha_l1 × Ol1  # [B,L,192]
```

为了占用 Qwen layer 0 的接口，代码还可把 `Fl1` 投影回 2048 并做外层残差；但真正
送入我们 Language Block 2 的是缓存的 `Fl1`，不是再次压缩那个 2048 维接口张量。

### 6.3 Language Block 2

```text
电子：Fl1 → 第二个 causal 1D Mixer → El2 [B,L,192]
光学：Fl1 → 重编码到振幅场 → global phase → CCD → Ol2 [B,L,192]
最终：Zl = LayerNorm(El2 + alpha_l2 × Ol2)  # [B,L,192]
```

Language Block 2 也会生成一个兼容 Qwen 的 2048 维返回值，但本检索任务明确读取
缓存的 `Zl`。因此 Language 的 `Linear(192,2048)` 对最终 retrieval 没有梯度，代码
将它冻结，避免把“存在于 state dict”误报成“真正参与学习”。

## 7. CCD 到 192 维 optical delta 的完整变化

Vision/Language、expert/global 四个阶段使用相同结构、不同参数的 CCD readout：

```text
raw CCD intensity                       [B,478,478]
→ clamp_min(0)                          [B,478,478]
→ 每帧除以该帧全局均值                 [B,478,478]
→ 相对强度最大截断到 12                 [B,478,478]
→ log1p(relative intensity)             [B,478,478]
→ AdaptiveAvgPool2D(224,224)            [B,224,224]
→ 每一行做非仿射 LayerNorm(224)         [B,224,224]
→ ReLU                                  [B,224,224]
→ 只取前 Nv 或 L 行                     [sum(tokens),224]
→ Linear(224,192)                       [sum(tokens),192]
→ 按样本和 padding 位置散回             [B,Nv/L,192]
```

没有背景帧，也没有背景扣除；没有对真实 CCD 做额外平方；仿真和实测均进入同一套
frame-mean、clip、log1p 和 readout。

## 8. MoE4 expert 光路内部发生了什么

### 8.1 2×2 布局

每个 expert 的相位区域为 `224×224`，pitch 为 254，相邻 expert 之间有 30 像素
间隙。四个 expert 正好覆盖 `478×478` 有效区域；外面每边再留 20 像素保护带，
构成 `518×518` FFT 传播 canvas。

### 8.2 Router

对 `224×224` 输入振幅场：

```text
AdaptiveAvgPool 224×224→14×14
→ flatten 196
→ 无仿射 LayerNorm
→ Linear(196,4)
→ softmax(temperature=2)
→ top-2 稀疏化并重新归一化
```

选中的两个 expert 各收到同一输入场的加权副本，未选 expert 的振幅为 0。Vision 与
Language 各有一个独立 router；各自的 expert/global 两级复用本模态的 routing。

### 8.3 相位与传播

每个相位参数采用：

```text
phi = 2π × sigmoid(raw_phase)
complex modulation = exp(i × phi)
```

expert 级有 4 张独立 `224×224` 相位；global 级有 1 张 `478×478` 相位。Vision 与
Language 各一套，因此共有 10 张逻辑相位阵列，但导出为 4 个物理阶段 BMP：两个
expert BMP 各自把 4 个 expert 拼成 2×2，另有两个 global BMP。

每次相位调制后用 angular-spectrum 方法传播 0.10 m，波长 532 nm，逻辑采样 17 μm，
并施加 `theta_max=0.65°` 的 k-space 带宽限制。CCD 计算复场模平方得到强度。

## 9. 训练时的鲁棒性扰动

这些扰动只在 training mode 生效，普通 clean 仿真评估不随机扰动：

| 扰动 | warmstart5 配置 |
|---|---:|
| 输入振幅位置 | 每个物理 stage 独立，x/y 各 `[-16,16]` logical px |
| phase mask 位置 | 每级独立 `[-16,16]` px |
| CCD/ROI 位置 | 每级独立 `[-16,16]` px |
| CCD gain | 每样本 `Uniform(0.4,2.5)` |
| 正偏置 | `Uniform(0,0.05×frame_mean)` |
| read noise | `Normal(0,0.015×frame_mean)` |
| phase dropout | 8×8 block，概率 0.08，旁路为零相位 |
| router logit noise | 训练时标准差 0.10 |

本 warmstart5 配置没有启用后来增加的 coherent zero-order leakage，也没有启用后来
的 truncated biased Gaussian 版本；不能把其他实验的噪声设置倒推到这一正式模型。

## 10. 从 Language token 到最终 64 维结果

只对每个样本的有效 `Zl [L,192]` 做：

```text
mean over L tokens → [192]
max  over L tokens → [192]
concatenate        → [384]
LayerNorm(384)
Linear(384,64)
L2 normalize       → [64]
```

64 维向量没有 softmax，也不是 10 类 logits。

评估时每类 3 张 gallery 图各自生成 64 维单位向量。同类向量求均值后再次 L2
归一化，形成 10 个 class prototype。每个 query 与 10 个 prototype 点积，得到 cosine
similarity，按相似度排序计算 Top-1、Top-3 与 MRR。

## 11. 训练、冻结与 loss

### 11.1 双源 warm start

Stage A 初始化时严格合并两个 checkpoint：

- 电子 checkpoint：加载 Vision/Language adapter、2D/1D Mixer 和最终 64D readout；
- robust 光学 checkpoint：只加载 `optical_branch`，包括 phase、router、CCD readout
  与光后 adapter；
- 四个融合 gate 不从任一旧 checkpoint 加载，统一重置为 0.055。

### 11.2 两阶段训练

| 阶段 | epoch | 可训练内容 | 冻结内容 |
|---|---:|---|---|
| Stage A optical calibration | 4 | 两模态 optical branch，共约 1,120,112 参数 | Mixer、外层 adapter、64D head、四个 gate、Qwen |
| Stage B joint | 12 | Mixer、optical branch、gate、有效 adapter、64D head，共 2,683,709 参数 | 原 Qwen；Language 无效输出 adapter 与无效 residual gate |

两阶段每个 epoch 固定 12 个 PK batch；每个 batch 为 10 类×3 图=30 张，所以它是
固定步数训练，不代表每个 epoch 完整遍历 2,625 张训练图。

### 11.3 总 loss

两阶段均无 teacher/KD：

```text
total = 1.0 × supervised contrastive loss
      + 1.0 × episodic prototype retrieval CE
      + 0.02 × CCD operating-point loss
      + 0.05 × router balance loss
      + 0.005 × router importance loss
```

CCD operating-point loss用 clean raw CCD 的帧均值与目标均值 0.25 的对数域 Smooth-L1，
目的是避免光学支路整体过暗或过亮；它不要求 CCD 图像与某张教师 CCD 做像素 MSE。

## 12. 硬件尺寸与坐标

- 仿真：`518×518` canvas，中央 `478×478` 有效区，17 μm/logical pixel；
- 振幅 SLM：`1024×1024`、17 μm，逻辑有效图按 1:1 像素播放并居中；
- 相位 SLM：`1920×1200`、8 μm，中心可设为 `(980,590)`；
- 相位导出保留 vertical flip；
- CCD 正式进入模型前必须变换为模型坐标下的 `478×478` 非负强度图。

17 μm 训练相位导出到 8 μm phase SLM 时是物理坐标重建问题，不改变网络内部的
`224/478/518` 逻辑张量定义。

## 13. 冻结 Qwen 参数为何是 340,274,176

学生路径实际调用的冻结模块为：

| 冻结模块 | 参数量 | 作用 |
|---|---:|---|
| Vision Conv3D patch embedding | 1,573,888 | RGB patch→1024 |
| Vision position embedding | 2,359,296 | 给 196 patch 加位置 |
| Qwen main merger | 25,174,016 | `4×1024→2048`，196 token→49 token |
| Language token embedding | 311,164,928 | 151,936 词表→2048，实际为查表 |
| Language final RMSNorm | 2,048 | 完成 Qwen 接口；不供 retrieval head 使用 |
| **合计** | **340,274,176** | 全部冻结 |

这解释了为什么参数统计会出现 3.40 亿：主要是巨大词表 embedding，而不是模型偷偷
执行了大量 Qwen attention。原 24 个视觉 block 和 28 个语言 block 均被替换/旁路。
