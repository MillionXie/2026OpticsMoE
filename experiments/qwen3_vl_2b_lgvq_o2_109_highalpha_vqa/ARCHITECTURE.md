# 架构与数据形状说明

## 1. 从视频到冻结 Qwen 特征

每段视频固定采样四帧：10%、37%、63%、90%。每帧取短边 65% 的中心区域并缩放为 `448×448`。

正式缓存使用 `Qwen3-VL-2B-Instruct`，而不是 Embedding 版：

```text
4 frames, each 448×448 RGB
  -> Qwen patch embedding + position embedding
  -> 完整冻结 Qwen Vision Transformer
  -> Qwen learned main merger
  -> [B, 4, 196, 2048]
```

这里的 `196=14×14` 是每帧经过主 merger 后保留的视觉 token 数，2048 是送入多模态语言空间的 hidden width。只使用最后的 main merger 输出；不读取三个 DeepStack tap，也不执行 DeepStack merger，因此没有重复的辅助视觉注入。

固定五档提示词走完整冻结的 Qwen Language 主干：

```text
prompt token IDs [B,L]
  -> token embedding + Qwen RoPE/decoder blocks
  -> [B,L,2048], L <= 64
```

Qwen 只在离线缓存时运行，正式学生训练读取缓存，因此 Qwen 参数被冻结且不会占用每个训练 step 的显存。缓存的正式合同必须是：

```text
qwen3vl_full_visual_main_merger_196x2048_v1
```

## 2. 进入 192 维光电学生网络

```text
Vision:   [B,4,196,2048] -> LayerNorm -> Linear 2048->192
Language: [B,L,2048]     -> LayerNorm -> Linear 2048->192
```

这两个投影是可训练电子模块。后续 Vision 使用 attention-free 的 `5×5 depthwise Conv2D + 1×1 Conv + channel MLP`；Language 使用因果 `kernel=5 depthwise Conv1D + pointwise + channel MLP`。它们提供每一层的电子残差 E，但不增加注意力层。

## 3. 光学 Router：固定 Top-2

Router 本身也由光传播实现，不存在电子 Router 备选路径。

Vision 的四帧同时放入 478 有效区域的四个 `232×232` lane。每个 lane 通过同一物理过程产生四个探测区域能量，标准化后 softmax，取 Top-2。Language 在单个 478 区域内完成同样的四专家打分。选中权重采用 power-L2 归一化：

```text
sum_i w_i^2 = 1, exactly two non-zero weights
```

训练使用 corrected straight-through estimator：前向严格稀疏 Top-2，反向仍能优化路由相位。Router 的输入偏移、相位偏移和 CCD 偏移均采用零填充的 ±8 px 扰动，不使用 `roll` 环绕；另有块状 phase dropout、能量捕获约束与标准化区域能量。

Router 测量复用同一套 SLM、10 cm 传播和 CCD，只增加时序上的路由曝光，不增加透镜、SLM、CCD、ROI 或传播距离。下面所称“四层”特指四个 feature replacement 层，Router 的测量不计为第五个 feature layer。

## 4. 518 / 478 / 232 / 109 的关系

```text
518 simulation canvas
└── 20 px margin + 478 active field + 20 px margin
    ├── 232×232 frame lane 0
    ├── 232×232 frame lane 1
    ├── 232×232 frame lane 2
    └── 232×232 frame lane 3
        lane 间水平/垂直间隔均为 14 px

each 232×232 lane
└── 2×2 experts, each 109×109
    starts: 0 and 123 on each axis
    expert gap: 123-109 = 14 px
```

Vision 一次并行处理四帧，所以仿真中共有 `4 lanes × 4 experts = 16` 个 109 相位 tile，但每个 lane 只激活两个专家。Language 是串行单 lane 四专家，同样只激活两个。

192 维 token 被变成光场时：Vision 先 `192->109`，再把 token 轴 `196->109`，形成 `109×109`；Language 将最多 `4+64=68` 个 token 按行写入 `109×109`，剩余区域补零。两者都在入光前做 RMS 归一化。

CCD 强度经过同一处理：非负截断、单 lane 均值归一化、相对强度截断到 8、`log1p` 压缩，再由 `AdaptiveAvgPool + LayerNorm + Softplus + Linear` 恢复为与电子分支相同的 token 数和 192 维宽度。

## 5. 四个光电融合层

严格顺序为：

```text
Qwen Vision main-merger tokens
  -> Vision expert optical layer + Vision Mixer 1
  -> balanced fusion 1
  -> Vision global optical layer + Vision Mixer 2
  -> balanced fusion 2
  -> 每帧 mean/max -> Linear，得到 4 个 image tokens
  -> 与固定 prompt tokens 拼接
  -> Language expert optical layer + Language Mixer 1
  -> balanced fusion 3
  -> Language global optical layer + Language Mixer 2
  -> balanced fusion 4
  -> dual-task readout
```

Expert 层使用 2×2 的 109 相位 tile；Global 层使用完整 478 有效区域的全局相位。四层都使用独立可学习 alpha，且都受所选配置的相同下界约束。

## 6. RMS 配平残差为什么这样写

令电子输出为 E、光学 CCD 读出为 O，有效 token 上的 RMS 分别为 `rE`、`rO`：

```text
M = (1-alpha) * E/rE + alpha * O/rO
F = rE * M / RMS(M)
```

`rE`、`rO` 和 `RMS(M)` 的统计量在反向传播中 detach。

这一步有三个作用：

1. E 与 O 先处于相同尺度，避免电子数值大而把光学淹没；
2. 显式使用 `(1-alpha)` 和 `alpha`，两侧贡献定义对称；
3. 最后的公共缩放把输出 RMS 恢复到电子残差原尺度，避免改变下游层熟悉的数值范围。

因此 alpha 表示“归一化特征混合系数”。它比直接比较原始 E/O 数值科学，但仍不应被误写成 CCD 物理能量百分比。

## 7. 双任务电子读出头

最后一部分允许加强电子读出，但不改变光路。

- Spatial：每帧对 196 token 求 mean/std/max，经小型 MLP；再对四帧求 mean/max，与 128 维语言摘要拼接，输出 Spatial MOS。
- Temporal：每帧 token mean/max 后，用并行 `depthwise Conv1D k=3/k=5` 建模四帧变化；同时统计序列 mean/max 与相邻帧差分 mean/max，再与语言摘要拼接，输出 Temporal MOS。
- 两个头最终各输出一个标量；没有 Alignment 头，没有 Transformer attention。

训练损失由 normalized Smooth-L1、pairwise ranking、batch Pearson correlation、光电特征 alignment、Router balance/importance/capture 组成。光电 alignment 只约束同层归一化特征接近，不是 LGVQ 的图文一致性标签。


