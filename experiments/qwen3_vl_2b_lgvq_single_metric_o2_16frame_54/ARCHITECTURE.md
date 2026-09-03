# LGVQ 单指标、文本条件、16 帧光电网络

## 1. 先明确模型在做什么

Spatial 与 Temporal **不是一个模型的两个输出**。它们是两次互相独立的实验：

- Spatial 模型只读取 Spatial prompt，只回归一个连续 Spatial MOS；
- Temporal 模型只读取 Temporal prompt，只回归一个连续 Temporal MOS；
- 两者有各自的训练参数、电子读出头、四层特征相位、两层 Router 相位和 checkpoint；
- 二者仅共享与任务无关的 16 帧 Qwen Vision 前端缓存。

采用连续 MOS，而不是把标签硬离散成五分类。`Excellent / Good / Fair / Poor / Bad`
仍作为 Qwen 文本提示的语义锚点；连续输出与师姐 LGVQ 评价脚本中的
SRCC/KRCC/PLCC/RMSE/MAE 口径一致。

## 2. 完整数据流与维度

```text
原始视频
  └─ 10%...90% 时间位置等间隔抽 16 帧
       └─ 每帧中心裁剪短边的 65%，缩放为 RGB 448×448
            │
            ├─ Qwen3-VL 官方 processor
            │    └─ 冻结 patch_embed + 官方二维位置嵌入
            │         [16, 784, 1024]
            │         └─ 按 Qwen 2×2 merge 顺序做均值：784→196
            │              └─ 再做二维 2×2 均值：14×14→7×7
            │                   Qwen tokens [16, 49, 1024]
            │
            └─ 固定质量测量旁路（无可训练参数）
                 [16, 49, 14]

Qwen tokens: LayerNorm + Linear 1024→192
Quality 14:  LayerNorm + Linear 14→192 + sigmoid 可学习门控
  └─ 相加并 LayerNorm，得到视觉输入 [B,16,49,192]

目标专属文本 prompt
  └─ Qwen chat template → tokenizer → 冻结 embed_tokens
       [B,L,2048] → LayerNorm + Linear 2048→192
       ├─ 无 Attention 的 FiLM 条件调制进入首层视觉输入
       └─ 后续与 16 个图像摘要 token 拼接

Vision Layer 1: 电子二维 Mixer 与 Optical Top-2 expert 并行 → RMS 同尺度凸融合
Vision Layer 2: 电子二维 Mixer 与 Optical global 并行      → RMS 同尺度凸融合
  └─ 每帧 49 token 的 mean/max 拼接后 Linear，得到 16 个图像 token

[16 个图像 token ; L 个 prompt token] → [B,S=16+L,192]，S≤96
  └─ 加可学习序列位置参数

Language Layer 1: 电子因果 DWConv1D 与 Optical Top-2 expert 并行 → 融合
Language Layer 2: 电子因果 DWConv1D 与 Optical global 并行      → 融合

目标专属无 Attention 读出头
  └─ 一个 normalized score → 训练集 mean/std 反归一化 → 一个连续 MOS
```

这里没有执行 Qwen 的 24 层 Vision Transformer、Vision merger、28 层 Language
Transformer 或 LM head，也没有任何 self-attention。Qwen 被保留的是输入最前端：

1. 官方图像预处理；
2. 冻结视觉 patch embedding；
3. 官方视觉位置 embedding；
4. chat template 与 tokenizer；
5. 冻结语言词 embedding。

正式训练前将这些结果缓存一次，训练和光路部署不反复加载完整 2B 模型。

## 3. 14 个质量通道到底是什么

它们不是 14 个类别，也不是 Qwen token。对每个 448×448 帧计算后池化到同一
个 7×7 网格：

| 通道 | 数量 | 含义 |
|---|---:|---|
| R/G/B | 3 | 原始颜色与亮度分布 |
| luminance | 1 | 灰度亮度 |
| Sobel x/y | 2 | 横、纵边缘响应 |
| gradient magnitude | 1 | 边缘强度 |
| absolute Laplacian | 1 | 高频/模糊敏感响应 |
| local std 5×5 | 1 | 局部对比度与纹理 |
| saturation | 1 | 颜色饱和度 |
| previous-frame luminance difference | 1 | 与上一抽样帧的亮度变化 |
| x/y/time coordinate | 3 | 网格位置与帧时间位置 |

合计 14。它们只通过有界的 sigmoid gate 补充 Qwen 前端。Qwen `1024→192`
始终是主路，质量通道不能覆盖或替换它。Spatial/Temporal 会分别学出自己的 gate。

## 4. 文本不是被删掉了

两个正式 prompt 分别是：

```text
Please evaluate the spatial quality of this video and rate it using one of the
following five levels: Excellent, Good, Fair, Poor, or Bad.
```

```text
Please evaluate the temporal quality of this video and rate it using one of the
following five levels: Excellent, Good, Fair, Poor, or Bad.
```

文本首先经 Qwen tokenizer 与冻结 `embed_tokens` 变成 `[B,L,2048]`，再投影到
`[B,L,192]`。它有两条作用路径：

- prompt 汇总生成 scale/shift，在第一层光学传播前调制视觉特征；
- prompt token 与 16 个图像 token 拼接，完整经过两层 Language 光电处理并进入读出头。

一个需要如实说明的统计事实是：同一个 Spatial 模型内 prompt 对所有视频相同，
所以它提供“当前要评 Spatial 还是 Temporal”的任务条件，而不是视频之间的额外
判别信息。真正的视频间排序信息来自 16 帧视觉特征。将两个任务分开训练可避免
共享读出头和共享相位之间的目标冲突。

## 5. 4×4 帧复用与 54×54 专家

物理合同保持：逻辑 canvas `518×518`、有效区 `478×478`、17 µm、传播 10 cm。

并行 Vision 平面：

- 16 帧按 4×4 排列；
- 每帧 lane `114×114`，pitch `120`；
- lane 在有效区内的坐标轴起点为 `2,122,242,362`；
- 每个 lane 内仍是 2×2 四专家，单专家 `54×54`，pitch `60`，间隔 6；
- Optical Router 为每帧产生四个区域能量，经 softmax 后只保留 Top-2；
- 16 帧的 64 张专家相位在一次相位图上并行加载。

由于 4×4 lane pitch 只有 120，10 cm 下若保留 1° 空间频率，理论最大横向传播约
103 个逻辑像素，容易串到相邻 lane。正式 16 帧配置把原有 k-space 限制收紧为
0.5°（约 51 像素），这是对同一光路有效孔径的仿真约束，不改变 518/478、像素
尺寸、传播距离或 2×2 专家拓扑。

后半段 Language 不再有 16 个独立 lane，保持原有串行 `109×109` 单输入与
2×2 四专家，pitch `123`。因此“专家减半”只发生在需要把 16 帧同时装入有效区
的 Vision 两层，不会无理由缩小 Language 两层。

## 6. 四层光电融合

每层电子特征 `E` 和光学 CCD 读出特征 `O` 先分别做样本内 RMS 尺度统一：

```text
E_n = E / rms(E)
O_n = O / rms(O)
M   = (1-alpha) E_n + alpha O_n
F   = rms(E) * M / rms(M)
```

四层各有一个独立 alpha，正式范围为 `[0.50,0.90]`，初值 0.57。这样电子明确
带 `(1-alpha)`，光也不会因为数值尺度较小而被淹没。`optical off` 只在同一个
已训练 checkpoint 上旁路四条光分支，不另训一套纯电子模型。

## 7. 两个读出头为何不同

Spatial 读出关注单帧空间统计：每帧对 49 token 取 mean/std/max，再跨 16 帧取
mean/std/max，并和最终 prompt 序列统计拼接，回归一个 Spatial MOS。

Temporal 读出先把每帧压成 mean/max，再用 depthwise Conv1D `k=3` 与 `k=5`
提取短时变化，同时显式统计一阶、二阶帧差，和 prompt 序列统计拼接，回归一个
Temporal MOS。两者都没有 attention 或 Transformer。

## 8. 公平对照与选模

- 每 5 epoch 在完整 558 个 test 视频上评估一次；
- 按当前单任务 SRCC 最高的 epoch 保存最佳 checkpoint；
- 不划 validation，明确记录 `test_used_for_selection=true`；
- 最终同时报告正常光电与同一 checkpoint 的四层去光结果；
- SRCC/KRCC/PLCC 越高越好，RMSE/MAE 越低越好；
- Spatial 与 Temporal 结果不能混成一个平均数掩盖弱项。
