# LGVQ Spatial：连续相位、无 K 空间裁剪版本

这轮的正式仿真候选是 `s437`。训练和测试前向均使用连续浮点相位，没有 256 级 straight-through 量化，也没有 K 空间裁剪、像素位移扰动或 phase dropout。最佳结果为 SRCC **0.62415**、PLCC **0.66191**；同一 checkpoint 旁路四个光学层后 SRCC 为 0.51812，光学部分带来 `+0.10603 SRCC`。

## 必须区分的两个结论

1. 新建 feature mask 时可以严格使用 `raw_phase=0`。相位约束为 `phase=2π·sigmoid(raw_phase)`，所以它对应均匀的物理相位 π，而不是物理相位 0。Vision/Language Router 的相位是为四个能量区生成的解析聚焦相位，不能随 feature mask 一起清零。
2. 当前最高性能 `s437` 是从已经学好的连续相位 checkpoint 精修，不是这一次从零重新训练。严格从 `raw_phase=0` 冷启动的 `s432` 达到 SRCC 0.61113。四个 feature mask 的 wrapped RMS 更新为 0.0530、0.0318、0.0244、0.0100 rad，证明相位确实被优化器训动，而不是只有电子头在动。

## 为什么最佳候选仍保留 20% 未调制分量

这里按实际结果作决定，而不是先验假设。对同一个连续相位 checkpoint 做无 K 空间的固定 DC 扫描，SRCC 为：

| 未调制功率 | 0% | 5% | 10% | 15% | 20% | 25% | 30% | 35% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SRCC | 0.61586 | 0.62089 | 0.62179 | 0.62239 | **0.62333** | 0.62261 | 0.62225 | 0.62026 |

20% 并没有拉低仿真性能，反而比 0% 高约 0.0075。原因是它与已调制复振幅相干叠加，提供了可学习的参考场/干涉项，而不只是加性背景。因此：

- `s437` 保留固定 20% DC，作为当前性能候选；
- `s431` 固定 0% DC，作为纯净光学对照，SRCC 0.61822；
- 两者不能混写为同一个设置。

## 输入张量到底是什么

冻结 Qwen 前端对每个视频均匀取 4 帧。每帧经过 Qwen Vision 的 patch embedding、二维位置编码和前端处理后，缓存为 196 个视觉 token：

```text
[B, 4, 196, 1024]
          196 = 14 × 14
LayerNorm + Linear(1024 → 192)
        ↓
[B, 4, 196, 192]
```

所以“每帧 `[14,14,192]`”不是原始 CCD 图，也不是又切出 14 帧；它只是把同一帧的 196 个 token 按 Qwen patch 的二维顺序还原成 14×14 网格，每个格子有 192 个通道。4 是一个视频实际抽取的四帧。

此外还有固定的 14 通道低层质量图（亮度、颜色、局部均值/方差、梯度、坐标等），经已经训练好的五层普通卷积编码成同样的 `[B,4,196,192]` 缓存。正式两支路模型没有让它直接输出 MOS，也没有在末端另开第三个预测分支；它只在第一层的唯一电子残差 `E1` 内作一次加性修正。主视觉输入仍是 Qwen token。若论文要求“电子支路也只能接 Qwen token”，应另做关闭该缓存的消融，不能把当前结构描述成完全没有低层质量输入。

Spatial prompt 经过 Qwen tokenizer 和冻结的 `embed_tokens` 得到约 `[B,38,2048]`，再用 `Linear(2048→192)` 进入学生网络。prompt 的汇总向量生成 192 维 scale/shift，只对上述 Qwen 视觉 token 做轻量条件调制；它不是单独的 MOS 预测分支。

## 四层光电主干

```text
Qwen视觉token + E1内的低层质量修正 + Spatial prompt条件调制
                │
                ├─ Vision optical router → Top-2/4 experts
                ├─ Vision E1 与 O1 同尺度 RMS 归一化后凸融合
                ├─ Vision E2 与 O2 global 同尺度融合
                │
        四帧各压成一个图像摘要 token
                │ + prompt tokens
                ├─ Language optical router → Top-2/4 experts
                ├─ Language E3 与 O3 同尺度融合
                └─ Language E4 与 O4 global 同尺度融合
                              ↓
                      一个 Spatial MOS
```

四个融合权重 α 分别约为 0.4852、0.4927、0.5138、0.5195。融合前对 E/O 分别做样本级 RMS 对齐，再计算 `(1-α)E + αO`，所以不能靠放大电子数值把光学支路淹没。

电子残差每层只有一条顺序路径。Vision 使用 `LayerNorm → 5×5 depthwise Conv2D → 1×1 Conv`，随后接一个 `LN → 5×5 depthwise → 1×1扩张(192→384) → GELU → 1×1投影(384→192)` 的残差块；Language 对应使用因果 depthwise Conv1D。没有 Attention、Transformer、VGG，也没有并列的多尺度预测分支。

## 读出头逐步解释

1. 最后一个 Vision 融合输出仍是 `[B,4,196,192]`，还原为 `[B×4,192,14,14]`。
2. `3×3 depthwise Conv` 只让每个通道观察相邻 patch，不做通道混合。
3. `1×1 projection 192→64` 是逐位置的线性通道压缩：保留 14×14 空间位置，把每格 192 个数字压为 64 个，避免后面的全连接层参数爆炸。这里的 projection 不是投影到 CCD，也不是光学传播。
4. 对每帧做 `adaptive average pooling 14×14→3×3` 与 `adaptive max pooling 14×14→3×3`。average 表示一块区域的整体质量，max 表示该区域最强的局部失真响应；二者互补。max 作用在已经卷积/GELU 的特征上，不是直接取 CCD 最亮像素。
5. 两个池化结果拼成 `64×3×3×2=1152`，经 MLP 得到每帧 512 维描述。
6. “四帧聚合”就是在 4 个 512 维帧描述上分别计算 mean/std/max，再拼成 1536 维。mean 表示稳定的总体质量，std 表示四帧之间波动，max 保留最明显的一帧证据。它没有生成额外帧，也没有把多个视频混在一起。
7. Language 最终序列（4 个图像摘要 token + prompt token）做 masked mean/std/max，得到 576 维，再投影为 512 维。
8. 1536 与 512 拼成 2048 维，经 `2048→1024→1` 输出一个连续 Spatial MOS。

读出头约 300.7 万参数，总学生网络约 550.9 万可训练参数。它是当前模型最大、也最容易过拟合的部分，因此这轮没有盲目继续加宽。

## 本轮优化为何停在当前结构

所有对照见 [`optimization_trials.csv`](optimization_trials.csv)。同起点下，加深电子残差、增加电子 identity skip、只训练相位、只训练读出头、增大 batch、checkpoint 插值都没有超过 `s437`。教师软目标权重已经设为 0；MOS 分层 batch、相关性/排序联合损失、EMA 和阶段回滚仍保留。继续添加并行电子模块虽然可能拟合训练集，但会违反“两支路且结构可解释”的约束，所以没有进入正式模型。

## 可视化与文件

- 相位总览：[`masks/phase_preview.png`](masks/phase_preview.png)
- 六个仿真 CCD 面示例：[`optical_visualization/sample_03_Hotshot-XL_A_water_skier_created_splashes_on_the_lake.mp4_six_ccd_planes.png`](optical_visualization/sample_03_Hotshot-XL_A_water_skier_created_splashes_on_the_lake.mp4_six_ccd_planes.png)
- 结构化指标：[`result.json`](result.json)
- 可直接加载的 1920×1200 BMP 位于 `masks/phase_slm_1920x1200/`。注意：BMP 文件格式必然只有 8 bit；“连续相位训练”指仿真前向与反向不做量化，不能据此宣称导出硬件后仍是无限精度。

当前 Vision Router 的 Top-2 选择占比为 22.65%、22.04%、32.53%、22.78%，没有坍缩。Language Router 为 0%、0%、50%、50%，仍集中于两个专家；这是当前结果的真实局限。
