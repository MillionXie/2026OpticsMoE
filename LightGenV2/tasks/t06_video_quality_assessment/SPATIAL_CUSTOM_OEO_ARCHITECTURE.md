# Spatial-4 自写卷积光电模型结构

## 一句话说明

每条视频均匀抽取 4 帧。Qwen3-VL 只在离线准备数据时生成图像前端 token 和文本
embedding；正式学生训练、仿真评估和实验室推理均不执行 Qwen 的 Vision/Language
Transformer、Attention、merger 或 LM head。

学生网络只有两条主支路：电子残差支路与光学 Top-2 MoE 支路。两支路在每个光电层
按相同尺度融合，最后由一个电子读出头输出一个 Spatial MOS。

## 输入与 Qwen 边界

```text
一条视频
  -> 均匀抽取 4 帧，每帧 RGB 224x224
  -> 离线 Qwen 图像前端：patch_embed + position embedding
  -> 每帧 14x14=196 个 token，每个 token 1024 维
  -> [B,4,196,1024]
  -> 学生 Linear 1024->192
  -> 图像 token V：[B,4,196,192]

固定 Spatial prompt
  -> 离线 Qwen tokenizer + chat template + frozen embed_tokens
  -> [B,38,2048]
  -> 学生 Linear 2048->192
  -> 文本 token T：[B,38,192]
```

上述离线张量一次生成后保存为缓存。使用缓存进行学生训练、评估、导出 SLM 图案和
硬件微调时，不加载 2B 完整权重，也不执行任何 Qwen block。只有更换原始数据、抽帧
策略或 prompt 并重新生成缓存时，才单独运行 Qwen 前端缓存程序。

## 自写卷积电子模块放在哪里

自写模块不在 Qwen 里面，也不是最终分数旁路。它只位于第一层电子残差 E1 内：

```text
                         +-> 原有电子映射 E1_base -------+
Qwen 前端 token V ------+                                +-> E1
                         +-> 光场编码 -> 光学 O1 --------+   |
                                                               | 同尺度融合
原始 4 帧 RGB -> 自写卷积 -> 校正量 C -> 只加到 E1 -----------+   v
                                                              F1

F1 -> Vision global 光电层 -> Language expert 光电层
   -> Language global 光电层 -> 唯一 MOS 读出头 -> 一个分数
```

第一层实际电子量为：

```text
E1 = E1_base + C + Q
```

其中 `C` 是自写卷积校正，`Q` 是原有的任务内质量校正；二者都只能进入 E1，不能直接
输出 MOS，也不能绕过后续三层光电融合。因此仍然只有“电子残差”和“光学 MoE”两条
主预测支路。

## 自写卷积的精确结构

每帧独立通过以下项目内定义的基础算子：

```text
RGB [3,224,224]
 -> Conv3x3, stride2, 3->24  + GroupNorm + GELU   [24,112,112]
 -> Conv3x3, stride2, 24->48 + GroupNorm + GELU   [48,56,56]
 -> Conv3x3, stride2, 48->64 + GroupNorm + GELU   [64,28,28]
 -> Conv3x3, stride2, 64->96 + GroupNorm + GELU   [96,14,14]
 -> Conv3x3, 96->96 + GroupNorm + GELU
 -> Conv3x3, 96->96 + GroupNorm，与输入局部相加，再 GELU
 -> 展平为 196 个 token
 -> LayerNorm(96) -> Linear 96->192 -> GELU -> Linear 192->192
 -> 1.4*tanh 限幅
```

该模块共 316,568 个参数，全部由本项目直接定义。最后一个 Linear 零初始化，因此首次
接入已有模型时校正量为零，不会突然破坏原来的 E1；先做特征预训练，再尝试小学习率
联合微调。

## 四层光电主干

1. Vision expert：4 帧以 2x2 并行；每帧使用四个 109x109 相位专家，光 Router 选 Top-2。
2. Vision global：在同一 478x478 有效光场内继续传播与融合。
3. Language expert：图像摘要与 prompt token 进入语言侧光电层，光 Router 选 Top-2。
4. Language global：最后一次光电融合，随后进入唯一 MOS 读出头。

每层先独立计算电子量和光学量的 RMS，再按同尺度公式融合：

```text
F = rE * ((1-alpha)*E/rE + alpha*O/rO) / rms(mixture)
```

这避免电子数值范围较大而淹没光学量。四层 alpha 独立。光学消融只在同一个 checkpoint
上旁路光支路，不单独训练一个纯电子模型冒充贡献实验。

## 参数与性能边界

- 自写卷积 E1 校正：316,568 参数。
- 完整学生网络：12,849,995 参数。
- 其中现有的最终单指标读出头：10,031,046 参数。
- 正式推理中的 Qwen Vision blocks：0。
- 正式推理中的 Qwen Language blocks：0。
- Attention/Transformer 模块：0。

因此“31.7 万参数”只描述本轮替换的图像校正前端，不代表整个学生网络。若后续继续
压缩，应该把读出头蒸馏作为独立实验，不能把两种压缩结果混在一起报告。

## 训练结论

1. 先让自写卷积拟合已有 E1 特征，200 epoch 后 held-out feature PCC 为 0.8279。
2. 直接一次性解冻完整光电图会降低 SRCC，说明成熟光学主干容易被破坏。
3. 只联合自写卷积与 MOS 头、以及两档 mask-only 微调，都没有稳定超过预训练候选。
4. 因此正式保留最佳 checkpoint，而不是机械采用最后一次训练的权重。
5. 正式 run 只保留 best 与 last；不再每 5 epoch 保存权重。

## 独立交付边界

仓库中的 LightGenV2 profile 锁定后端源码 SHA，避免静默分叉。实验室 ZIP 会把运行所需
源码、配置、checkpoint、前端缓存、硬件控制、六张相位 BMP、微调入口和 SHA256 manifest
一并放入包内；解压后不依赖外部 LightGenV2 目录。完整 Qwen 权重不是学生运行依赖，
仅“从新原始视频重建 Qwen 前端缓存”这一可选步骤需要它。
