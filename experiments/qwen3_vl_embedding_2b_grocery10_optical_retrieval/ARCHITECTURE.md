# MoE4 architecture

## Teacher

```text
RGB image + fixed retrieval instruction
→ frozen Qwen3-VL-Embedding-2B
→ official 64D Matryoshka embedding
→ L2 normalize
```

Teacher 始终为 `eval()`、`requires_grad_(False)`，embedding 在 Student 训练前缓存。

## Student

```text
RGB image + same instruction
→ frozen Qwen processor / patch embedding
→ Vision input adapter: hidden → 224
→ Vision Optical MoE4
→ Vision output adapter and frozen Qwen visual bridge
→ frozen token embedding + one multimodal injection
→ Language input adapter: 2048 → 224
→ Language Optical MoE4
→ Language output adapter: 224 → 2048
→ frozen final RMSNorm
→ retrieval LayerNorm + Linear(2048,64)
→ L2 normalize
```

最终部署比较 Student query 与 Student gallery 的 cosine similarity，不存在十分类 head。

## 每个 Optical MoE4 stack

```text
[B,224,224] amplitude
→ electronic Top-2 router
→ copy weighted amplitude into selected regions of a 2×2 expert bank
→ four independent 224×224 phase-only masks
→ 5 cm angular-spectrum propagation
→ square-law CCD
→ selected expert regions: independent LN → ReLU
→ reapply the original routing weights once
→ hard-zero all unselected expert regions
→ reload nonnegative amplitude
→ 478×478 global phase-only mask
→ 5 cm angular-spectrum propagation
→ square-law CCD
→ 2×2 detector integration and 224×224 readout
```

Router 只在 stack 入口计算一次，层间不会生成新的 routing weights。振幅入口和 phase mask
在仿真中视为共面；实验中的理想 4f relay 不额外模拟传播。

## 物理尺寸

| 项目 | 仿真 |
|---|---:|
| wavelength | 532 nm |
| pixel pitch | 16 μm |
| expert | 224×224 |
| expert pitch | 254 |
| expert gap | 30 |
| active area | 478×478 |
| FFT canvas | 518×518 |
| outer padding | 20 |
| propagation distance | 5 cm |

硬件 SLM 为 8 μm，因此每个仿真像素最近邻展开为物理 `2×2` pixels；active area 对应
956×956。CCD 读入后先注册至 956×956，再做 2×2 mean binning 回 478×478。

## 训练边界

训练：

- Vision/Language input、output adapters；
- 两个 electronic routers；
- 8 张 expert phase masks；
- 2 张 global phase masks；
- retrieval LayerNorm/Linear。

冻结：

- 完整 Teacher；
- Qwen tokenizer、processor、patch/token embeddings；
- visual bridge/merger；
-未替换 Qwen 参数和 final RMSNorm。

## 损失

```text
8.0 × pointwise embedding KD
+ 0.1 × relational KD
+ 1.0 × supervised contrastive retrieval
+ 0.25 × student-gallery loss
+ 0.10 × teacher-gallery anchor loss
+ 0.02 × router balance
+ 0.005 × router importance
```

Stage 2 使用相同目标，但降低各组学习率并增强裁剪、亮度、对比度和旋转扰动。

## 光路适配项

- 仿真相位参数化：`2π·sigmoid(raw_phase)`；raw zero 对应初始相位 π。
- K-space 限制：开启，0.65°。
- phase DC loss：关闭。
- amplitude BMP：正值 95th percentile 标定，gamma 0.65，提高暗输入可见性。
- phase BMP：导出前上下翻转。
- CCD：当前实验标定为上下、左右均翻转；其他光路必须重新确认。
- 不逐样本归一化最终 embedding 之前的 CCD 功率关系；只有最终 64D embedding 做 L2 normalize。

## 数据隔离

Gallery、train、test 不共享图像路径。Teacher cache identity 包含数据 manifest、SKU 顺序、
模型 ID、instruction、processor pixel budget 和 embedding 维度；不匹配时拒绝复用。
