# Qwen3-VL-2B BDD100K TimeOfDay-3 fullstack4 架构说明

本文对应实验：

`experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4`

本文中的尺寸来自当前服务器实际运行配置和第一批训练数据。模型为 `Qwen/Qwen3-VL-2B-Instruct`，模型计算使用 BF16，光学传播和蒸馏损失中的关键数值计算使用 FP32。

## 1. 当前日志属于什么阶段

以下是旧版永久截断配置产生的 **student_train 学生训练** 日志。它仍可用于识别训练阶段，但其中 `3375` 已不代表新版 sampler：

```text
epoch 1/30 batch 820/3375
loss_total=1.80754 loss_vision=0.16630 loss_answer=0.65510 loss_kd=1.22625 loss_ce=0.74604 running_top1=0.6948 lr=5.000e-04
```

判断依据是 student 训练同时计算四项损失：

- `loss_vision`：学生视觉 optical4 输出对齐电子 teacher 完整视觉栈输出。
- `loss_answer`：学生最终 answer hidden 对齐 teacher answer hidden。
- `loss_kd`：学生三分类 logits 对齐 teacher logits。
- `loss_ce`：学生 logits 对真实 TimeOfDay-3 标签做交叉熵。

Teacher 不在这个循环中运行。训练读取提前生成的 `teacher_cache`，因此不会在每个 student batch 中再执行一次电子 teacher forward。

## 2. 数据规模和 batch 数量

BDD100K TimeOfDay-3 类别顺序为：

```text
0 = daytime
1 = night
2 = dawn_dusk
```

新版主配置不再永久截断训练集：

```text
train_limit_per_class               = null
teacher cache                       = 完整 BDD100K train source
validation_fraction                 = 0.1，先从完整数据分层划分
train_samples_per_class_per_epoch   = 5000
student_batch_size                  = 4
每个 epoch student samples          = 最多 5000 × 3 = 15000
每个 epoch batch 数                 = 最多 15000 / 4 = 3750
```

服务器实际完整 source 为 69726 张；分层切出 validation 后，student train 为 `daytime=33055, night=25174, dawn_dusk=4524`。因此每个 epoch 实际抽取 `5000 + 5000 + 4524 = 14524` 张，即 batch size 4 时为 3631 个 batch。每类 epoch window 会随 epoch 轮换，并在 batch 中交错三个类别，所以多轮训练可以覆盖完整数据。旧日志的 `3375` 来自先永久截断到每类 5000、再切掉 10% validation 的旧逻辑；新版不再这样处理。

## 3. Teacher 与 Student 总览

### 3.1 Teacher：完整电子 Qwen3-VL-2B

```text
BDD image + prompt
  -> Qwen processor / chat template / tokenizer
  -> vision patch embedding
  -> 24 个电子 vision transformer blocks
  -> vision merger
  -> 与 text token embeddings 组成 multimodal sequence
  -> 28 个电子 language decoder layers
  -> final RMSNorm
  -> 最后一个非 padding token hidden [B, 2048]
  -> teacher MLP 2048 -> 1024 -> 3
```

Teacher 的 Qwen 参数全部冻结。Teacher MLP 单独使用缓存的 answer hidden 训练。完成后，teacher logits 也写入缓存供 student KD 使用。

### 3.2 Student：vision optical4 + language optical4

```text
BDD image + prompt
  -> Qwen processor / chat template / tokenizer                 保留
  -> Qwen vision patch embedding                               冻结
  -> VisionOpticalStackSurrogate，4 次光电转换                  训练
  -> Qwen vision merger                                        冻结
  -> multimodal sequence / token embedding                     冻结
  -> LanguageOpticalStackSurrogate，4 次光电转换                训练
  -> Qwen final RMSNorm                                        冻结
  -> answer-position hidden [B, 2048]
  -> student MLP 2048 -> 1024 -> 3                              训练
```

Student 不保留原始 transformer block 的电子 residual bypass。光学 surrogate 的输出直接替代完整 stack 输出。

## 4. 服务器第一批实际张量尺寸

当前服务器保存的 `metrics/first_batch_shapes.json` 表明：

| 张量 | 实际尺寸 | 含义 |
|---|---:|---|
| `input_ids` | `[4, 80]` | 4 个图文样本，每个 padding 后序列长度 80 |
| `attention_mask` | `[4, 80]` | 文本/多模态有效 token 掩码 |
| `pixel_values` | `[720, 1536]` | 4 张图片共 720 个 packed patch，每 patch 1536 维 |
| `image_grid_thw` | `4 × [1, 10, 18]` | 每张图的时间、高、宽 patch grid |
| 每图视觉 token 数 | `180` | `1 × 10 × 18`，进入 vision stack 前 |
| vision packed hidden | `[720, 1024]` | 4 张图拼接的 vision hidden |
| teacher vision target | 每张 `[180, 1024]` | 完整 24 层电子视觉栈的输出 |
| vision optical field | 4 层均为 `[4, 256, 256]` | 每张图拥有独立光场 |
| teacher answer hidden | `[4, 2048]` | 缓存的 teacher 答案位置特征 |
| student answer hidden | `[4, 2048]` | language optical4 后的答案位置特征 |
| language optical field | 4 层均为 `[4, 256, 256]` | 每个文本/多模态序列拥有独立光场 |
| logits | `[4, 3]` | daytime/night/dawn_dusk 三分类 |

`pixel_values` 的 1536 来自：

```text
3 个 RGB 通道 × temporal_patch_size 2 × patch_size 16 × patch_size 16
= 3 × 2 × 16 × 16
= 1536
```

Processor 当前固定 `min_pixels=max_pixels=50176`，但会保留图像纵横比，所以实际服务器样本得到 `[1,10,18]` 的视觉 grid，而不是强制正方形 grid。

## 5. 输入与 processor

每张图片与固定 prompt 组成 Qwen chat message：

```text
Classify this driving scene into one of the following time-of-day conditions: daytime, night, dawn_dusk. Answer:
```

处理流程：

1. PIL RGB 图片进入 Qwen3-VL processor。
2. `apply_chat_template()` 插入 Qwen 官方 image placeholder 和文本格式。
3. Image processor 完成缩放、patch 化和视觉输入构造。
4. Tokenizer 生成 `input_ids` 和 `attention_mask`。
5. Processor 返回 `pixel_values`、`image_grid_thw`、`input_ids`、`attention_mask`。
6. 整批输入从 host 移到 GPU。

这一阶段不是可训练网络，但每个 student batch 都会执行，因此包含 CPU 图像处理和 tokenizer 时间。

## 6. 视觉路径逐层结构

### 6.1 Vision patch embedding：保留且冻结

Qwen3-VL-2B 的视觉 patch embedding 是一个 Conv3D：

| 属性 | 值 |
|---|---:|
| 输入通道 | 3 |
| 输出通道 / vision hidden size | 1024 |
| kernel | `[2, 16, 16]` |
| stride | `[2, 16, 16]` |
| bias | true |

服务器当前 batch：

```text
packed patches [720, 1536]
  -> patch embedding
packed vision hidden [720, 1024]
```

### 6.2 Teacher 的 24 个电子 vision blocks

Teacher 中 `visual.blocks[0:24]` 全部保留。24 个 block 使用相同模板：

```text
x [N, 1024]
  -> LayerNorm(1024, eps=1e-6)
  -> multi-head self-attention
       qkv Linear(1024 -> 3072, bias=True)
       16 attention heads
       head_dim = 64
       vision RoPE
       output Linear(1024 -> 1024)
  -> electronic residual add
  -> LayerNorm(1024, eps=1e-6)
  -> MLP
       Linear(1024 -> 4096)
       GELU tanh approximation
       Linear(4096 -> 1024)
  -> electronic residual add
  -> [N, 1024]
```

视觉 stack 深度为 24，vision intermediate size 为 4096。Teacher cache 只保存 block 23 之后的 stack-level output，不保存各 block 的中间输出。

### 6.3 Student 的视觉 stack 替换

Student 的模块替换方式：

```text
visual.blocks[0]    = VisionOpticalStackSurrogate
visual.blocks[1:24] = VisionBypass
```

`VisionBypass` 只原样返回 hidden，不进行电子计算。完整 24-block 视觉 stack 的表示能力被一个 optical4 surrogate 替代，而不是“每 4 个 blocks 对应一层光学”。

Vision surrogate：

```text
packed [sum(T_i), 1024]
  -> 根据 cu_seqlens 拆成每张图 [T_i, 1024]
  -> LayerNorm(1024)
  -> Linear(1024 -> 256)
  -> ReLU，得到非负编码 [T_i, 256]
  -> bilinear tokens_to_field [T_i, 256] -> [256, 256]
  -> OpticalConversion × 4
  -> bilinear field_to_tokens [256, 256] -> [T_i, 256]
  -> Linear(256 -> 1024)
  -> 拼回 packed [sum(T_i), 1024]
```

关键约束：每张图片单独构造一个 `[256,256]` 光场。batch 只增加并行样本数，不会把多个样本拼进同一光场。

### 6.4 Vision merger：保留且冻结

Vision optical stack 的 `[T_i,1024]` 输出继续进入原始 Qwen vision merger：

```text
LayerNorm(1024)
  -> spatial merge 2 × 2
  -> 每 4 个相邻 token 拼成 4096 维
  -> Linear(4096 -> 4096)
  -> GELU
  -> Linear(4096 -> 2048)
```

当前每张图 180 个 pre-merge visual tokens。按 2×2 spatial merge 后，主视觉序列约为 45 个 2048 维 visual embeddings，随后注入 language sequence。

Qwen3-VL 配置还包含 deepstack visual indexes `[5, 11, 17]`。Student 的 blocks 1 到 23 都是 bypass，因此这些位置不再表示不同深度的电子 transformer 特征；它们看到的是 optical surrogate 输出沿 bypass 继续传递的结果。对应的 Qwen deepstack merger/injection 结构仍保留且冻结。

## 7. 多模态序列构造

Qwen 将 text token embedding 中的 image placeholder 位置替换成 vision merger 输出，形成统一的：

```text
multimodal hidden [B, sequence_length, 2048]
```

服务器首批 padding 后为 `[4,80,2048]`。保留的部分包括：

- tokenizer 和 chat template；
- token embedding，词表大小 151936；
- vision embedding 注入逻辑；
- multimodal position IDs / MRoPE 相关输入构造；
- final language norm；
- answer-position 提取。

## 8. Language 路径逐层结构

### 8.1 Teacher 的 28 个电子 decoder layers

Teacher 中 28 层完整运行。每层模板为：

```text
x [B, S, 2048]
  -> RMSNorm(2048, eps=1e-6)
  -> grouped-query causal self-attention
       q: Linear(2048 -> 2048), 16 heads × 128
       k: Linear(2048 -> 1024),  8 KV heads × 128
       v: Linear(2048 -> 1024),  8 KV heads × 128
       q/k head RMSNorm(128)
       multimodal RoPE / MRoPE
       SDPA attention
       attention dropout = 0
       output Linear(2048 -> 2048)
  -> electronic residual add
  -> RMSNorm(2048, eps=1e-6)
  -> SwiGLU MLP
       gate Linear(2048 -> 6144)
       up   Linear(2048 -> 6144)
       SiLU(gate) * up
       down Linear(6144 -> 2048)
  -> electronic residual add
  -> [B, S, 2048]
```

其他主要属性：

| 属性 | 值 |
|---|---:|
| decoder layers | 28 |
| hidden size | 2048 |
| intermediate size | 6144 |
| attention heads | 16 |
| KV heads | 8 |
| head dimension | 128 |
| vocabulary size | 151936 |
| max positions | 262144 |
| RoPE theta | 5000000 |
| MRoPE sections | `[24,20,20]` |
| attention bias | false |
| attention dropout | 0 |

### 8.2 Student 的 language stack 替换

Student 的替换方式：

```text
language_model.layers[0]     = LanguageOpticalStackSurrogate
language_model.layers[1:28]  = LanguageBypass
```

Language surrogate：

```text
hidden [B, S, 2048]
  -> 根据原始 2D attention_mask 取每个样本有效 token [S_i, 2048]
  -> LayerNorm(2048)
  -> Linear(2048 -> 256)
  -> ReLU，得到非负编码 [S_i, 256]
  -> bilinear tokens_to_field [S_i, 256] -> [256, 256]
  -> OpticalConversion × 4
  -> bilinear field_to_tokens [256, 256] -> [S_i, 256]
  -> Linear(256 -> 2048)
  -> 写回 [B, S, 2048]，padding 位置保持为 0
```

同样只有一个 input adapter 和一个 output adapter。4 次光学转换之间没有 Linear、CNN、transformer 或电子 residual bypass。

### 8.3 Final norm 与 answer hidden

Language optical4 输出经过原 Qwen final RMSNorm：

```text
RMSNorm(2048, eps=1e-6)
```

代码使用 `attention_mask` 找到每个样本最后一个非 padding token：

```text
answer_position = max(position where attention_mask == 1)
answer_hidden = final_hidden[batch_index, answer_position]
```

得到 `[B,2048]`，代表模型准备在 `Answer:` 后输出答案时的位置特征。

## 9. MLP 分类头

Teacher MLP 和 student MLP 结构相同：

```text
[B, 2048]
  -> Linear(2048 -> 1024)
  -> GELU
  -> Dropout(p=0.1)
  -> Linear(1024 -> 3)
  -> logits [B, 3]
```

Student MLP 从已训练的 teacher MLP checkpoint 初始化，然后与 vision/language optical surrogate 一起继续训练。

## 10. 参数规模与可训练范围

服务器模型报告：

| 部分 | 参数量 | 是否训练 |
|---|---:|---|
| Qwen3-VL-2B backbone | 2,127,532,032 | 否 |
| 原始 vision 模块 | 406,957,056 | 否 |
| 原始 language 模块 | 1,720,574,976 | 否 |
| Vision optical4 surrogate | 1,051,908 | 是 |
| Language optical4 surrogate | 1,579,268 | 是 |
| Student MLP | 2,101,251 | 是 |
| Student 总可训练参数 | 4,732,427 | 是 |

每个 optical surrogate 的参数组成：

### Vision surrogate

```text
LayerNorm(1024)                     2,048
input adapter 1024 -> 256          262,400
4 × optical conversion             524,292
output adapter 256 -> 1024         263,168
合计                              1,051,908
```

### Language surrogate

```text
LayerNorm(2048)                     4,096
input adapter 2048 -> 256          524,544
4 × optical conversion             524,292
output adapter 256 -> 2048         526,336
合计                              1,579,268
```

8 次 optical conversions 合计包含：

```text
phase 参数       = 8 × 256 × 256 = 524,288
amplitude 参数   = 8 × 256 × 256 = 524,288
detector bias    = 8
```

## 11. 训练损失

总损失为：

```text
L_total = 1.0 * L_vision
        + 1.0 * L_answer
        + 0.5 * L_KD
        + 0.5 * L_CE
```

具体为：

```text
L_vision = mean_i MSE(
    LayerNorm(student_vision_output_i),
    LayerNorm(teacher_vision_output_i)
)

L_answer = MSE(
    LayerNorm(student_answer_hidden),
    LayerNorm(teacher_answer_hidden)
)

L_KD = T² * KL(
    log_softmax(student_logits / T),
    softmax(teacher_logits / T)
)

L_CE = CrossEntropy(student_logits, labels)
```

温度 `T=2.0`。LayerNorm 后做 hidden MSE，减少 teacher/student hidden 绝对尺度差异对蒸馏的干扰。

## 12. 为什么一个 epoch 仍约 15 分钟

旧配置的 15 分钟对应：

```text
900 秒 / 3375 batch ≈ 0.267 秒/batch
```

每个 batch 仍执行：

1. PIL 图像读取、Qwen image processor、chat template 和 tokenizer。
2. Qwen patch embedding、vision merger、多模态 embedding 和 final norm。
3. Vision optical stack 的 4 次传播。
4. Language optical stack 的 4 次传播。
5. 每次传播都在 pad 后的 400×400 complex field 上执行 FFT2 和 IFFT2。
6. 共 8 次 conversion，即 forward 至少 8 次 FFT2 + 8 次 IFFT2。
7. 对 8 次光学传播、两个 adapter 和 MLP 做完整反向传播。
8. 四项 loss 计算，并从 CPU teacher cache 读取目标。

因此这个循环虽然没有在线 teacher，但也不是一个轻量 MLP 训练。Student DataLoader 为了读取 cache 使用 `num_workers=0`；新版 sampler 采用类别交错和 shard 局部性，并配合多 shard LRU，避免同类连续 batch 和随机访问造成的重复 shard 加载。

如果只想缩短实验时间，优先顺序是：

1. 在显存允许时提高 `student_batch_size`，观察每 epoch 总时间而不是单 batch 时间。
2. 将 `train_samples_per_class_per_epoch` 从 5000 降到 2000 或 1000；不要使用 `train_limit_per_class` 永久删除训练样本。
3. 用 smoke/debug 配置先确认 loss 和验证指标趋势。
4. 不建议为了 baseline 随意改变 optical field/padding，因为这会同时改变物理模型和实验结论。

## 13. 当前日志的合理解读

`running_top1=0.6948` 是 epoch 1 前 820 个训练 batch 上的累计训练准确率，不是 validation/test 准确率。它可能受到类别难度、teacher MLP 初始化和训练集拟合影响。

判断实验是否有效，应至少等 epoch 结束后联合查看：

- `validation_top1_accuracy`
- `validation_macro_f1`
- `validation_balanced_accuracy`
- 三类 per-class recall
- `loss_vision` 与 `loss_answer` 是否持续下降
- student 与 teacher 的最终准确率差距

不要用单个 epoch 中途的 `running_top1` 作为最终结论。
