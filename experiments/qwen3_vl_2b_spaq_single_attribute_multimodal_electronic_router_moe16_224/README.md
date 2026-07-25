# Qwen3-VL-2B SPAQ Optical MoE16-224

这是一个独立的 SPAQ 单属性图像质量蒸馏实验。它从
`qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9`
派生，但不会修改或复用其学生权重。

本版本的目标是在不使用额外 attention prelude 的情况下，将 vision 和
language Transformer stack 分别替换为一个更大、参数量约束明确的同构光学
MoE：

- 4×4，共 16 个相位专家；
- 每个专家为 224×224；
- 相邻专家有效区间隔 30 像素，pitch 为 254；
- 16 个专家组成的 footprint 为 986×986；
- global phase 的有效范围同样为 986×986；
- 四周各加 20 像素传播 guard band，FFT 画布为 1026×1026；
- 每个专家 4 个 phase/OEO stage；
- 输入相关电子 router 执行 top-4；
- 每次 phase 到 CCD 的传播距离均为 10 cm；
- 最终 CCD 只读取 986×986 有效区，再池化到 224×224；
- vision 与 language 两套光学参数不共享。

## 模型流程

Qwen processor 仍使用 RGB 图像、chat template 和短任务 prompt。完整电子
Qwen teacher 冻结，只用于生成 teacher hidden cache。学生保留 frozen patch
embedding、vision merger、多模态注入和 final RMSNorm。

一个学生光学 core 的流程为：

```text
hidden [T,H]
-> Linear(H,224)
-> LayerNorm(224)
-> Softplus
-> zero-pad token rows to [224,224]
-> electronic top-4 router
-> weighted copies loaded on 4 selected amplitude-SLM regions
-> ideal 4f identity relay to co-planar phase SLM
-> 4 x (expert phase -> 10 cm propagation -> square-law CCD
        -> selected-expert non-affine LayerNorm -> ReLU
        -> reapply original routing weight -> hard-zero unselected experts
        -> zero-phase amplitude reload)
-> 986x986 global phase
-> 10 cm propagation
-> final square-law CCD, crop [20:1006,20:1006]
-> AdaptiveAvgPool2d(224,224)
-> non-affine per-token LayerNorm(224)
-> ReLU
-> valid token rows only
-> Linear(224,H)
-> fixed residual: X + optical_delta
```

理想 4f relay 不显式执行数值传播；仿真直接把 amplitude 与 phase 作用在同一
平面。这是理想共面模型，不额外引入 4f 损耗、像差或偏移。

## 为什么保留两处 LayerNorm

Transformer 常见的 pre-LN 形式是 `X + F(LN(X))`。LN 放在分支之前，使恒等
残差路径不被归一化改变，深层反向传播更稳定。本实验没有 attention prelude，
但 optical input adapter 后仍使用 `LayerNorm(224)`，用于稳定加载到光路的非负
特征。

光学层间的 LayerNorm 解决的是另一个问题：平方律探测使强度分布偏斜且动态
范围扩大。它必须位于 `|E|²` 之后才会直接处理这个问题。当前采用每个已选专家
独立、无仿射 LayerNorm，再执行 ReLU；由于 LN 会消除入射幅度尺度，随后重新
乘回同一次输入 router 给出的 routing weight。未选专家在每次重新加载前强制为
零。最终 CCD 的 readout 也使用无仿射 per-token LN。

这种设计强调空间对比而不保留绝对光功率。若后续要比较能量敏感方案，应单独
做 RMS/mean normalization 消融，不应把 detector LN 简单移动到平方探测之前。

## 参数量

默认 vision + language optical 模式：

| 项目 | 参数量 |
|---|---:|
| 单 core 专家相位 | 3,211,264 |
| 单 core global phase | 972,196 |
| 单 core光学相位合计 | 4,183,460 |
| vision adapters | 460,448 |
| language adapters | 920,224 |
| 每个 router | 3,152 |
| regression head | 6,145 |
| vision + language 学生可训练参数总数 | 9,760,041 |

attention 参数为 0，残差支路为固定系数 1，因此也没有额外 scale 参数。

## 可复用缓存

冻结 Qwen 的缓存不放在某个 run 中，而放在：

```text
experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/cache/<task>/<identity>/
```

其中包含 `teacher_cache/` 和 `processor_cache/`。identity 由数据划分、
任务、prompt、Qwen 模型、processor 像素预算、dtype 等生成。同一输入配置、
不同 `output_dir` 会复用缓存；任何影响输入或 teacher target 的字段变化都会
得到新 identity。`cache/` 已被 Git 忽略。

不使用 attention 并不意味着缓存可以删除：teacher cache 保存完整电子 Qwen
的 vision taps 和 answer hidden，processor cache 保存图像与文本经过 Qwen
processor 后的张量。没有它们，每个学生 epoch 都要重复跑电子 teacher 或
重复解码高清 SPAQ 图像，速度会显著下降。

## 任务

四个单任务配置均已提供：

- MOS
- Brightness
- Colorfulness
- Contrast

默认学生为 vision optical MoE + language optical MoE。另有 MOS 的
vision optical MoE + frozen electronic language 诊断配置。
