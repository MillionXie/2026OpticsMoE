# T12 现有大／小模型架构审计（2026-09-26）

审计对象是当前大版权重 `expanded_large_v2_unified_editor.pt` 与改进后的小版权重
`sourcegate_best_model.pt`。本报告核对**实际推理代码、权重、数据构造**，并做固定样本的
seed 与光学消融。它不把训练脚本中的计划描述当成实际部署架构。

## 一句话结论

现在的模型可以诚实地叫作“冻结 Qwen 词嵌入条件驱动的单步光电图像编辑器”。
**不能**叫作“把原始 Qwen Transformer block 换成光学 block”：原版 Qwen 的
28 个预训练语言 block 在这两份模型的生成前向中均没有运行；运行的是另外训练的两层
窄版 Qwen 式语言头。光学运算位于图像生成瓶颈，而非语言 Transformer 的替换位置。
大版的一个缩窄 UNet 和一套 VAE 是真实存在的；小版没有 VAE 或 Diffusers UNet，
但有一个带 skip 的 CNN 编解码器。当前 seed 作用真实但视觉上几乎可忽略。

## 输入到输出：小版，9,502,665 counted parameters

1. 输入图裁成 256² RGB，归一化到 [-1,1]。文字经 Qwen3-VL-2B 原版的
   processor/tokenizer 和**冻结的 311,164,928 参数词嵌入表**，得到最多 64 个
   2048 维 token 向量。Qwen 视觉塔、原版语言 block 的 forward、LM head
   都不参与该路径。通用文字推理入口是 `qwen_mini_infer.py`；展示脚本为已知
   prompts 使用提前缓存的词嵌入，不能把缓存展示当成零开销的任意文字推理。
2. 新训练的 `2048→512` 投影、两层 512 宽 Qwen 式 causal attention/RoPE/
   SwiGLU block、RMSNorm，取末个有效 token，经 `512→160` 投影得到条件向量。
   这些层**不是**原版 Qwen 的预训练 block。文字头共 6,376,096 参数。
3. seed 生成与图同形的 Gaussian noise，乘 0.08 后与输入 RGB 拼成 6 通道。
   Stem `6→48`，三个 stride-2 down block `48→96→160→224`，到 32²。
4. 32²×224 瓶颈从同一输入并行计算电子残差与光学支路。电子支路为
   depthwise+pointwise 条件残差；光学支路有 amplitude/readout **电子层**、
   四路相位 Router→Top-2 专家→global 相位 FFT 模拟，随后 RMS 同尺度融合。
   alpha 约 0.500。光学支路加融合约 132,868 参数，其中真正的相位 mask
   只有 9,216 个数。此处光学**不是语言 TF block**。
5. 三层有 skip 的上采样块 `224→160→96→48`，`to_delta` 输出三通道残差。
   另有 49 参数的网络预测 source-retention gate，最终
   `tanh(sigmoid(gate)·atanh(input)+delta)` 得到 256² RGB。没有 VAE，
   没有 Diffusers UNet，也没有迭代扩散或 GAN。

小版主要参数：文字头 6.376M；下采样 1.326M；上采样 1.367M；电子瓶颈
0.276M；光学支路及融合 0.133M；其余 stem/输出头很小。因此“小版主要是光学
模型”也不成立，但光学运算在消融中确实有明显功能影响。

## 输入到输出：大版，142,544,637 counted parameters

1. 输入图同样为 256² RGB。冻结 SD-Turbo VAE encoder + quant conv 把它编码成
   `4×32×32` reference latent（34,163,664 参数）。文字仍由 Qwen 原版
   tokenizer/冻结词嵌入得到 2048 维 token 向量；原版 28 个语言 block 的
   forward 不执行。实际加载器先加载 Qwen 模型再裁到 1 层，但推理代码只调用
   `embed_tokens`，这层原版 block 仍未被调用，加载峰值内存不等于参数报告。
2. 独立训练的 `2048→640` 投影、两层 640 宽 Qwen 式 block、RMSNorm，
   取末 token；`640→2048` bridge 连接条件适配器。文字头+bridge 共
   11,904,288 参数；不是原版 Qwen block 的直接保留。适配器可训练参数
   4,273,280；另外有 10,171,648 个**固定 PCA basis 等 buffer 数值**，
   不在“参数量”中，但推理确实加载并执行矩阵乘法。
3. 只有**一个** BK-SDM-v2-tiny 拓扑的缩窄 Diffusers UNet，通道宽度
   `[128,256,512]`，`in_channels=8`，总 42,682,473 参数。输入是
   `[reference latent, 0.05×seeded Gaussian latent]`。UNet 做一次前向，
   输出残差乘 0.75 后加到 reference latent；并非从纯噪声生成。
4. 原 BK-SDM 空的 `mid_block` 位置插入 628,197 参数的光电模块：
   参数零的电子 identity 与 FFT Router/Top-2 专家/global 分支并行，
   RMS 融合 alpha 约 0.510。它处于最深瓶颈、**进入 decoder up blocks 之前**；
   后续 up blocks 都还是电子 UNet。实际相位 mask 只有 576 个数。故先前若称
   “光替代一个 decoder residual transform”或“光学 decoder 占半数”，均不准确。
5. 冻结 VAE decoder + post quant conv（49,490,199 参数）一次解码 RGB。
   Euler scheduler 初始化了一步并传入 sigma，但当前 `one_step_edit` 明确
   `del sigma`，也固定 timestep=0；它不是标准的 SD-Turbo 一步采样公式。
   `design_router` 有 30,733 参数被打包且计入总量，现有通用推理入口不调用，
   实际是训练辅助／冗余状态。

大版 counted 参数账：文字 11.904M + VAE encoder 34.164M + UNet 42.682M
+ 适配器 4.273M + 未调用 router 0.031M + VAE decoder 49.490M = 142.545M。
其中光电 mid-block 0.628M，仅占 counted 参数约 0.44%；这 0.628M 也包含
电子 input/output projection/readout，不全是光学相位参数。额外排除共享的
311.165M Qwen 词嵌入以及适配器固定 10.172M buffer。口径用于项目内部
“可训练/模块参数”比较可以，但论文中需同时报告实际加载资产和 buffer。

## 数据到底让它学了什么

三类产品是 lamp/table/pillow；训练 source 有 1,728 个视图，验证/测试各
192 个视图。每个 source 派生 12 个编辑配对，目标款式来自**固定的 12 件商品**
（每类 4 件）。背景不是任意自然照片：由 4 种房间几何模板 × 3 种色调 × 2
种亮度 × 2 个窗光方向共 48 个控制组合进行程序绘制，40 组用于训练、8 组
留作测试；每张背景有按样本 ID 确定的小幅几何/颜色噪声。训练 target 使用
`Image.composite` 将指定商品原图及其 mask 与程序背景合成。

**区别必须讲清：** 推理代码不访问目标商品目录，也不调用 `Image.composite`
去贴目标；输出确实由网络计算。可是训练的监督目标来自固定目录商品和固定
规则背景，模型学会的是受限的条件重绘/商品编辑，不足以据此证明“能按任意
新文字生成未见过的商品或背景”。任意 prompt 可以送进推理接口，能力却没有
相应的开放词汇验证。它是广义的条件图像生成，但不是常见的开放式文生图。

## seed 是真的，但目前几乎不起作用

固定 9 个测试配对，四个 seed 11/29/47/83，直接比较输出 RGB，像素均在
[-1,1]。小版 seed 作用于 RGB 输入噪声×0.08，大版作用于 latent 噪声×0.05；
两版 VAE/输出其余部分均是确定性的。不同 seed 的输出平均绝对差：

| 模式 | 小版 seed 间 RGB MAE | 大版 seed 间 RGB MAE | seed 变化/目标编辑幅度 |
| --- | ---: | ---: | --- |
| 背景 | 0.000673 | 0.000639 | 小 0.35%，大 0.33% |
| 物体 | 0.000600 | 0.000335 | 小 0.51%，大 0.28% |
| 两者 | 0.000841 | 0.000489 | 小 0.17%，大 0.10% |

因此 seed **不决定**选哪件商品、哪套背景，也没有可展示的强多样性；
基本只是微弱纹理扰动。把这些结果宣传为“同 prompt 可生成明显不同的样本”
不符合实测。固定 prompt/输入/seed 的输出基本确定。

## 光学贡献：不要拿 alpha 代替消融

在同一 288 个测试配对中，把 FFT expert 与 global 输出设为零，同时保留电子
支路、identity/base、融合及其他网络权重：

| 版本/模式 | 原 RGB MSE | 去衍射 RGB MSE | 变化 |
| --- | ---: | ---: | ---: |
| 小版 背景 | 0.001789 | 0.005278 | 约 2.95× |
| 小版 物体 | 0.002475 | 0.028468 | 约 11.50× |
| 小版 两者 | 0.003753 | 0.032174 | 约 8.57× |
| 大版 背景 | 0.005464 | 0.005496 | +0.6% |
| 大版 物体 | 0.001764 | 0.001894 | +7.4% |
| 大版 两者 | 0.003021 | 0.003223 | +6.7% |

这说明小版在当前权重下实质性依赖衍射模拟；大版有贡献但并不主导生成，
大多数容量仍在电子 UNet/VAE。消融是在训练后直接置零，有分布漂移，
只证明“当前权重依赖程度”，不是重新训练无光 baseline 的最终性能差。
alpha 是两条**RMS 归一化后的特征**的融合系数，不是光学 FLOPs、光子能量、
参数占比，也不能代替性能消融。

## 与原始论文目标的符合程度

| 原始要求 | 当前状况 | 审计判定 |
| --- | --- | --- |
| Qwen 为基础 | 使用 Qwen tokenizer/冻结词嵌入；两层窄语言头为新训练的 Qwen 式 block | 部分符合，不能叫保留两层原 Qwen |
| TF block 换成光 | 光在图像瓶颈；Qwen 式两层仍全电子 | 不符合 |
| 末端可有 decoder | 大版有 UNet+VAE；小版有 RGB CNN decoder | 符合“允许 decoder”，但大版电子容量占主导 |
| 电残差与光并行 | 两版瓶颈的两支路都吃同一输入并融合 | 局部符合 |
| 光学有真实衍射 | 使用可微 phase-only FFT 模拟；没有在本轮实机 SLM 上验证 | 模拟符合，硬件未验证 |
| 条件生图且 seed 多样 | 条件编辑成立；seed 效果极小、目标封闭 | 仅有限条件生成 |

原 Qwen3-VL-2B 的 28 个语言 block 合计 1,409,408,000 参数，平均
**每层 50,336,000**。在“严格计入 block 参数、小版 <10M”约束下，保留
哪怕一个**原样**预训练 block 都数学上不可能。要满足小版指标，只能采用
结构裁剪/蒸馏的 Qwen-derived student；该模型可合理继承 Qwen 信息，
但应明确它不是原版 block。若必须展示“语言 block 光替代”，下一版需要把
至少一个**学生语言 token mixer**移到光学分支并给出逐块替换、蒸馏与消融，
不能靠改名把现有图像瓶颈算作 Qwen block。

## 论文与组会建议用语

当前可以展示为：**Qwen-embedding-conditioned single-pass hybrid optical
product editor**。小版可强调低参数和实测光学消融，大版可强调质量，但不能
宣称“大版一半 UNet 被光替换”或“Qwen Transformer block 已换成光”。
速度表里的 6.2682 ms 是从既有 `1.0447×6` 引用来的**硬件延迟假设**，
当前 compact FFT 模拟与该硬件路径的 mask 数、尺寸、调度未做等价验证；
只可标作估算，不可标作当前模型实测光学加速。

审计原始数值见同目录 `architecture_component_audit.json`、
`architecture_seed_audit.json`、`architecture_optical_ablation.json`。
