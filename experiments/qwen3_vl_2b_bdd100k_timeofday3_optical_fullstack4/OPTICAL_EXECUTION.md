# fullstack4 光学部分逐步执行与 phase mask 诊断

本文只解释 `qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4` 的光学路径。完整 Qwen 输入与电子保留层见 `ARCHITECTURE.md`。

## 1. 光学替换的实际含义

当前 student 有两个相互独立的 optical stack：

```text
VisionOpticalStackSurrogate   = 4 次 OpticalConversion
LanguageOpticalStackSurrogate = 4 次 OpticalConversion
```

因此一个 student forward 共执行 8 次光电转换。它不是“一个 4-layer 模块重复使用”，而是 8 组各自独立的 phase mask、amplitude mask 和 detector bias：

```text
vision_phase_1 ... vision_phase_4
language_phase_1 ... language_phase_4
```

每个 phase mask 为 `[64,64]`，有 4,096 个独立可训练相位参数。传播时中心 pad 到 `[128,128]`。

## 2. Vision optical4 的完整执行过程

### 2.1 恢复单图 token 边界

Qwen vision hidden 是 packed 格式：

```text
[sum(T_i), 1024]
```

服务器首批为 `[720,1024]`。代码必须使用 Qwen 提供的 `cu_seqlens` 恢复每张图边界：

```text
cu_seqlens -> [180, 180, 180, 180]
```

然后分为 4 个 `[180,1024]`。如果没有可靠边界，代码会直接报错，不允许把整个 batch 映射成一个光场。

### 2.2 电子 input adapter

对每张图：

```text
H_i [180,1024]
  -> LayerNorm(1024)
  -> Linear(1024 -> 256)
  -> ReLU
Z_i [180,256], Z_i >= 0
```

ReLU 在 adapter 后面，而不是直接作用于 Qwen 原 hidden，因此 adapter 可以先学习如何把正负 hidden 投影成适合非负光场编码的表示。

### 2.3 token-to-field

第一版使用二维 bilinear interpolation：

```text
[180,256] -> [64,64]
```

这里把“token 轴 × optical feature 轴”视为二维平面并插值到光场。这是可微的工程近似，不代表 token 天然位于真实二维光学坐标。它会保留低频连续性，但不保证光场每个像素具有图像空间中的直接物理含义。

### 2.4 四次光电转换

```text
field_0 [B,64,64]
  -> OpticalConversion 1 -> intensity_1 [B,64,64]
  -> OpticalConversion 2 -> intensity_2 [B,64,64]
  -> OpticalConversion 3 -> intensity_3 [B,64,64]
  -> OpticalConversion 4 -> intensity_4 [B,64,64]
```

四次转换之间没有 Linear adapter、CNN、transformer 或电子 residual bypass。

### 2.5 field-to-token 和 output adapter

对每张图：

```text
intensity_4 [64,64]
  -> bilinear interpolation [180,256]
  -> Linear(256 -> 1024)
  -> student vision output [180,1024]
```

四张图再拼回 `[720,1024]`，直接进入冻结的 Qwen vision merger。

这里明确没有：

```text
output = original_hidden + optical_output
```

实际是：

```text
output = output_adapter(optical_tokens)
```

所以不存在原电子 hidden 的 residual bypass。

## 3. Language optical4 的完整执行过程

Language 输入为 multimodal sequence：

```text
[B,S,2048]
```

服务器首批为 `[4,80,2048]`。处理步骤：

1. 使用原始二维 `attention_mask` 找到每个样本的有效 token。
2. padding token 不参与 token-to-field。
3. 每个样本单独进行 `LayerNorm(2048)`。
4. `Linear(2048->256) + ReLU` 得到 `[S_i,256]`。
5. Bilinear interpolation 得到独立 `[64,64]` 光场。
6. 连续执行 4 次 OpticalConversion。
7. Bilinear interpolation 恢复 `[S_i,256]`。
8. `Linear(256->2048)` 恢复 language hidden。
9. 写回 `[B,S,2048]`，padding 位置为 0。
10. 进入冻结的 Qwen final RMSNorm，再提取 answer position。

Vision 和 language 的光场互不共享；两个 stack 的 4 个 masks 也互不共享。

## 4. 单次 OpticalConversion 的逐行物理过程

配置：

| 参数 | 值 |
|---|---:|
| field size | 64×64 |
| padded propagation size | 128×128 |
| wavelength | 532 nm |
| pixel pitch | 8 µm |
| propagation distance | 5 cm |
| phase initialization | 配置项 `phase_init`，默认 `zeros` |
| amplitude logits initialization | 全 4，sigmoid 后约 0.982 |
| detector bias initialization | 0 |

相较旧的 `256/400/17 µm` 配置，有效光学面的物理宽度从约 `4.352 mm`
缩小为 `0.512 mm`，padding 面从约 `6.8 mm` 缩小为 `1.024 mm`。在传播距离仍为
5 cm 时，这不仅减少计算，也会改变衍射尺度和裁剪损耗，因此应作为一个新的光学配置记录，
不要与旧 checkpoint 的 phase mask 直接混用。

`phase_init` 支持以下取值：

- `zeros` / `identity`：全零相位，当前默认；
- `uniform` / `uniform_0_2pi`：在 `[0, 2π]` 均匀随机初始化；
- `normal` / `small_normal`：均值 0、标准差由 `phase_init_std` 指定。

`amplitude_mask_enabled=true` 会额外学习逐像素透过率
`sigmoid(amplitude_mask_logits)`；设为 `false` 时只有 phase mask，调制退化为
`exp(j * phase_mask)`。该开关不影响 teacher cache。

输入是 intensity-like nonnegative field：

```text
I_in [B,64,64]
```

### 4.1 输入非负化和均值归一化

```text
I_0 = ReLU(I_in)
I_1 = I_0 / clamp(mean_hw(I_0), eps)
eps = 1e-6
```

所以每次 conversion 的输入都经过：

- 非线性：ReLU；
- 归一化：每个样本按整个 64×64 光场的空间均值归一化。

### 4.2 complex field loading

```text
E_0 = complex(I_1, 0)
```

当前实现把归一化后的 intensity-like 数值直接作为 complex field 的实部。主训练路径中没有执行：

```text
sqrt(intensity)
```

这符合当前 fullstack4 实验“检测强度直接传到下一次 conversion”的定义，但要注意：从严格物理量命名看，它是 intensity-like state 作为下一层实场载入的可微仿真，而不是严格的 `amplitude=sqrt(intensity)` 重编码。

### 4.3 phase 和 amplitude mask

每层调制为：

```text
M_phase = exp(j * phase_mask)
M_amp   = sigmoid(amplitude_mask_logits)
M       = M_phase * M_amp
E_masked = E_0 * M
```

参数尺寸：

```text
phase_mask              [64,64], FP32, trainable
amplitude_mask_logits   [64,64], FP32, trainable（仅 amplitude_mask_enabled=true）
```

相位 mask 放在传播前。若把纯相位 mask 紧贴在 square-law detection 前且中间没有传播，`|E exp(jφ)|²=|E|²`，相位会完全抵消并拿不到有效梯度。当前顺序避免了这个问题。

### 4.4 pad 和角谱传播

中心 pad：

```text
[64,64] -> [128,128]
```

角谱传播：

```text
F = FFT2(E_masked_padded)
F_prop = F * H(fx, fy)
E_prop = IFFT2(F_prop)
```

传递函数：

```text
H(fx,fy) = exp(j * 2π/λ * z * sqrt(1-(λfx)²-(λfy)²))
```

倏逝波区域在当前实现中置 0。传播后中心 crop 回 `[64,64]`。

### 4.5 square-law detection

```text
I_detected = |E_prop|²
```

这是每次 conversion 的光电检测步骤。

### 4.6 检测后归一化与非线性

```text
I_norm = I_detected / clamp(mean_hw(I_detected), eps)
I_out  = ReLU(I_norm + detector_bias)
```

所以每层拿到检测强度后，确实同时执行了：

1. 每样本空间均值归一化；
2. trainable scalar detector bias；
3. ReLU-like detector nonlinearity。

`I_out` 直接传入下一次 conversion，没有 `sqrt(I_out)`。

## 5. 一次 forward/backward 的计算量

每个 OpticalConversion 都包含一个 128×128 complex FFT2 和一个 IFFT2。

一个 student forward：

```text
vision 4 conversions   -> 4 FFT2 + 4 IFFT2
language 4 conversions -> 4 FFT2 + 4 IFFT2
合计                   -> 8 FFT2 + 8 IFFT2
```

训练还要对这 8 次传播反向求导，因此实际代价明显高于只看 forward 的 16 次 FFT。当前 batch size 4、每 epoch 3375 batches，约 15 分钟即约 0.267 秒/batch。

## 6. 为什么 phase mask 看起来“割裂”

当前 mask 看起来碎，不等价于“没有训练”。有六个直接原因。

### 6.1 每个像素都是独立参数

单个 phase mask 有 65,536 个自由参数。当前没有：

- total variation 平滑正则；
- 邻域连续性约束；
- Zernike/低频基函数参数化；
- 制造分辨率约束；
- phase quantization；
- block sharing。

优化器可以让相邻像素走向完全不同的相位，这在数学上是允许的。

### 6.2 相位是周期变量

可视化使用：

```text
wrapped_phase = phase mod 2π
```

例如原始相位 `-0.001` 会显示成约 `6.282`，而相邻的 `+0.001` 显示成 `0.001`。两者在物理上几乎相同，但普通数值图上位于色条两端，看起来像一次剧烈断裂。

因此应使用 cyclic colormap，并同时观察 `cos(phase)`、`sin(phase)` 或原始未 wrap 相位统计，不能只看 wrapped phase 图。

### 6.3 phase 和 amplitude mask 同时学习

每层还有同尺寸的 trainable amplitude mask。优化可能把一部分任务交给 amplitude transmission，phase 的形态不一定像传统透镜或平滑 DOE。

### 6.4 token-to-field 不是自然图像坐标

Vision surrogate 把 `[token, feature]` 矩阵插值成二维光场；language surrogate 把 `[sequence, feature]` 插值成二维光场。特别是 feature 轴并不是物理空间坐标，因此训练出来的相位没有理由呈现传统光学元件的平滑圆对称结构。

### 6.5 蒸馏目标是高维 representation

Phase mask 优化目标不是“把光聚焦到一个亮点”，而是同时降低：

- 1024 维 vision stack output loss；
- 2048 维 answer hidden loss；
- teacher logits KD；
- hard-label CE。

对应的最佳散射/干涉图案本来就可能很复杂。

### 6.6 第一轮尚不能判断收敛

`epoch 1 batch 820/3375` 只完成首轮约 24%。此时看到的 mask 只能代表早期更新。当前配置在 epoch 1、10、20、30 保存可视化；至少应比较多个 epoch 的相位变化和验证集指标。

## 7. 如何判断 phase mask 是否真的在训练

不要只看 phase 图。应检查以下证据。

### 7.1 参数梯度非零

在 `loss_total.backward()` 后、`optimizer.step()` 前检查每层：

```text
phase_mask.grad is not None
phase_mask.grad.norm() > 0
```

Vision 和 language 的 8 个 masks 都应分别检查。

### 7.2 参数相对初始化发生变化

当前 phase 初始值全 0。每个 epoch 记录：

```text
raw phase mean
raw phase std
raw phase min/max
mean absolute phase
phase update norm
wrapped circular variance
```

如果 raw phase std 和 update norm 长期严格为 0，才说明 phase 没有更新。

### 7.3 优化器确实包含 phase 参数

当前 optimizer 参数来源为：

```text
replacement.vision_surrogate.parameters()
replacement.language_surrogate.parameters()
student_head.parameters()
```

因此 phase masks、amplitude masks、detector biases、input/output adapters 和 MLP 均在 AdamW 中。原 Qwen 参数冻结。

### 7.4 loss 与 validation 联合变化

有效学习至少应体现为：

- `loss_vision` 或 `loss_answer` 有稳定下降趋势；
- validation top1/macro-F1 不只是训练 top1 上升；
- 不同 epoch 的 phase raw statistics 有变化；
- phase grad norm 在多数 batch 非零且非 NaN；
- amplitude sigmoid 不应全部饱和到 0 或 1。

### 7.5 当前图像本身不足以证明失效

当前 `visualization.py` 只保存 `phase mod 2π` 的 overview，没有保存：

- 原始 phase 值；
- phase 梯度；
- 与初始化的差值；
- amplitude mask；
- circular statistics。

因此仅凭现有 overview 判断“没有训动”证据不足。

## 8. 当前实现的研究限制

以下是需要在论文中明确说明的近似，不应把它们误写成真实硬件已经完整实现：

1. Token-to-field 和 field-to-token 使用 bilinear interpolation。
2. 检测后的 intensity-like state 作为下一层 complex field 实部载入。
3. 每次 conversion 都假设可执行检测、归一化、ReLU 和重新加载。
4. Phase mask 目前是逐像素连续实数，没有量化和制造约束。
5. Amplitude mask 也参与训练，系统不是纯 phase-only optical network。
6. Vision 和 language 各自有 4 次独立光电转换，共 8 次，而不是整个模型总共 4 次。
7. 当前 mask 没有平滑正则，因此不能预期得到平滑的物理透镜外观。

## 9. 对 mask 形态的后续改进顺序

如果目标是先验证当前代码是否正确，建议保持模型不变，先增加诊断，不要立即加平滑约束。推荐顺序：

1. 记录 8 层 phase 的 grad norm、raw std 和 update norm。
2. 同时可视化 raw phase、wrapped phase、cos phase、amplitude transmission。
3. 确认 loss 和 validation 指标能随 epoch 改善。
4. 若梯度正常但 mask 过于高频，再单独做带 TV regularization 的可控对照实验。
5. 若面向真实 SLM/DOE，再增加 phase quantization、空间分辨率和制造约束。

不建议仅为了让图“好看”就直接平滑 phase，因为这会改变模型容量和 baseline 定义，必须作为明确的实验变量报告。
