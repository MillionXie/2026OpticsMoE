# CCD 固定图样与漏光诊断

## 结论先行

不能通过拉对比度把一张受强漏光污染的 CCD 唯一恢复成理论光场。显示增强可以帮助人眼观察，
但真正可复用的校正必须使用同一相位 mask 下独立采集的暗场、漏光场和均匀场。

当前 `vision_expert/test__00__airplanes__f0f7c2fb99` 的确定性诊断使用了 128 张同一
phase mask、不同输入的 CCD 帧估计固定图样，没有用理论纹理生成或修补实测图。结果位于：

```text
experiments\lab_qwen\results\ccd_fixed_pattern_diagnostic\vision_expert_f0f7c2fb99
```

先看：

```text
OPEN_ME_COMPARISON.png
OPEN_THIS_BEST_DIAGNOSTIC.png
07_overlay_theory_red_actual_cyan.png
diagnostic_report.json
```

原始 linear/network-input PCC 分别为 `0.1746/0.2984`。50% 固定图样抑制再加 MoE4
局部坐标共模抑制后为 `0.2432/0.4532`；linear SSIM 从 `0.0651` 提高到 `0.1642`。
这说明确实存在可以分离的固定痕迹，但改善仍然有限，不能把处理结果写成“恢复后的真实光场”。

理论图上下专家区平均能量比约 `119:1`，原始实测约 `1.30:1`，处理后约 `3.43:1`。
大量能量已经进入理论暗区，属于测量前的物理混合，而不是单纯显示太暗。

## 重现命令

在独立实验包根目录执行：

```powershell
python -m experiments.lab_qwen.ccd_fixed_pattern_diagnostic `
  --actual-file experiments\lab_qwen\four_accuracy_first_full\01_vision_expert\ccd_captured\test__00__airplanes__f0f7c2fb99.png `
  --theory-file experiments\lab_qwen\qwen_theoretical_ccd_accuracy_first_full\01_vision_expert\theoretical_ccd\transport_quantized\test__00__airplanes__f0f7c2fb99.npz `
  --background-dir experiments\lab_qwen\four_accuracy_first_full\01_vision_expert\ccd_captured `
  --output-dir experiments\lab_qwen\results\ccd_fixed_pattern_diagnostic\vision_expert_f0f7c2fb99 `
  --background-samples 128 `
  --stable-fraction 0.35 `
  --maximum-shift 16
```

工具输出三档固定图样抑制，以及两档仅适用于 2×2 expert 层的共模抑制。理论图只用于
报告指标和单张位移诊断，不参与固定图样估计。`theory-assisted shift` 不能逐帧用于正式推理；
应通过多张人工探针确定一个固定几何变换。

## 下一次应补采的标定帧

每一层、每一张正式 phase mask 分别采集：

1. 激光关闭，16 帧：CCD 暗场 `D`；
2. 保持正式 phase mask，振幅 SLM 请求灰度 0，16 帧：光学漏光场 `B`；
3. 保持正式 phase mask，振幅 SLM 播放不饱和均匀场，16 帧：实测均匀场 `F_meas`；
4. 用同一均匀输入和 mask 生成理论 `F_sim`。

建议的受约束校正为：

```text
B = median(gray0 frames)
G = smooth(F_meas - B) / smooth(F_sim)
G 限制在经过标定的合理范围内
I_corrected = max(I_measured - B, 0) / G
```

然后理论和实测都执行完全相同的单帧均值归一化、相对强度截断、`log1p` 和 224×224
池化。不能对每张图分别做 min-max，也不能用理论图逐像素拟合实测图。

## 光路侧优先事项

- 在相位 SLM 傅里叶面抑制零级，或加入载波后只选一级衍射；
- 在 478×478 有效孔径外做物理限光，排查四周及底边未调制光；
- 检查 532 nm 下相位 LUT、偏振和调制深度；
- 用灰度 0 漏光场测量实际消光比。若理论暗区仍接近亮区，先修光路，再谈后处理；
- 将测得的零级比例和增益变化写回仿真噪声范围，而不是继续假定当前的 0–5%。

强零级光与信号是相干叠加：`|E_signal + E_leak|²` 含交叉项，因此暗场相减也不可能
完全恢复丢失的信息。后处理只能降低固定偏置，不能替代零级抑制和重新标定。
