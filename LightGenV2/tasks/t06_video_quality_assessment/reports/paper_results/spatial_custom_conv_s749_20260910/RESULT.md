# Spatial-4 自写卷积正式候选

## 结果

固定划分为 2,250 train / 558 test，不设 validation。正式候选在完整 test 上得到：

| 模式 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| 正常光电 | **0.6553** | 0.4750 | 0.6849 | 9.122 | 7.148 |
| 同 checkpoint 关闭光 | 0.5624 | 0.3992 | 0.6146 | 9.993 | 7.968 |
| 光学贡献（on-off） | **+0.0929** | +0.0758 | +0.0704 | -0.871 | -0.819 |

## 为什么选择 s749

- 30 epoch 初次特征预训练的完整模型 SRCC 为 0.6462。
- 延长到 200 epoch 后，自写 E1 与训练期教师的 held-out feature PCC 达到 0.8279，
  完整模型 SRCC 提高到 0.6553。
- 直接联合解冻完整光电图时，最好仍是开始前的 epoch 0，后续指标下降。
- 仅联合自写卷积与 MOS 头训练 100 epoch，最好仍是 epoch 0。
- 两档 mask-only 学习率也未稳定超过 0.6553。

因此没有让较差的最后权重覆盖正式候选。

## 正式推理结构

Qwen3-VL 只在离线缓存阶段提供官方图像 `patch_embed + position embedding`，以及文本
`tokenizer + embed_tokens`。学生训练、评估和实验室推理执行 0 个 Vision block、0 个
Language block、0 个 Attention/Transformer；完整 Qwen 权重不加载。

原始 4 帧经过项目自写的卷积/全连接小模块，产生 `[B,4,196,192]` 校正量，只加入第一层
电子残差 E1。它没有分数输出，随后仍须经过四层光电融合与唯一 MOS 头。自写模块共
316,568 个参数；完整学生网络为 12,849,995 参数，其中最终读出头为 10,031,046 参数。

四层 alpha 为 `[0.45, 0.55, 0.40, 0.775]`。视觉 Router 四专家占比为
`[22.2%, 21.6%, 32.8%, 23.4%]`，语言 Router 为
`[22.9%, 20.9%, 27.4%, 28.8%]`，没有专家坍缩。

详细结构见 [SPATIAL_CUSTOM_OEO_ARCHITECTURE.md](../../../SPATIAL_CUSTOM_OEO_ARCHITECTURE.md)。

## 复现与交付

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_custom_conv `
  --phase preflight

python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_custom_conv `
  --phase evaluate
```

生成自包含实验室 ZIP：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.build_lab_package `
  --profile spatial_single_video4_custom_conv
```

正式 checkpoint SHA256：

```text
f913c96a0a82a4dea19aa3e3296ec292f70881846425b5b1f2e9cfd942f610e1
```
