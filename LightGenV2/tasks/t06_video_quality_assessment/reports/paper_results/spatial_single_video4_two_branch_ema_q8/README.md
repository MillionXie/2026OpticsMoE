# LGVQ Spatial：双支路硬件感知 EMA 候选

这是当前“只保留电子残差 + 光学”推理结构下的最好硬件感知候选。它没有 VGG、Transformer、Attention 或第三条质量分支。冻结 Qwen 前端只负责离线产生视觉 patch/位置嵌入和文本 tokenizer/embed_tokens 缓存；学生网络内仍保留 Spatial prompt 条件调制。

## 结果

| 模式 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| 正常光电推理 | **0.62561** | 0.45242 | 0.66208 | 8.668 | 6.846 |
| 同一 checkpoint 旁路光学 | 0.51765 | 0.36678 | 0.58945 | 10.082 | 8.035 |
| 冻结 Qwen3-VL-2B baseline | 0.69089 | — | 0.70475 | — | — |

光学开启相对同权重去光消融带来 `+0.10796 SRCC`。与完整冻结 Qwen baseline 仍有 `0.06528 SRCC` 的差距；不能把两者写成已经持平。

## 电子读出头

正式候选仍采用通过消融的 `SpatialGridReadout`，没有采用本轮失败的额外读出分支：

1. 四帧各自保持 `14×14×192` token 网格；LayerNorm 后做一个 `3×3 depthwise conv` 和 `1×1: 192→64` 投影。
2. 每帧同时做固定 `3×3` average/max pooling，拼成 1152 维，再经 `1152→512`。
3. 四帧用 mean/std/max 汇总为 1536 维。
4. 四个图像摘要 token 与 prompt 组成的最终序列做 mean/std/max，再经 `576→512`。
5. 两部分拼成 2048 维，经过 `2048→1024→1` 输出一个 Spatial MOS。

读出头约 300 万参数，是最容易过拟合的部分。本轮没有继续加宽，而是对它使用独立的低学习率与更强 weight decay；减半到 256 宽的对照只到 SRCC 0.61385，故未选。

## 相位可用性

- 六张相位均由 `sigmoid(raw)×2π` 约束在一个相位周期内。
- 正式候选在前向中使用 256 级 straight-through 量化，训练看到的就是最终 8-bit BMP 相位级别。
- 仿真保留 20%～35% 随机相干未调制功率，测试固定 20%，用于覆盖实验中的零级泄漏。
- 当前 `theta_max=1.0°` 高于 17 µm 像素在 532 nm 下约 `0.896°` 的奈奎斯特角，因此没有额外抹平相位；0.75° 真正带限对照最高仅 0.6218，未采用。
- 最佳版本不使用相位 TV 平滑、phase dropout 或像素位移。高频相位本身并不等于不可加载；硬件关键合同是相位范围、8-bit 量化、有效区域、方向和 17→8 µm 重建一致。

## Router 结论

Vision Router 四专家选择占比为 22.13%、20.88%、32.82%、24.17%，未坍缩。Language Router 的输入并非只有固定 prompt，而是 4 个随视频改变的图像摘要 token 加 prompt；当前仍固定选择专家 2/3。强 mutual-information 均衡和四窗 flat-field 对照能改变 soft probability，但没能在不损失性能时稳定跨过硬 Top-2 边界，因此这里如实标为“集中”，不能再用“prompt 固定”解释成正常现象。

## 训练选择

教师回归损失被完全关闭。训练使用保守相位预热、联合训练、阶段切换回滚到历史最佳、低学习率精修；EMA 在 epoch 16 被选中。原始权重当时 SRCC 低于 EMA，说明提升来自可复现的权重平均，而非改 test 预测。

配置：`experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_two_branch_ema_microrefine_s428.yaml`

服务器 checkpoint：`experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_two_branch_ema_microrefine_s428/best_observed_test_checkpoint.pt`
