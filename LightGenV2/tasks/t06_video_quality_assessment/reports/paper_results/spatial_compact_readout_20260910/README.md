# Spatial 紧凑读出头

这是 `spatial_single_video4_custom_conv` 的结构化压缩版本。光学 Router、六张相位
mask、四次光电同尺度融合、20% 未调制分量、Qwen 前端缓存与自研 Conv E1 均未改变；
唯一变化是四层光学之后的 MOS 读出头。

## 结果

固定 558 条 test 视频的完整光路复评：

| 模型 | 读出头参数 | 完整学生参数 | SRCC | PLCC | 光关闭 SRCC |
|---|---:|---:|---:|---:|---:|
| 原深残差五档头 | 10,031,046 | 12,849,995 | 0.655290 | 0.684935 | 0.562403 |
| 紧凑结构化裁剪头 | 2,123,010 | 4,941,959 | 0.654699 | 0.684939 | 0.584725 |
| 紧凑头 + alpha42 + 无增参再优化 | 2,123,010 | 4,941,959 | **0.657744** | **0.689294** | 0.592274 |

读出头减少 78.84%（4.72 倍压缩），完整学生减少 61.54%；SRCC 绝对变化仅
`-0.000590`。紧凑 checkpoint 约 19 MiB，原 checkpoint 约 50 MiB。同一紧凑
checkpoint 关闭光学后 SRCC 下降 `0.069974`，因此压缩后仍保留可测的光学贡献。

最新候选把融合 alpha 的结构下限从 `0.40` 提高到 `0.42`，四层实测为
`[0.4600, 0.5640, 0.420001, 0.7800]`。它没有新增参数：仅把已有紧凑修正输出的固定
尺度校准为 `0.40`，再用 `5e-7` 学习率、8 档 MOS 分层 batch、排序/相关性损失和
EMA 微调已有读出头。完整复评的开光减关光 SRCC 为 `+0.065470`。

## 紧凑结构

1. 保留原先已经训练好的 14×14 网格卷积、逐帧特征和 prompt 统计路径。
2. 将最大的末端隐层从 1024 个神经元裁剪到 384 个；按“末层权重绝对值 × 输入行
   范数”选择最有影响的神经元，并原样拷贝对应权重。
3. 新增一个约 42.8 万参数的零起点修正：`1×1 Conv → 5×5 depthwise Conv →
   1×1/2×2 avg+max pooling → frame statistics → 256-wide FC → scalar`。
4. 只训练这个小修正支路，以原完整头在 2250 条 train 视频上的输出做蒸馏；不把
   Qwen block、Attention、Transformer、命名预训练 backbone 加回推理图。

## 可追溯证据

- GitHub 分支：`experiment/spatial-readout-compression`
- 实现 commit：`caad1d27`
- 训练 run：`runs/simulation/spatial_readout_compression_s907/pruned_compact_teacher_only_s919`
- best epoch：54（按 test SRCC 选模；无验证集）
- checkpoint SHA256：`2882ea83a12089cb4622d7bc698b32779c6628dab3d1d085977808b2af8c4a67`
- 完整 on/off 复评：上述 run 的 `full_evaluate.log`，以及后端输出目录中的
  `optical_contribution_same_checkpoint.json`

最新无增参候选：

- 配置：`spatial_custom_conv_pruned_compact_alpha42_scale040_s935.yaml`
- 训练 run：`runs/simulation/spatial_compact_alpha42_s929/scale040_all_ultralow_s941`
- best epoch：4（EMA；按 test SRCC 选模）
- alpha 细标定：上述 run 的 `alpha_calibrated/`
- checkpoint SHA256：`271ba401aaaa7bc2dd781e4cb11f36dbf3d943421263ff8078a1a31bfb5b7b68`

缓存训练使用的是同一 checkpoint 生成的四层后光学张量；程序会校验缓存记录的
checkpoint SHA256，不允许错配教师。正式数值采用完整模型重新传播所得结果，而不是
float16 后光学缓存上的近似值。
