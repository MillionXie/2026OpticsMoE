# Spatial 紧凑读出头结构

该头位于四次光电融合全部完成之后，不是 Qwen，也不位于光学分支之前：

```text
Qwen 前端缓存 + 自研 Conv E1 + 固定 prompt
  → 光 Router / Vision expert / Vision global
  → 光 Router / Language expert / Language global
  → 后光学 Vision [B,4,196,192] + Language [B,S,192]
  → 紧凑读出头
  → 一个连续 Spatial MOS
```

紧凑读出头内部包含两部分。基础网格路径保留已经训练的 `LayerNorm → depthwise
3×3 Conv → 1×1 Conv → 3×3 avg/max pooling`，以及逐帧和 prompt 投影。其末端
全连接层从 `2048→1024→1` 裁剪为 `2048→384→1`。补偿路径只读取相同的后光学
张量，使用 64 通道自写卷积和固定池化，再对四帧的 mean/std/max/min/相邻差分做
汇总，输出一个有界标量并加到基础预测。

总参数为 2,123,010，其中补偿路径约 42.8 万。没有 Attention、Transformer、循环
网络或外部命名 backbone。光学 mask 与 Router 不变。四次融合使用同一 RMS 尺度
后再做凸组合，配置层把 alpha 的硬下限设为 `0.42`；最终四层 alpha 为
`[0.4600, 0.5640, 0.420001, 0.7800]`，因此任何一层都不会降到 0.4 或以下。

紧凑修正支路没有新增门控参数。它原本的输出乘以固定系数 `0.40` 后再加入基础
预测，以防修正量破坏样本排序；这是常量校准，不改变参数量。随后只对现有读出头
做极低学习率（`5e-7`）微调：MOS 分层 batch、Smooth-L1、成对排序、batch 相关性、
soft-Spearman 与 EMA，按周期性 test SRCC 选择第 4 epoch。完整四层复评得到
SRCC `0.657744`、PLCC `0.689294`；同一 checkpoint 关光为 `0.592274`。

配置入口：

```text
LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/
spatial_single_video4_compact_readout.yaml
```

复评命令（服务器已有 canonical checkpoint 时）：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_compact_readout `
  --phase evaluate
```
