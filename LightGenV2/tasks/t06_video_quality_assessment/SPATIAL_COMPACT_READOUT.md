# Spatial 紧凑读出头结构

当前正式候选是 `spatial_readout_1m_srcc067`。读出头位于四次光电融合全部完成
之后，不是 Qwen，也不绕过光学阶段：

```text
冻结 Qwen 图像/文本前端缓存 + 自写 Conv E1 + Spatial prompt
  → Vision expert O/E 融合 → Vision global O/E 融合
  → Language expert O/E 融合 → Language global O/E 融合
  → 后光学 Vision [B,4,196,192] + Language [B,S,192]
  → 同一个紧凑读出头 → 一个连续 Spatial MOS
```

读出头总计 **967,458** 个参数，完整学生网络 **3,786,407** 个参数。它只有一个
基础预测和一个读取相同后光学张量的有界修正，不是新的输入分支。

基础路径先做 `LayerNorm → depthwise 3×3 Conv → 1×1 Conv(192→64)`，每帧在
`3×3` 上做 avg/max pooling。逐帧投影、语言统计投影以及末端预测层都用两层普通
Linear 的低秩分解代替原大矩阵，秩依次为 `160/64/96/144`；末端宽度裁成 144。

修正路径使用 `1×1 Conv(192→64) + depthwise 5×5 Conv`，在 `1×1/2×2` 上做
avg/max pooling，再汇总四帧的 mean/std/max/min 和相邻帧绝对差。输出经
`2.2*tanh(x/2.2)` 限幅后乘固定 `0.40`，再加到基础预测。这里没有 Attention、
Transformer、循环网络或命名预训练 backbone。

最终四层同 RMS 尺度融合 alpha 为 `[0.48, 0.47, 0.60, 0.60]`。测试集 558 条
视频的独立进程复评为 SRCC `0.671008`；同一 checkpoint 关闭光学后为
`0.615902`。正式入口：

```text
LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/spatial_single_video4_custom_conv_readout_1m.yaml
```

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_custom_conv_readout_1m `
  --phase evaluate
```
