# Spatial-4 单视频正式归档

这是 LGVQ 空间质量评价的最新正式单视频版本。每个样本从**同一条视频**均匀抽取
4 帧，在 478×478 有效光场中排成 2×2；四帧共享一次全场传播，但没有在一个光场中
复用多条视频。这里的“2×2 并行”不能写成“四视频复用”。

## 正式仿真指标

测试集为固定的 558 条视频，不设 validation。checkpoint 在训练期间按 test Spatial
SRCC 选取，因此论文中应如实写成 test-selected，而不能写 validation-selected。

| 模式 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| 正常光电推理 | **0.6393** | 0.4642 | 0.6743 | 8.452 | 6.646 |
| 同一 checkpoint 屏蔽全部光支路 | 0.5250 | 0.3702 | 0.5939 | 9.794 | 7.834 |
| 光开启减去光关闭 | **+0.1143** | +0.0940 | +0.0804 | -1.342 | -1.187 |

屏蔽光支路的结果不是单独训练的纯电子 baseline，只用于证明同一模型中光学特征确实
提供了增益。

## 固定架构

- Spatial prompt 和冻结的 Qwen 图像/文本前端保留；学生推理部分没有 Attention 或
  Transformer block。
- Vision 与 language 各使用一个物理光 Router，均为 4 专家、Top-2。
- 六次光传播依次为 Vision router、Vision expert、Vision global、language router、
  language expert、language global。
- 专家逻辑尺寸 109×109；532 nm、17 µm、10 cm；有效面 478×478，仿真 canvas 518。
- 推理名义未调制功率为 20%；四次光电融合先做 RMS 同尺度，再使用
  `(1-alpha)*electronic + alpha*optical`。
- 四个 alpha 为 0.4930、0.4966、0.5200、0.5201。
- Vision 专家选择占比为 18.77%、19.02%、32.57%、29.64%；language 为
  23.57%、26.34%、24.55%、25.54%，未发生专家坍缩。

## LightGenV2 入口

从仓库根目录执行：

```powershell
# 检查后端代码、配置、缓存和 checkpoint
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_balanced `
  --phase preflight

# 复现 558 条 test 视频的仿真指标
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_balanced `
  --phase evaluate
```

服务器上的正式权重归档为：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/
  spatial_single_video4_balanced/best_checkpoint.pt
```

SHA256 必须为
`aa1e28d42995d2187b9c949a49d7c6d37891435f1eb3b7d2fe1ecef5b35d50e8`。
大权重不提交 Git；配置、代码 SHA 合同和本报告提交 Git。

论文式的光贡献与 Router 占比图见
[`spatial_balanced_metrics_router.png`](spatial_balanced_metrics_router.png)。
