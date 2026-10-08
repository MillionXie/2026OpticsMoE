# Spatial-4 光贡献与最后相位随机对照

## 公平对照口径

三组都使用同一个 `best_checkpoint.pt` 和同一批 558 个固定测试视频，没有为消融组重新训练读出头：

1. **正常光电**：四次 O/E 融合、两个光 Router 和六张已训练相位全部启用。
2. **全部去光**：在同一 checkpoint 上令 `optical_enabled=false`，四个光特征阶段和两个光 Router 都被旁路，只保留电子路径。
3. **仅最后相位随机**：光路保持启用，只把 `serial_optics.raw_global_phase` 替换成独立的均匀物理相位 `Uniform[0,2π)`。它是第 4 个 O/E 阶段的 `Language global` 相位面；Router、其余五张相位、四个 alpha、电残差和最终读出头逐 tensor 不变。

## 结果

| 设置 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| 正常光电 | **0.671008** | 0.488545 | 0.688354 | **8.243705** | 6.587364 |
| 全部去光，同 checkpoint | 0.615902 | 0.443188 | 0.655070 | 9.485909 | 7.566093 |
| 最后一层随机相位，5 seeds 均值 | 0.670974 | 0.488542 | **0.688393** | 8.244110 | 6.587445 |
| 最后一层随机相位，预先指定 seed 1346 | 0.670970 | 0.488545 | 0.688433 | 8.244115 | **6.587302** |

随机相位五个固定种子为 `1346..1350`。SRCC 范围为 `0.670932–0.671001`，总体标准差仅 `2.48e-5`，没有从中挑最好种子充当主结果。

正常开启全部光学相对全部去光的 SRCC 增益是 **+0.055106**，PLCC 增益是 **+0.033283**，RMSE 降低 **1.242204**。因此完整光学路径对当前结果有明确贡献。

但是，只随机化最后一张 `Language global` 相位后，SRCC 平均只下降 `0.000034`。seed 1346 的最终 MOS 预测与已训练相位预测的 PCC 为 `0.99999938`，预测差值 RMSE 为 `0.01781 MOS`。这说明当前最终读出对这张单独的最后相位面几乎不敏感。

## 审查结论

代码检查确认最后相位后仍然执行了一次 10 cm 角谱传播，所以并不是 mask 没有进入仿真。额外探针也确认 seed 1346 会明显改变首批 64 个视频的 `Language global` CCD 张量：CCD 张量 PCC 约 `0.726`、相对 RMS 变化约 `67.6%`。变化经过第四次融合和最终读出后几乎被抑制。

因此可以对老师/师姐表述为：

- “完整光学路径”相对同 checkpoint 去光带来约 **0.0551 SRCC** 的净增益；
- 该增益主要来自 Router 和更早的光学层，**不能归功于最后一张相位 mask**；
- 当前最后一张 mask 在光场层面有效，但在最终评分层面处于弱利用状态。若论文需要证明每一层 mask 都不可替代，后续应增加逐层随机/置平消融，或重新训练时加强最后光层到读出头的可辨识贡献。

## 可复现文件

正式审计运行目录：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/
  spatial_readout_1m_srcc067/ablation_last_phase/
```

其中包括：

- `last_phase_ablation_report.json`：完整五种子指标和 SHA256；
- `predictions_optical_on.csv`、`predictions_optical_off.csv`：同 checkpoint 正常/去光逐视频预测；
- `predictions_random_last_phase_seed*.csv`：五个随机相位逐视频预测；
- `random_last_phase_seed1346.pt`：仅替换最后相位的轻量推理 checkpoint；
- `random_seed1346_hardware_masks/phase_slm_1920x1200/language_global.bmp`：对应相位 SLM BMP；
- `random_seed1346_hardware_masks/hardware_mask_export_report.json`：硬件导出合同与文件哈希。

复算命令：

```powershell
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.evaluate_last_phase_ablation `
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_readout_1m_srcc067.yaml `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/spatial_readout_1m_srcc067/best_checkpoint.pt `
  --output-dir LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/spatial_readout_1m_srcc067/ablation_last_phase `
  --seeds 1346 1347 1348 1349 1350 `
  --device cuda
```
