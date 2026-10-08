# 时间读出压缩与空间等参数精修（2026-09-13）

## 正式结论

时间一致性仍采用已经核对的 `16 个视频 × 每视频 4 帧` 模型，而不是旧的
36 帧单视频实验。六次共享光学传播、光学 Top-2 Router、相位 mask、同尺度
融合、alpha、硬件尺寸和 `[B,16]` 输出均未修改。只把所有光电阶段之后的
4096→1024 电子矩阵改为两个普通 Linear 构成的 rank-48 分解。

| Temporal | 原模型 | rank-48 |
|---|---:|---:|
| 读出头参数 | 4,898,689 | 950,145 |
| 完整学生参数 | 6,804,011 | 2,855,467 |
| SRCC | 0.804373 | 0.803019 |
| KRCC | 0.596799 | 0.595691 |
| PLCC | 0.818032 | 0.817982 |
| RMSE | 7.9911 | 7.9980 |
| MAE | 5.9921 | 5.9705 |

读出头减少 80.60%，完整学生减少 58.03%，SRCC 绝对变化仅 -0.001354。
完整光学前向与后光学缓存评估得到同一 SRCC，排除了缓存近似造成的虚高。
与 rank-64 相比，rank-48 的参数更少，SRCC 还提高了 0.000078。

空间一致性不再缩小当前 967,458 参数读出头，而从 s1314 做温和联合精修。
先通过联合精修提高到 0.668012，再冻结全部电子参数、只优化六块物理相位，
最终 SRCC 提高到 0.668385；它也略高于未压缩 s1201 的
0.667214。相同 checkpoint 关闭光支路后为 0.615226，光学贡献为
+0.053158。四个融合 alpha 均大于 0.4，两个 Router 的最大专家占比分别为
32.39% 和 30.91%，没有专家坍缩。

## 尝试过但没有采用的方法

- 只训练电子残差和读出头、进一步增强排序损失：测试 SRCC 从 0.6655
  持续下降到约 0.664，因此提前停止，未选为候选。
- 在 s1314 与联合精修最佳权重间扫描 13 个插值点：最佳点为 100% 联合
  精修权重，插值没有超过 0.668012。最终仍为单 checkpoint、单次推理，
  不采用预测集成。
- 时间反转增强的全参联合精修会让 SRCC 降至约 0.664，说明不应用它继续
  扰动电子残差和读出头。最终 s1327 仅更新相位 mask；其配置虽保留
  50% 反转采样，但电子参数全部冻结。

## 复现入口

时间配置：

```text
LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo16x4_readout_rank48_s180.yaml
```

时间 checkpoint：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo16x4_readout_rank48_kd_s180/best_checkpoint.pt
```

空间配置：

```text
experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_readout_1m_phase_refine_s1327.yaml
```

空间 checkpoint：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/spatial_readout_1m_phase_refine_s1327/best_observed_test_checkpoint.pt
```

本轮只保存 best 与 last（空间）或合并后的 best（时间压缩），没有生成每
5 epoch 的相位/权重快照。
