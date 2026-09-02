# 结果记录与验收标准

## 1. 参考目标

老师提供的 `Optical 500×500` 结果是本轮的最低目标，不是本工程已经取得的结果：

| Target | 参考方法 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---|---:|---:|---:|---:|---:|
| Spatial | Optical 500×500 | 0.6710 | 0.4909 | 0.7106 | 8.197 | 6.493 |
| Temporal | Optical 500×500 | 0.8604 | 0.6623 | 0.8784 | 6.721 | 5.144 |

相关系数越高越好，RMSE/MAE 越低越好。新工程的物理合同是 `518 canvas / 478 active / 109 expert`，因此表中 `500×500` 只是性能参照，不能写成新工程的实际光学尺寸。

理想验收条件是十项指标全部不差于上表。若三档均不能全部达到，必须如实报告，并优先比较 Spatial/Temporal SRCC 与 PLCC，不得只报较好的 Temporal 结果。

## 2. 三档结果表

当前状态：等待正式训练，以下单元格不得在训练完成前填入估计值。

| alpha 下界 | 最佳 epoch | 四层最终 alpha | Spatial SRCC | KRCC | PLCC | RMSE | MAE | Temporal SRCC | KRCC | PLCC | RMSE | MAE | 是否达到参考 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.20 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.35 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

结果必须直接抄自各目录的：

```text
metrics_best_observed_test.json
test_metrics.json
fusion_diagnostics.json
router_diagnostics.json
training_summary.json
```

## 3. 最终模型选择

选择顺序如下：

1. 先筛选达到参考目标的配置；
2. 在合格配置中选择 alpha 下界最高者；
3. 若同一 alpha 下界有多个 checkpoint，沿用训练代码的最高 `mean(Spatial SRCC, Temporal SRCC)`；
4. 同时检查四层实际 alpha，不能只用配置下界声称光贡献；
5. 若没有配置全面达标，报告最高可接受 trade-off，不得宣称达到目标。

建议给老师同时展示：两任务五项指标、四层 alpha、E/O RMS、Router 四专家选择占比、phase 相对初始化的变化量。这样才能证明“光占比提高”没有被电子特征尺度或专家塌缩掩盖。

## 4. 与旧 LGVQ 结果的关系

旧工程的 E1/E2/E4 电子 Router 和旧 O2 仅作为开发历史，不属于本轮正式矩阵。本轮只比较三个 alpha 下界，其他条件固定：

```text
Top-2 optical router
Qwen3-VL-2B-Instruct full Vision main merger
no DeepStack
518/478/109 geometry
four optical feature layers
same split, seed, loss, optimizer and dual readout
```

因此三档之间主要反映强制光学融合比例的性能代价，而不是 Router 类型或专家数量变化。


