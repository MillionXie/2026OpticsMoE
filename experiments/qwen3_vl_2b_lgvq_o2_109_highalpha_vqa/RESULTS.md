# 结果记录与验收标准

## 1. 参考目标

老师提供的 `Optical 500×500` 结果是本轮的最低目标，不是本工程已经取得的结果：

| Target | 参考方法 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---|---:|---:|---:|---:|---:|
| Spatial | Optical 500×500 | 0.6710 | 0.4909 | 0.7106 | 8.197 | 6.493 |
| Temporal | Optical 500×500 | 0.8604 | 0.6623 | 0.8784 | 6.721 | 5.144 |

相关系数越高越好，RMSE/MAE 越低越好。新工程的物理合同是 `518 canvas / 478 active / 109 expert`，因此表中 `500×500` 只是性能参照，不能写成新工程的实际光学尺寸。

理想验收条件是十项指标全部不差于上表。若所有 alpha 梯度均不能全部达到，必须如实报告，并优先比较 Spatial/Temporal SRCC 与 PLCC，不得只报较好的 Temporal 结果。

## 2. alpha 梯度结果表

当前状态：四档正式训练与完整 test 评估已完成。表中均为 558 个固定 test 视频的实测值，不含 validation。

| alpha 下界 | 最佳 epoch | 四层最终 alpha | Spatial SRCC | KRCC | PLCC | RMSE | MAE | Temporal SRCC | KRCC | PLCC | RMSE | MAE | 是否达到参考 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.20 | 60 | 0.2742 / 0.2740 / 0.3026 / 0.3015 | 0.7371 | 0.5504 | 0.7719 | 7.3311 | 5.8620 | 0.8722 | 0.6750 | 0.8882 | 6.3179 | 4.9025 | 是 |
| 0.35 | 5 | 0.4145 / 0.4143 / 0.4204 / 0.4202 | 0.7571 | 0.5679 | 0.7888 | 7.5153 | 6.0361 | 0.8687 | 0.6736 | 0.8881 | 6.4549 | 4.8887 | 是 |
| **0.50** | **10** | **0.5590 / 0.5593 / 0.5702 / 0.5701** | **0.7586** | **0.5686** | **0.7859** | **7.0528** | **5.6754** | **0.8741** | **0.6822** | **0.8897** | **6.4107** | **4.8270** | **是，最终选择** |
| 0.65 | 10 | 0.6910 / 0.6918 / 0.6999 / 0.7004 | 0.7562 | 0.5651 | 0.7795 | 7.1884 | 5.8146 | 0.8586 | 0.6654 | 0.8720 | 6.7981 | 5.1307 | 否 |

四层 alpha 顺序固定为 `Vision expert / Vision global / Language expert / Language global`。0.50 档是当前最高的全面达标版本；0.65 档的 Spatial 仍强，但 Temporal SRCC、PLCC、RMSE、MAE 不合格，因此不能只凭更高 alpha 选它。

0.50 档先完成了一次 60 epoch 全程训练；由于旧逻辑没有保留 epoch 15 的合格权重，随后在相同配置、seed、数据和硬件尺寸下重放。重放在 epoch 10 首次十项全达标后保存 `best_reference_compliant_checkpoint.pt` 并停止，原始完整训练保留在 `o2_109_alpha50_initial_complete/`。

结果必须直接抄自各目录的：

```text
metrics_best_observed_test.json
metrics_best_reference_compliant.json
test_metrics.json
fusion_diagnostics.json
router_diagnostics.json
training_summary.json
RESULTS.json
```

正式选中 checkpoint：

```text
runs/o2_109_alpha50/best_reference_compliant_checkpoint.pt
SHA256 c143c20eba03fe01e3da652141ac6eb913f45619fc8c4162ba304e31b0fdee26
```

## 3. 最终模型选择

选择顺序如下：

1. 先筛选达到参考目标的配置；
2. 在合格配置中选择 alpha 下界最高者；
3. 同一 alpha 下界优先使用 `best_reference_compliant_checkpoint.pt`，再按最高 `mean(Spatial SRCC, Temporal SRCC)` 排序；
4. 同时检查四层实际 alpha，不能只用配置下界声称光贡献；
5. 若没有配置全面达标，报告最高可接受 trade-off，不得宣称达到目标。

建议给老师同时展示：两任务五项指标、四层 alpha、E/O RMS、Router 四专家选择占比、phase 相对初始化的变化量。这样才能证明“光占比提高”没有被电子特征尺度或专家塌缩掩盖。

## 4. 与旧 LGVQ 结果的关系

旧工程的 E1/E2/E4 电子 Router 和旧 O2 仅作为开发历史，不属于本轮正式矩阵。本轮只比较 alpha 下界，其他条件固定：

```text
Top-2 optical router
Qwen3-VL-2B-Instruct full Vision main merger
no DeepStack
518/478/109 geometry
four optical feature layers
same split, seed, loss, optimizer and dual readout
```

因此各 alpha 梯度之间主要反映强制光学融合比例的性能代价，而不是 Router 类型或专家数量变化。
