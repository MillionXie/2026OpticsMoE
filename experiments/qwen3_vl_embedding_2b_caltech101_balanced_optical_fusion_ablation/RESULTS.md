# 正式结果（2026-09-02）

本页记录服务器完成的单 seed、30 epoch Caltech101 目标 10 类消融。数据口径为
`train=2625`、`gallery=30`、`test=200`，不设 validation；epoch 5、10、15、20、
25、30 测试，并按周期 test 选权重。由于 test 被用于选模，以下数值是
test-selected 结果，不应表述为无偏泛化估计。

## 主要 EMA 对照

| 组别 | 最佳 EMA epoch | 完整模型 Top-1 | 同 checkpoint 去光 Top-1 | 光的净变化 |
|---|---:|---:|---:|---:|
| free alpha `(0.01, 0.95)` | 10 | 90.5% | 89.5% | **+1.0 pp** |
| low alpha `(0.05, 0.49)` | 10 | 90.0% | 90.0% | **0.0 pp** |
| high alpha `(0.51, 0.95)` | 15 | 70.0% | 74.5% | **-4.5 pp** |
| electronic-only retrained | 20 | 90.0% | N/A | N/A |

`remove_optical` 不是另训一个模型：它在同一个已选 checkpoint 上跳过 Vision 两层
和 Language 两层的全部光学传播，只运行相应电子 Mixer。因而“完整模型减去同权重
去光模型”是本实验回答光分支净贡献的主要因果对照。电子-only 重训回答的是电子
容量上限，不能替代上述对照。

200 个 test query 意味着一个预测对应 0.5 个百分点。因此 free 的 `+1.0 pp` 只
对应两个 query，low 的 `0.0 pp` 表示目前未分辨出净收益，不宜过度解释。

## 最佳 EMA checkpoint 中的四层 alpha

| 组别 | Vision expert | Vision global | Language expert | Language global |
|---|---:|---:|---:|---:|
| free | 0.054658 | 0.055233 | 0.054411 | 0.054468 |
| low | 0.054962 | 0.055029 | 0.054935 | 0.054937 |
| high | 0.549434 | 0.549632 | 0.548594 | 0.548552 |

因为两支路在融合前已匹配为相同 RMS，上表的 `(1-alpha, alpha)` 是明确的名义混合
系数；但当电、光特征相关时，不能把 alpha 直接解释为最终输出能量百分比。实际
任务贡献应以上一节的同 checkpoint 去光差值为准。

free/low 都停留在约 5.5% 光系数附近；强制 `alpha>0.5` 后性能明显下降。当前证据
支持“低光占比可以基本保住精度”，不支持“让光占主导即可提高性能”。

## live checkpoint 补充

周期 test 选出的最佳 live Top-1 分别为：free 90.5%（epoch 25）、low 91.0%
（epoch 15）、high 70.0%（epoch 15）、electronic-only 90.5%（epoch 10）。正式主表
统一使用 EMA，以避免在不同组之间混用 live/EMA 口径。

## 服务器证据

结果根目录：

```text
experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/
```

每组完整结果：

```text
alpha_free_scale_matched/metrics/ema_best_observed_test.json
alpha_low_lt_0p5/metrics/ema_best_observed_test.json
alpha_high_gt_0p5/metrics/ema_best_observed_test.json
electronic_only_retrained/metrics/ema_best_observed_test.json
```

四层 alpha：

```text
alpha_free_scale_matched/best_alpha_report.json
alpha_low_lt_0p5/best_alpha_report.json
alpha_high_gt_0p5/best_alpha_report.json
```

同 checkpoint 去光评估：

```text
final_eval_free_remove_optical/metrics/evaluation_summary.json
final_eval_low_remove_optical/metrics/evaluation_summary.json
final_eval_high_remove_optical/metrics/evaluation_summary.json
```

完整启动、复评和报告导出命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
