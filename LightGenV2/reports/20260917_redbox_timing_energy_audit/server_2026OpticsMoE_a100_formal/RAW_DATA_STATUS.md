# 原始数据状态

| 任务 | Ours 原始记录 | Baseline 原始记录 | 是否存在缺口 |
|---|---|---|---|
| LGVQ temporal | `all_per_call_timings.csv`、`latency_component_summary.csv`、`component_energy.csv`、任务功率采样，以及完整源报告 | 服务器 `per_video_predictions_and_timing.csv`、`power_samples.csv`、`report.json` | 无；表中 1200.053 ms 的基线仍是历史 batch-1×16 口径 |
| LGVQ spatial | 同上 | 服务器 `per_video_predictions_and_timing.csv`、`power_samples.csv`、`report.json` | 无 |
| ABO image-to-text | 同上 | `redbox_source_report.json`；另有 `current_firstblock_formal/` 的逐样本计时/功率/进程记录 | 有：历史红框那一次的逐样本 CSV 未在当前归档中发现，不能用新口径文件冒充 |
| ABO image-to-image | 同上 | `redbox_source_report.json`；另有 `current_firstblock_formal/` | 有：同上 |
| LSP | 同上 | `redbox_source_report.json`；另有 `current_firstblock_formal/` | 有：同上 |
| SALICON | 同上 | `redbox_source_report.json`；另有 `current_firstblock_formal/` | 有：同上 |
| OpenMoji | 同上 | `redbox_source_report.json`；另有 `current_firstblock_formal/` | 有：同上 |

这里的“有缺口”只表示“与红框数值完全同一轮的逐样本原始文件没有找到”。红框使用的历史汇总 JSON 仍然保留；后来统一首层到读出头的逐样本复测也保留，但它的时间边界不同，已明确隔离。
