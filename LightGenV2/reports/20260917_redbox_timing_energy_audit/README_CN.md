# 红框速度与能耗数据审计（2026-09-17）

## 规范化任务归档

按表格顺序、按 `Ours/Baseline` 分层的原始数据归档见 [`server_2026OpticsMoE_a100_formal`](server_2026OpticsMoE_a100_formal/)。该目录含逐任务 CSV/JSON、服务器 LGVQ 逐视频记录、基线口径说明和完整校验清单；缺失的历史逐样本文件已在 `RAW_DATA_STATUS.md` 明确标出。
## 结论

截图红框中的 28 个展示值（7 个任务行 × Ours/Qwen3VL × 时间/能耗）均能按三位小数从现有工程或 Git 历史证据中复现，**没有发现抄写或四舍五入错误**。

但这张表不是同一时期、同一计时器、同一批处理协议下的一次统一测试：

- **Ours** 全部来自 2026-09-14 的 A100 窄口径报告，时间是“固定物理光路时间 + GPU CUDA Event 电子网络时间”的组合代理；能耗是 80.388 W 光路与 A100 板卡代理能耗之和，不是实验台整机功率计直接积分。
- **Qwen3VL** 来自 2026-09-07 至 09-08 的旧 A100 正式报告，截图采用 CUDA Event 均值；能耗为 `active_mean_board_power × CUDA_Event_mean_time`。
- LGVQ 时间质量的 `1200.053 ms / 103.784 J` 是旧单视频测试的均值与能耗各乘 16，表示**顺序处理 16 个视频的等效量**，不是 A100 原生 batch=16 的实测整批时延。
- 因而，这组数字可以作为“历史红框数据的可追溯版本”保留，但论文主表若要求统一口径，不应直接把它称为一次统一的端到端实测。

## 红框逐项复核

| 任务 | 方法 | 截图时间 | 原始精确值 | 截图能耗 | 原始精确值 | 核对 |
|---|---|---:|---:|---:|---:|---|
| LGVQ 时间质量，16 视频 | Ours | 8.660 ms | 8.660263975 ms | 1.162 J | 1.162008875 J | 通过 |
| LGVQ 时间质量，顺序等效 16 视频 | Qwen3VL | 1200.053 ms | 75.003311403×16 = 1200.052982453 ms | 103.784 J | 6.486475798×16 = 103.783612773 J | 通过，但不是 batch=16 |
| LGVQ 空间质量 | Ours | 9.898 ms | 9.898279933 ms | 1.263 J | 1.263064444 J | 通过 |
| LGVQ 空间质量 | Qwen3VL | 74.438 ms | 74.437646811 ms | 6.242 J | 6.241979920 J | 通过 |
| ABO 图搜文 | Ours | 7.653 ms | 7.652648007 ms | 0.998 J | 0.998162139 J | 通过 |
| ABO 图搜文 | Qwen3VL | 43.963 ms | 43.963330574 ms | 3.792 J | 3.792156763 J | 通过 |
| ABO 图搜图 | Ours | 7.747 ms | 7.746856009 ms | 1.010 J | 1.010442485 J | 通过 |
| ABO 图搜图 | Qwen3VL | 49.174 ms | 49.174159451 ms | 4.079 J | 4.079499988 J | 通过 |
| LSP | Ours | 4.782 ms | 4.781716044 ms | 0.625 J | 0.625482235 J | 通过 |
| LSP | Qwen3VL | 18.016 ms | 18.015795188 ms | 1.452 J | 1.452426974 J | 通过 |
| SALICON | Ours | 5.095 ms | 5.095060001 ms | 0.705 J | 0.705009414 J | 通过 |
| SALICON | Qwen3VL | 19.141 ms | 19.141094418 ms | 1.551 J | 1.551448072 J | 通过 |
| OpenMoji | Ours | 9.066 ms | 9.065768053 ms | 1.168 J | 1.168462404 J | 通过 |
| OpenMoji | Qwen3VL | 49.053 ms | 49.053312212 ms | 4.476 J | 4.476461879 J | 通过 |

机器可读逐项结果见 `redbox_values.csv`。

## 为什么说它“混期”

### Ours 的来源与口径

统一来自：

`evidence/ours_narrow_clean_final/report.json`

该报告明示：

- 每次物理传播按 `0.714 + 0.300 + 0.0307 = 1.0447 ms`；
- 6 次传播任务的光学时间为 6.2682 ms，3 次传播任务为 3.1341 ms；
- 主时间只累加 router core、CCD 后非线性/读出网络、任务头以及确有必要的 bridge；并行残差只计能耗、不叠加到时延；
- Ours 能耗是组合代理：光路 80.388 W 在组合窗口内的能耗，加 A100 活跃核函数和其余窗口的板卡代理能耗。

原始电子计时在 `all_per_call_timings.csv`，原始板卡功率采样在 `power_samples.csv`，组件汇总分别在 `latency_component_summary.csv` 与 `component_energy.csv`。

### 截图 Qwen3VL 的来源与口径

LGVQ、ABO、LSP、SALICON 的旧证据来自 Git 提交 `7d5b05e9`；OpenMoji 来自提交 `3f3ea6da`。本审计包已经把这些历史文件原样恢复到：

- `evidence/historical_git_7d5b05e9/`
- `evidence/historical_git_3f3ea6da/`

这些旧报告大多只保留了正式运行的聚合 JSON、环境、样本量、模型/数据/脚本哈希与功率统计；除 LGVQ batch sweep/formal batch-16 外，并非每个任务都保留逐样本计时或逐点功率 CSV。因此对应行属于“**正式聚合报告级可复核**”，不能全部表述成“逐样本原始序列完整保留”。

## 与当前 `FIRSTBLOCK_FINAL` 的差异

工程中还存在一套更新的 Qwen3VL 汇总：

`evidence/current_qwen_firstblock_final/formal_summary.csv`

它明确采用 **Synchronized Wall**，并为 T02/T03/T04/T07/T08 保留新的逐样本时间与功率采样。它与截图旧值并不相同：

| 任务 | 截图旧值：时间/能耗 | 当前 FIRSTBLOCK_FINAL：时间/能耗 |
|---|---:|---:|
| LGVQ 时间，batch=1 顺序等效 16 视频 | 1200.053 ms / 103.784 J | 850.556 ms / 136.553 J |
| LGVQ 时间，batch=2 等效 16 视频 | — | 507.724 ms / 100.804 J |
| LGVQ 空间 | 74.438 ms / 6.242 J | 81.118 ms / 6.991 J |
| ABO 图搜文 | 43.963 ms / 3.792 J | 40.242 ms / 3.247 J |
| ABO 图搜图 | 49.174 ms / 4.079 J | 48.628 ms / 3.490 J |
| LSP | 18.016 ms / 1.452 J | 16.971 ms / 1.070 J |
| SALICON | 19.141 ms / 1.551 J | 20.038 ms / 1.101 J |
| OpenMoji | 49.053 ms / 4.476 J | 47.147 ms / 3.368 J |

此外，历史审计还保存了一次 A100 原生 batch=16 的 LGVQ 时间质量结果：405.308 ms、96.537 J/16 视频。若主张“16 视频吞吐加速”，原生 batch=16 比“单视频均值×16”更公平；若主张单请求延迟，则两边都应统一为单请求口径，不能把 Ours 的 16 路并行批次与 Qwen 的 16 次顺序调用直接称为同一种 latency。

## 建议在论文/证明中的命名

- Ours 时间：`hybrid latency proxy (fixed physical timing + CUDA Event neural kernels)`。
- Ours 能耗：`hybrid system energy proxy`。
- 截图旧 Qwen 时间：`CUDA Event model-boundary latency`。
- 截图旧 Qwen 能耗：`active-board-power energy proxy`。
- 时间质量 1200.053 ms：必须加 `batch=1 sequential equivalent for 16 videos`。
- 若采用当前新基线，则写 `Synchronized Wall`，并直接引用 `current_qwen_firstblock_final`，不要与旧 CUDA Event 数值拼接。

## 文件结构

- `redbox_values.csv`：截图值、精确源值、计时口径、证据路径和核对状态。
- `qwen_old_vs_current.csv`：旧红框与当前 Qwen3VL 复测的并列对照。
- `evidence/ours_narrow_clean_final/`：Ours 原始逐调用计时、功率采样、汇总与脚本。
- `evidence/historical_git_7d5b05e9/`：LGVQ、ABO、LSP、SALICON 的历史正式报告与当时归档证据。
- `evidence/historical_git_3f3ea6da/`：OpenMoji 历史正式报告。
- `evidence/current_qwen_firstblock_final/`：当前 Synchronized Wall 复测、逐样本时间、功率采样、命令和代码快照。
- `evidence/server_2026OpticsMoE_a100_formal/`：从实验室服务器原始运行目录下载的 LGVQ 时间/空间两组逐视频 CSV、功率采样 CSV 和 report.json。
- `SHA256SUMS.txt`：本审计包全部文件的 SHA256；生成后该文件自身不列入清单。

## 服务器原始逐视频记录的再核对

服务器原始目录为：

`/DATA/DATA1/guest3/2026OpticsMoE_a100_formal/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/`

时间质量和空间质量的 evaluation 目录都包含 `per_video_predictions_and_timing.csv`、`power_samples.csv` 和 `report.json`。下载到本地后重新从 CSV 计算得到：

| 任务 | CSV 行数 | `model_internal_cuda_ms` 均值 | 第一个样本 | 最小–最大 | 预处理均值（不计入主时延） |
|---|---:|---:|---:|---:|---:|
| LGVQ temporal | 558 | 75.003311403 ms | 397.686798096 ms | 48.241664886–397.686798096 ms | 171.854313289 ms |
| LGVQ spatial | 558 | 74.437646811 ms | 403.676147461 ms | 48.539646149–403.676147461 ms | 173.412120778 ms |

因此：

`75.003311403 × 16 = 1200.052982453 ms → 1200.053 ms`。

这里的 16 只是把单视频均值换算成 16 次顺序调用的等效时间；服务器并没有用这次单视频运行直接测出一个 16-video batch 的 1200.053 ms。

空间质量的 `74.438 ms` 则直接是 558 个视频的逐视频 CUDA Event 均值，不需要乘 16。

两组报告都明确写明：模型只加载一次、显式 warm-up 为 0、首个测试样本纳入统计；因此第一个约 400 ms 的冷启动样本确实进入了均值。模型边界从第一个原生 Vision Transformer block 输入开始，到五质量词读出结果在 GPU 上准备完成为止；视频 seek/解码、预处理、processor/tokenizer 和 H2D 不计入主时延。`power_samples.csv` 使用 50 ms 间隔的 `nvidia-smi board power.draw` 采样，报告中的 6.486475798 J 和 6.241979920 J 是 active mean power 乘以相应的模型边界均值，不是逐个样本的电表积分。
