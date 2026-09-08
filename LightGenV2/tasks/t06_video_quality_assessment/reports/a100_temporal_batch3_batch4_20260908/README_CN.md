# A100 时间质量 batch=3/4 与 Ours 能耗复核（2026-09-08）

## 结论

- 正式 `batch=4` 在 LGVQ 全部 558 条 test 上得到：SRCC `0.769496`、PLCC `0.780177`、KRCC `0.567953`、RMSE `8.7130`、MAE `6.7609`。
- 正式 `batch=4` 的模型区间平均耗时为 `112.714 ms / 4 videos`；换算成与 Ours 相同的 16 视频工作量为 `450.854 ms`。对应实测 A100 板卡能耗为 `22.211 J / 4 videos`，即 `88.846 J / 16 videos`。
- 在同一种短时 steady-sweep 协议下，`batch=3` 已占 A100 250 W 额定功率的 `90.09%`，`batch=4` 为 `91.41%`。后者仅高 `1.32` 个百分点，因此二者都已接近该工作负载的功率平台。
- 计入承载残差、CCD 融合、必要 bridge 和任务头的 A100 板卡能耗后，LGVQ 时间质量 Ours 的组合能耗由原先只列光学设备的 `0.901 J` 修正为 `1.634 J / 16 videos`。
- 使用正式 `batch=4` Qwen3-VL 与 Ours 对比：速度优势 `40.23×`，组合能耗优势 `54.37×`。相对同口径正式 `batch=16` 的 `36.17×` 与 `50.22×`，batch=4 下光学优势略增，但不是数量级变化。

## batch=3 与 batch=4 功率占用（同协议）

这部分只使用 `3 warm-up + 30 timed forwards` 的 steady-sweep。每个 batch 中均为不同视频，每个视频抽 4 帧；预处理只做一次且不计入下表的模型延迟。

| Batch | 平均耗时/批 | 吞吐率 | A100 平均功率 | 额定功率占比 | 平均 GPU 利用率 | 能耗/批 | 能耗/视频 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 87.515 ms | 34.280 video/s | 225.221 W | 90.09% | 80.00% | 19.710 J | 6.570 J |
| 4 | 110.911 ms | 36.065 video/s | 228.525 W | 91.41% | 83.38% | 25.346 J | 6.337 J |

`batch=4` 的每视频能耗比 `batch=3` 低约 `3.55%`，吞吐率高约 `5.21%`。因此若显存允许，batch=4 更合适，但“显卡更满”并不是主要变化。

## 正式 batch=4 结果

正式结果使用单进程、单次模型加载，顺序跑完 558 条 test；不做显式 warm-up，首个 test batch 也计入统计。模型计时边界为“进入第一个原生 Vision Transformer block”到“GPU 上连续质量分数就绪”。模型/processor 加载、MP4 解码、裁剪缩放、processor/tokenizer、H2D 和 block 0 之前的 patch embedding 均不计入主延迟。

| 项目 | 数值 |
|---|---:|
| test 视频数 | 558 |
| 满 batch 数 | 139（另有最后 2 条） |
| SRCC / PLCC | 0.769496 / 0.780177 |
| KRCC | 0.567953 |
| RMSE / MAE | 8.7130 / 6.7609 |
| 模型平均耗时 | 112.714 ms / 4 videos |
| 模型中位数 / P95 | 109.909 / 110.893 ms |
| 折算 16 视频耗时 | 450.854 ms |
| A100 平均功率 | 197.061 W |
| A100 能耗 | 22.211 J / 4 videos |
| 折算 16 视频能耗 | 88.846 J |

正式全 test 和短时 sweep 的功率窗口不同，因此正式能耗只使用正式运行自身的 `telemetry.csv`；功率占比讨论只在相同 sweep 协议内比较，不能用 sweep 的 228.525 W 替换正式运行的 197.061 W。

## Ours：计入 A100 后的修正能耗

旧表中的 Ours 能耗只包含 `80.388 W × 组合推理时间` 的光学设备项。现在增加 A100 板卡项：

```text
E_ours = 80.388 W × T_wall
       + P_A100,idle × T_wall
       + Σ_serial (P_component - P_idle) × t_component
       + Σ_parallel-residual (P_component - P_idle) × t_component
```

并行电子残差不增加总延迟，但其高于 idle 的增量能耗仍然计入。串行项包括 CCD 后神经融合、必要的 modality/sequence bridge 和任务头。router 后处理、下一块 SLM 排布/重建、文件 I/O 与绘图排版不在本次神经推理边界内。

| 任务 | Ours 时间 | 旧光学设备项 | A100 板卡代理值 | 修正组合能耗 | 组合平均功率 | 理论上界（A100=250 W） |
|---|---:|---:|---:|---:|---:|---:|
| LGVQ 时间质量（并行 16 视频） | 11.207 ms | 0.901 J | 0.733 J | **1.634 J** | 145.81 W | 3.703 J |
| LGVQ 空间质量（单视频 4 帧） | 10.949 ms | 0.880 J | 0.698 J | **1.578 J** | 144.09 W | 3.618 J |
| ABO 图搜文 | 10.027 ms | 0.806 J | 0.640 J | **1.446 J** | 144.21 W | 3.313 J |
| LSP 关键点 | 5.913 ms | 0.475 J | 0.384 J | **0.860 J** | 145.40 W | 1.953 J |
| SALICON 显著性 | 6.178 ms | 0.497 J | 0.420 J | **0.916 J** | 148.35 W | 2.041 J |
| OpenMoji 语义交互 | 11.562 ms | 0.929 J | 0.745 J | **1.675 J** | 144.84 W | 3.820 J |

这里的 A100 数值是“相同 A100 上的组件功率与严格 wall-median 时间”组成的代理估计，不是光学设备与 GPU 接到同一功率计后的墙插实测。论文表格应标注为 composed estimate，并同时保留上界列。

## 修正后的速度与能耗优势

| 任务 | Qwen3-VL 时间 | Ours 时间 | 加速比 | Qwen3-VL 能耗 | Ours 组合能耗 | 能耗优势 |
|---|---:|---:|---:|---:|---:|---:|
| LGVQ 时间质量（16 视频，Qwen batch=4） | 450.854 ms | 11.207 ms | **40.23×** | 88.846 J | 1.634 J | **54.37×** |
| LGVQ 空间质量 | 74.438 ms | 10.949 ms | 6.80× | 6.242 J | 1.578 J | 3.96× |
| ABO 图搜文 | 43.963 ms | 10.027 ms | 4.38× | 3.792 J | 1.446 J | 2.62× |
| LSP 关键点 | 18.016 ms | 5.913 ms | 3.05× | 1.452 J | 0.860 J | 1.69× |
| SALICON 显著性 | 19.141 ms | 6.178 ms | 3.10× | 1.551 J | 0.916 J | 1.69× |
| OpenMoji 语义交互 | 49.053 ms | 11.562 ms | 4.24× | 4.476 J | 1.675 J | 2.67× |

ABO 图搜图当前没有对应的 Ours 光学执行图，因此仍应留空，不能套用图搜文的时间或能耗。

## 数据和复算

- `evidence/formal_batch4/`：batch=4 全 test 的预测、逐 batch 计时和 50 ms 原始功率遥测。
- `evidence/sweep_batch3/`：batch=3 的原始逐采样功率遥测与结果。
- `evidence/t06_temporal_batch_sweep_a100.json`：batch=1/2/4/8/16 的同协议 sweep 参照。
- `evidence/ccd_fusion_only_a100.*`、`task_heads_a100.json`：Ours 严格串行组件时间。
- `evidence/optical_moe*_a100.json`、`t04_optical_electronics_power_a100.json`：各组件 A100 功率和并行残差时间。
- `summary.json`：完整结构化汇总，并记录每个 evidence 文件的 SHA-256 与字节数。
- `ours_combined_energy.csv`、`comparisons.csv`：可直接填表的数据。
- `build_report.py`：可复算脚本；运行 `python build_report.py` 会重建 JSON/CSV。

服务器原始目录：

```text
/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_temporal_batch3_sweep/
/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_temporal_batch4_formal/
```
