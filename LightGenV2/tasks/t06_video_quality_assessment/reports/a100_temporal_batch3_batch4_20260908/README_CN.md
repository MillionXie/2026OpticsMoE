# A100 时间质量 batch=2/3/4 与统一 Ours 口径（2026-09-08）

## 结论

- 正式 `batch=2` 在 LGVQ 全部 558 条 test 上得到：SRCC `0.766415`、PLCC `0.776848`；平均 `62.259 ms / 2 videos`，等效 16 视频为 `498.068 ms、103.772 J`。
- 在相同短时 steady-sweep 协议下，`batch=2/3/4` 分别占 A100 250 W 额定功率的 `84.26% / 90.09% / 91.41%`。从 3 增到 4 只提高 `1.32` 个百分点。
- 统一按原表的 CUDA-event 口径后，LGVQ 时间质量 Ours 为 `10.637 ms / 16 videos`；加入 A100 推理设备后，组合能耗为 `1.547 J`。
- 使用正式 `batch=2` Qwen3-VL 与统一 Ours 对比：速度优势 `46.83×`，组合能耗优势 `67.10×`。

## batch=2、3、4 功率占用（同协议）

这部分只使用 `3 warm-up + 30 timed forwards` 的 steady-sweep。每个 batch 中均为不同视频，每个视频抽 4 帧；预处理只做一次且不计入下表的模型延迟。

| Batch | 平均耗时/批 | 吞吐率 | A100 平均功率 | 额定功率占比 | 平均 GPU 利用率 | 能耗/批 | 能耗/视频 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 63.985 ms | 31.257 video/s | 210.656 W | 84.26% | 64.47% | 13.479 J | 6.739 J |
| 3 | 87.515 ms | 34.280 video/s | 225.221 W | 90.09% | 80.00% | 19.710 J | 6.570 J |
| 4 | 110.911 ms | 36.065 video/s | 228.525 W | 91.41% | 83.38% | 25.346 J | 6.337 J |

`batch=4` 的每视频能耗比 `batch=3` 低约 `3.55%`，吞吐率高约 `5.21%`。因此若显存允许，batch=4 更合适，但“显卡更满”并不是主要变化。

batch=2 并没有比 batch=1 的“单次前向”更快：同一 sweep 中，batch=1 为 `51.156 ms / 1视频`，batch=2 为 `63.985 ms / 2视频`。batch=2 更快的是单位视频和固定 16 视频总工作量，因为两个视频在 GPU 内并行：同协议折算后，batch=1 为 `818.503 ms、139.924 J / 16视频`，batch=2 为 `511.877 ms、107.830 J / 16视频`。

## 正式全 test：batch=2 与 batch=4

正式结果使用单进程、单次模型加载，顺序跑完 558 条 test；不做显式 warm-up，首个 test batch 也计入统计。模型计时边界为“进入第一个原生 Vision Transformer block”到“GPU 上连续质量分数就绪”。模型/processor 加载、MP4 解码、裁剪缩放、processor/tokenizer、H2D 和 block 0 之前的 patch embedding 均不计入主延迟。

| 项目 | batch=2 | batch=4 |
|---|---:|---:|
| test 视频数 | 558 | 558 |
| SRCC / PLCC | 0.766415 / 0.776848 | 0.769496 / 0.780177 |
| KRCC | 0.565905 | 0.567953 |
| RMSE / MAE | 8.7734 / 6.8165 | 8.7130 / 6.7609 |
| 模型平均耗时/批 | 62.259 ms / 2 | 112.714 ms / 4 |
| 模型中位数 / P95 | 60.813 / 62.259 ms | 109.909 / 110.893 ms |
| 等效 16 视频耗时 | **498.068 ms** | 450.854 ms |
| 正式运行 A100 平均功率 | 208.349 W | 197.061 W |
| 每批能耗 | 12.972 J | 22.211 J |
| 等效 16 视频能耗 | **103.772 J** | 88.846 J |

正式全 test 和短时 sweep 的功率窗口不同，因此正式能耗只使用各自正式运行的 `telemetry.csv`；功率占比讨论只在相同 sweep 协议内比较。不能用 sweep 的 210.656/228.525 W 替换正式运行的 208.349/197.061 W。

## Ours 唯一正式时间口径

图中 `10.637、10.600、9.941、5.861、6.125、11.236 ms` 可以由原始 CUDA-event 数据逐项精确复现，继续作为论文主表值。唯一正式定义为：

```text
T_ours = N × 1.314 ms 物理光场
       + CUDA-event median CCD→融合
       + CUDA-event median 完整任务头
```

例如 LGVQ 时间质量：

```text
7.884 ms  = 6 × 1.314 ms 物理光场
1.892 ms  = 2 × frame fusion + 2 × video fusion
0.860 ms  = 完整 16 视频质量任务头
-----------------------------------------------
10.637 ms = 唯一正式 Ours 时间
```

并行残差的时间被物理光路覆盖，不增加主延迟；bridge、router 后处理和下一 SLM 排布不属于这张表声明的“融合+任务头”边界。此前出现的 `11.207 ms` 是错误地把 synchronized wall median 和 bridge 混入 CUDA 主表所得，现已撤销；wall 数据只保留作诊断，不能用于主表倍率。

## Ours：计入 A100 后的修正能耗

旧表中的 Ours 能耗只包含 `80.388 W × CUDA 主表时间` 的光学设备项。现在增加 A100 板卡项：

```text
E_ours = 80.388 W × T_table
       + P_A100,idle × T_table
       + Σ_fusion/head (P_component - P_idle) × t_cuda
       + Σ_parallel-residual (P_component - P_idle) × t_cuda
```

并行电子残差不增加总延迟，但其高于 idle 的增量能耗仍然计入。串行项只包括主表声明的 CCD 后神经融合与任务头；bridge、router 后处理、下一块 SLM 排布/重建、文件 I/O 与绘图排版不在本表边界内。

| 任务 | Ours 时间 | 旧光学设备项 | A100 板卡代理值 | 修正组合能耗 | 组合平均功率 | 理论上界（A100=250 W） |
|---|---:|---:|---:|---:|---:|---:|
| LGVQ 时间质量（并行 16 视频） | 10.637 ms | 0.855 J | 0.692 J | **1.547 J** | 145.40 W | 3.514 J |
| LGVQ 空间质量（单视频 4 帧） | 10.600 ms | 0.852 J | 0.674 J | **1.526 J** | 143.93 W | 3.502 J |
| ABO 图搜文 | 9.941 ms | 0.799 J | 0.633 J | **1.433 J** | 144.10 W | 3.284 J |
| LSP 关键点 | 5.861 ms | 0.471 J | 0.380 J | **0.852 J** | 145.31 W | 1.936 J |
| SALICON 显著性 | 6.125 ms | 0.492 J | 0.416 J | **0.908 J** | 148.23 W | 2.024 J |
| OpenMoji 语义交互 | 11.236 ms | 0.903 J | 0.723 J | **1.626 J** | 144.72 W | 3.712 J |

这里的 A100 数值是“相同 A100 上的组件功率与 CUDA-event median 时间”组成的代理估计，不是光学设备与 GPU 接到同一功率计后的墙插实测。论文表格应标注为 composed estimate，并同时保留上界列。

## 修正后的速度与能耗优势

| 任务 | Qwen3-VL 时间 | Ours 时间 | 加速比 | Qwen3-VL 能耗 | Ours 组合能耗 | 能耗优势 |
|---|---:|---:|---:|---:|---:|---:|
| LGVQ 时间质量（16 视频，Qwen batch=2） | 498.068 ms | 10.637 ms | **46.83×** | 103.772 J | 1.547 J | **67.10×** |
| LGVQ 时间质量（16 视频，Qwen batch=4） | 450.854 ms | 10.637 ms | **42.39×** | 88.846 J | 1.547 J | **57.45×** |
| LGVQ 空间质量 | 74.438 ms | 10.600 ms | 7.02× | 6.242 J | 1.526 J | 4.09× |
| ABO 图搜文 | 43.963 ms | 9.941 ms | 4.42× | 3.792 J | 1.433 J | 2.65× |
| LSP 关键点 | 18.016 ms | 5.861 ms | 3.07× | 1.452 J | 0.852 J | 1.70× |
| SALICON 显著性 | 19.141 ms | 6.125 ms | 3.12× | 1.551 J | 0.908 J | 1.71× |
| OpenMoji 语义交互 | 49.053 ms | 11.236 ms | 4.37× | 4.476 J | 1.626 J | 2.75× |

ABO 图搜图当前没有对应的 Ours 光学执行图，因此仍应留空，不能套用图搜文的时间或能耗。

## 数据和复算

- `evidence/formal_batch2/`、`evidence/formal_batch4/`：对应全 test 的预测、逐 batch 计时和 50 ms 原始功率遥测。
- `evidence/sweep_batch3/`：batch=3 的原始逐采样功率遥测与结果。
- `evidence/t06_temporal_batch_sweep_a100.json`：batch=1/2/4/8/16 的同协议 sweep 参照。
- `evidence/ccd_fusion_only_a100.*`、`task_heads_a100.json`：Ours 严格串行组件时间。
- `evidence/optical_moe*_a100.json`、`t04_optical_electronics_power_a100.json`：各组件 A100 功率和并行残差时间。
- `summary.json`：完整结构化汇总，并记录每个 evidence 文件的 SHA-256 与字节数。
- `ours_combined_energy.csv`、`comparisons.csv`、`temporal_formal_batch_comparison.csv`：可直接填表的数据。
- `build_report.py`：可复算脚本；运行 `python build_report.py` 会重建 JSON/CSV。

服务器原始目录：

```text
/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_temporal_batch3_sweep/
/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_temporal_batch2_formal/
/DATA/DATA1/guest3/2026OpticsMoE_a100_measurements/20260908_temporal_batch4_formal/
```
