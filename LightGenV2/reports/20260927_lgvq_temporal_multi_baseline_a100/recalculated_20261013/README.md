# LGVQ batch comparison: frozen reevaluation and recalculation, 2026-10-13

## LGVQ 时间质量评估的能耗与时延比较

图 a 比较时间质量评价性能与处理 16 个视频所需能量，图 b 比较同一性能与相同工作量的处理时间。两图的纵轴均为预测平均意见分数（mean opinion score, MOS）与真实 MOS 的 Spearman 秩相关系数（SRCC），不是分类准确率或百分数。横轴分别为每 16 视频的能量（J）和处理时间（ms），采用对数刻度。电子曲线上 B 表示每次 GPU 调用的视频数；LightGen-II 星形点的 N=16 表示单个光场承载 16 个视频，不是 GPU batch。当前 LightGen-II 点采用物理评估 SRCC 0.7977138739，不使用仿真 SRCC 0.8043868643。

### 实验方法和统计口径

LGVQ 固定划分包含 2,250 个训练视频和 558 个测试视频，共 2,808 个不同视频。电子基线在各视频的 10%、37%、63% 和 90% 相对位置抽取 4 帧。Qwen、DeepSeek、CLIP 和 YOLO 的输入尺寸分别为 448×448、384×384、224×224 和 640×640，后面三者保留以短边 65% 为边长的中心正方形裁剪协议。Qwen3-VL-2B、DeepSeek-VL2-Tiny、CLIP ViT-B/32 和 YOLO11s 在 B=1、2、4、8 下分别执行冻结推理，各设置均对同一完整 558 视频计算 SRCC，不把单个 B 的分数复制到其他 B。成本统计只取固定 TEST 顺序的前 200 个不同视频。16 视频是工作量归一化单位，不是性能测试集的大小。

电子基线在指定 NVIDIA A100-PCIE-40GB 上按每模型每 B 一个新进程运行，模型加载一次。执行顺序为当前批视频的解码、裁剪及 processor、CPU 至 GPU 传输、冻结前向和质量输出，然后处理下一批。计时起点是首个视觉 Transformer block 的前向预钩子（Qwen、CLIP、DeepSeek），YOLO 对应首个卷积 block。终点是 GPU 上五级质量读出生成标量 MOS 并完成 CUDA 同步。模型加载、视频解码、CPU 预处理、H2D 和随后预测复制到 CPU 不计入最终时延与能量。无显式预热，第一次测量调用保留。预钩子内部没有再次 CUDA 同步，因此这是代码实现的首 block 入口至 GPU MOS 就绪的同步墙钟边界，不等同于只累加首 block 后的纯 GPU kernel 耗时。

Qwen 与 DeepSeek 的主干采用 BF16，CLIP 和 YOLO 的视觉前向采用 FP16 或 FP16 autocast，五级质量行及 MOS 读出采用 FP32。CLIP、YOLO 和 DeepSeek 沿用原协议的 FP16 特征取值转换再转 FP32 读出。LightGen-II 电子部件计时为 FP32 CUDA event。两类成本没有被描述为相同计算精度或相同计时器的测量。

设前 200 个视频的平均 batch 同步墙钟时间为 \(\bar t_B\)（ms），则

\[
T_{16}(B)=\frac{16}{B}\bar t_B,
\qquad
E_{16}(B)=\bigl(338.2+\bar P_{\mathrm{A100},B}\bigr)\frac{T_{16}(B)}{1000}.
\]

其中 338.2 W 为既定主机代理功率，\(\bar P_{\mathrm{A100},B}\) 为该 batch 设置下推理活跃标签窗口的 A100 平均板卡功率。板卡功率按 50 ms 采样；均值功率乘边界时间是能量估算，不是逐个短调用的校准能量积分，也不加入两次推理之间的 CPU 准备时间。

| 每次调用视频数 B | 完整 558 TEST 的调用次数 | TEST 最后一批真实视频数 | 前 200 视频的成本调用次数 | 16 视频折算调用次数 | 满批调用帧数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 558 | 1 | 200 | 16 | 4 |
| 2 | 279 | 2 | 100 | 8 | 8 |
| 4 | 140 | 2 | 50 | 4 | 16 |
| 8 | 70 | 6 | 25 | 2 | 32 |

LightGen-II 使用既定单光场 16 视频配置。六个光学周期合计 \(6(0.714+0.300+0.0307)=6.2682\) ms，串行电子部件合计 3.9773695913702247 ms，包括 router 0.8302489486336708 ms、非线性/CCD 电子读出 2.0079718434810636 ms、桥接网络 0.2413414404541254 ms 和任务头 0.8978073588013649 ms，总计 10.245569591370225 ms。并行 residual 支路不另外加入串行关键路径。能量沿既定功率项计算：

\[
E_{16}^{\mathrm{LightGen-II}}
=0.0062682(80.358+41.388)
+0.0039773695913702247(41.388+62.842)
=1.1776895097085185\ \mathrm{J}.
\]

该成本来自光学周期模型和电子部件计时估算，而不是本轮重新测得的精确正式权重光路端到端耗时。对应的每视频摊销能量为 0.0736055943567824 J/video，等效任务能效为 13.5859238518309 videos/J。10.245569591370225/16=0.6403480994606391 ms/video 是摊销服务时间，不能称为单视频调用延迟。

### 图 a 能量与时间质量评价性能

在 LGVQ 时间质量评估中，LightGen-II 的物理 SRCC 为 0.7977，对应的每 16 视频估算能量为 1.1777 J。Qwen3-VL-2B 的 SRCC 在 B=1–8 下为 0.7663–0.7695，其能量由 462.1134 J 降至 227.7404 J。相同工作量下，Qwen B=1 和 B=8 的能量与 LightGen-II 能量之比分别为 392.39 和 193.38，等效 videos/J 能效比分别相同，对应约 99.745% 和 99.483% 的能量降低。CLIP ViT-B/32、YOLO11s 和 DeepSeek-VL2-Tiny 的能量分别由 B=1 时的 65.7520、131.1640 和 1329.3279 J 降至 B=8 时的 14.1378、45.4445 和 832.8558 J。它们的 SRCC 范围分别为 0.72524–0.72526、0.70159–0.70160 和 0.5680–0.5704。图 a 因此显示，在本图的推理能量估算口径内，LightGen-II 点具有更高物理参考 SRCC 和更低成本。增大 batch 时电子基线的能量下降是当前数据的观察结果，不作为所有模型或 batch 的理论约束。

### 图 b 处理时间与时间质量评价性能

LightGen-II 的每 16 视频处理时间估算为 10.2456 ms。随着 B 从 1 增至 8，Qwen3-VL-2B 的同工作量时间由 1188.3861 ms 降至 504.5565 ms，对应相对 LightGen-II 的时间比分别约为 115.99 和 49.25。CLIP ViT-B/32、YOLO11s 和 DeepSeek-VL2-Tiny 的同工作量时间分别由 173.9405、347.5287 和 2874.0137 ms 降至 37.3055、120.4859 和 1554.5023 ms。横向移动表示归一化到 16 视频的成本变化，不表示单次大 batch 调用反而更短。例如 Qwen 的平均单次调用时间从 B=1 的 74.2741 ms 增至 B=8 的 252.2782 ms，但完成 16 视频只需两次 B=8 调用。SRCC 均由各 B 的完整 558 视频预测独立计算，CLIP 和 YOLO 的跨 B 最大差异仅约 0.00001699 和 0.00000660，所以曲线近水平并非采用统一固定分数。

### 两张图的全部数据

下表列出两图全部 17 个点，便于核对正文和重画。展示 SRCC 保留 10 位小数、成本保留 6 位小数，不代表测量具有相应有效数字或已估计置信区间。绘图实际读取同目录 `plot_data.csv` 的未提前舍入数值。表中的 B 只适用于电子基线，LightGen-II 只提供 N=16 光场点。

| Model | B or N | Full TEST SRCC | Energy per 16 videos (J) | Processing time per 16 videos (ms) |
| --- | ---: | ---: | ---: | ---: |
| LightGen-II | N=16 | 0.7977138739 | 1.177690 | 10.245570 |
| Qwen3-VL-2B | 1 | 0.7663428037 | 462.113401 | 1188.386146 |
| Qwen3-VL-2B | 2 | 0.7664148425 | 289.256106 | 751.153173 |
| Qwen3-VL-2B | 4 | 0.7694958345 | 229.762004 | 554.005565 |
| Qwen3-VL-2B | 8 | 0.7694089803 | 227.740378 | 504.556478 |
| DeepSeek-VL2-Tiny | 1 | 0.5684669010 | 1329.327914 | 2874.013747 |
| DeepSeek-VL2-Tiny | 2 | 0.5703507558 | 1003.341514 | 2039.332098 |
| DeepSeek-VL2-Tiny | 4 | 0.5680463058 | 866.009778 | 1711.967508 |
| DeepSeek-VL2-Tiny | 8 | 0.5698244506 | 832.855838 | 1554.502339 |
| CLIP ViT-B/32 | 1 | 0.7252558080 | 65.751971 | 173.940464 |
| CLIP ViT-B/32 | 2 | 0.7252452404 | 35.597545 | 94.420065 |
| CLIP ViT-B/32 | 4 | 0.7252388170 | 19.244618 | 51.033648 |
| CLIP ViT-B/32 | 8 | 0.7252388170 | 14.137797 | 37.305453 |
| YOLO11s | 1 | 0.7015970082 | 131.164005 | 347.528660 |
| YOLO11s | 2 | 0.7015940037 | 72.310650 | 191.883636 |
| YOLO11s | 4 | 0.7015916554 | 53.291359 | 141.054289 |
| YOLO11s | 8 | 0.7015904121 | 45.444519 | 120.485892 |

对 Qwen 的能量比、时间比均用未舍入成本计算：

| Qwen batch B | Qwen 能量 / LightGen-II 能量 | 能量降低比例 | Qwen 时间 / LightGen-II 时间 |
| --- | ---: | ---: | ---: |
| 1 | 392.389842 | 99.745151% | 115.990247 |
| 2 | 245.613214 | 99.592856% | 73.314926 |
| 4 | 195.095568 | 99.487431% | 54.072696 |
| 8 | 193.378964 | 99.482881% | 49.246308 |

### English Methods

We compared temporal video-quality prediction and inference cost on a fixed LGVQ split of 2,250 training and 558 test videos. Electronic baselines used four frames sampled at relative positions of 10%, 37%, 63% and 90%. Qwen3-VL-2B, DeepSeek-VL2-Tiny, CLIP ViT-B/32 and YOLO11s retained input resolutions of 448×448, 384×384, 224×224 and 640×640, respectively. The latter three retained the original centered square crop with a side length of 65% of the shorter frame dimension. Frozen checkpoints were evaluated at batch sizes of 1, 2, 4 and 8 videos per GPU call. SRCC was recomputed independently from all 558 test predictions at every batch size. Cost statistics used the same first 200 test videos and were normalized to a workload of 16 videos. These batch sizes therefore corresponded to 200, 100, 50 and 25 measured calls, and 16, 8, 4 and 2 calls per normalized workload, respectively.

Qwen used its original frozen quality rows. CLIP, YOLO and DeepSeek used original-protocol reproduced five-level quality heads because their historical trained heads were not recovered. Only the quality rows were trained for 50 epochs with seed 42, training batch size 512, AdamW at a learning rate of 0.001 and zero weight decay, cosine scheduling and cross-entropy. Backbones remained frozen. Checkpoints were selected by highest TEST SRCC without a separate validation set, so these reproduced-head scores are development evaluations. No retraining or checkpoint selection was performed across the frozen batch-size comparisons.

Electronic inference was evaluated on an NVIDIA A100-PCIE-40GB in a fresh process for each model and batch size. A synchronized host-wall timer started at the forward pre-hook of the first visual Transformer block, or the first convolutional block for YOLO11s, and ended after the scalar MOS prediction was ready on the GPU and CUDA synchronization had completed. Model loading, video decoding, CPU preprocessing, host-to-device transfer and the subsequent download of predictions were excluded. No explicit warm-up was performed, and the first measured call was included. We calculated \(T_{16}=(16/B)\bar t_B\) and \(E_{16}=(338.2+\bar P_{\mathrm{A100},B})T_{16}/1000\), with time in milliseconds and power in watts. The host term is a fixed power proxy. A100 power is the mean of 50-ms telemetry samples tagged during forward execution, so the reported energy is a mean-power estimate rather than a calibrated per-call integral.

LightGen-II was represented by its previously measured physical SRCC and an approved 16-video optical-field cost estimate. Six optical cycles contributed 6.2682 ms, and serial electronic components contributed 3.9773695913702247 ms, yielding 10.245569591370225 ms and 1.1776895097085185 J per 16 videos under the specified power model. These costs are component-based estimates, not newly measured exact-checkpoint optical-bench end-to-end costs. LightGen-II electronic components were timed in FP32 using CUDA events. Qwen and DeepSeek backbones used BF16, whereas CLIP and YOLO visual inference used FP16 or FP16 autocast, with FP32 quality readout. The comparison is therefore not an equal-arithmetic-precision, identical-timer benchmark.

### English Results for panel a

LightGen-II achieved a physical temporal-quality SRCC of 0.7977 at an estimated energy of 1.1777 J per 16 videos. Across batch sizes of 1 to 8, Qwen3-VL-2B yielded SRCC values of 0.7663–0.7695, while its energy decreased from 462.1134 to 227.7404 J for the same workload. The corresponding Qwen-to-LightGen-II energy ratios were 392.39 at B=1 and 193.38 at B=8, equivalent to energy reductions of 99.745% and 99.483%, respectively. Increasing B from 1 to 8 also reduced the estimated energy of CLIP ViT-B/32, YOLO11s and DeepSeek-VL2-Tiny from 65.7520, 131.1640 and 1329.3279 J to 14.1378, 45.4445 and 832.8558 J. Under the specified inference-energy accounting, LightGen-II combined a higher physical-reference SRCC with a lower estimated energy cost than all displayed electronic baseline settings.

### English Results for panel b

The estimated LightGen-II processing time was 10.2456 ms per 16 videos. Qwen3-VL-2B required 1188.3861 ms at B=1 and 504.5565 ms at B=8, corresponding to processing-time ratios of 115.99 and 49.25 relative to LightGen-II. For CLIP ViT-B/32, YOLO11s and DeepSeek-VL2-Tiny, increasing B from 1 to 8 reduced the normalized processing time from 173.9405, 347.5287 and 2874.0137 ms to 37.3055, 120.4859 and 1554.5023 ms, respectively. These values describe a fixed 16-video workload rather than the latency of a single batch call. Each plotted SRCC was computed from the corresponding batch-specific predictions on all 558 test videos. The nearly horizontal CLIP and YOLO curves reflect measured SRCC variations below 0.000017, not scores copied across batch sizes.

### English figure caption

**Temporal quality prediction versus inference energy and processing time on LGVQ.** **a,** SRCC versus estimated energy per 16 videos. **b,** SRCC versus processing time per 16 videos. Both cost axes are logarithmic. Electronic baselines were frozen-evaluated at B=1, 2, 4 and 8 videos per GPU call. SRCC uses all 558 test videos, whereas costs use the first 200 test videos and are normalized to a 16-video workload. The LightGen-II star denotes N=16 videos per optical field and combines a physical-reference SRCC with an optical-cycle and electronic-component cost estimate. It does not denote a GPU batch-size sweep or newly measured optical-bench end-to-end costs. The Qwen B=1 cost is the retained main-table measurement. CLIP, YOLO and DeepSeek use original-protocol reproduced quality heads selected by TEST performance (development evaluation).

### 结果来源和论文表述边界

Qwen B=1 使用已确认总表成本 1188.3861464262009 ms 和 462.1134010767563 J。新 B=1 复测的 1150.4916036128998 ms 和 442.9714733293941 J 只保留在审计记录，不进入两图。B=2/4/8 为 2026-10-13 同协议复测，因此这些点不是同日完成的一个统一测速轮次，不用缩放系数强迫它们一致。

Qwen 使用原冻结质量行。CLIP、YOLO 和 DeepSeek 的历史任务头未找回，当前任务头按原协议重训，之后对每个 B 重新执行原生冻结推理。三组训练均为 2,250 TRAIN、558 TEST、seed 42、50 epochs、AdamW lr 0.001/wd 0、cosine schedule 和五级交叉熵，仅训练质量行，不训练主干。最高 TEST SRCC 选模没有独立 VAL，论文应注明为 development evaluation，不能称为独立未参与选择的 test 泛化指标。

能量的主机功率采用代理值，板卡功率采用 50 ms 活跃窗口均值。CLIP 各设置只有 10–52 个被标为 active 的采样点，DeepSeek 的原始遥测包含 307.72–346.12 W 峰值，而设备报告功率上限为 250 W。原始记录保留，未裁剪或人为修正。本文可据此描述规定推理边界内的成本估算，不能把图上的能量称为每个视频直接测得的高时间分辨率能量积分，也不能把两类不同计时器的数值称为公平同精度端到端测速。现有图不提供独立重复测量的误差条或置信区间，不据此作显著性检验。

未舍入绘图数据为 `plot_data.csv`，逐 B 成本输入为 `timing_energy_recalculated.csv`，逐 B 性能与 checkpoint 身份为 `batch_metric_audit.csv`，计算来源和 SHA 为 `recalculation.json`。LightGen-II 物理性能来源为 T06 当前版本登记，成本输入来源为 `../../20260928_nine_task_a100_table/raw_remote/ours_narrow_200/report.json`。两图使用 `srcc_cost_comparison_reference.png/svg` 的 a、b 两个面板。以下为保留的实验与历史审计记录，不替代本节当前绘图数据。

## Completed original-protocol reproduction, 20:27 +0800

CLIP, YOLO and DeepSeek have completed the authorized original-protocol
Temporal-head reproduction and all twelve fresh-process B1/B2/B4/B8 frozen
evaluations. Each setting has 558 distinct TEST predictions; costs use the
same first 200 videos. The original trained heads were not recovered. These
are new `retrained_original_protocol` PTs, not renamed historical checkpoints.
The updated workbook and figures now use actual batch-specific predictions
and costs for all four electronic baselines, rather than copying reference
SRCCs onto historical timing-only profiles.

| Reproduced head | Historical SRCC reference | New B1 full558 SRCC | Difference | Best epoch |
|---|---:|---:|---:|---:|
| CLIP ViT-B/32 | .7251568 | .7252558079932262 | +.0000990079932262 | 48 |
| YOLO11s | .7015822 | .7015970081958657 | +.0000148081958657 | 49 |
| DeepSeek-VL2-Tiny | .5688169 | .5684669009566514 | -.0003499990433486 | 43 |

The original 2250 TRAIN / 558 TEST manifest, four frames, 65% crop, original
input resolutions, frozen backbones and five quality rows were retained.
Only the Temporal rows were trained: 50 epochs, seed 42, batch 512, AdamW
lr .001 / wd 0, cosine schedule and cross-entropy. Highest TEST SRCC selection
is explicitly a development metric with no VAL, not an independent test.
Best/last strict-CPU verification and all 50 history entries were checked.

The canonical local/server run is
`LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_reproduction_a100_20261013/`.
Each model has best/last PT, consolidated feature cache, history, receipt,
strict-CPU results, four batch reports and their predictions/timing/power CSVs.
`batch_audit.json` binds all 60 batch files, 12 batch logs and final suite
status to exact SHA-256. All 73 copies match the server. Consolidated caches
are backed up locally; extraction shard parts remain protected on the server.
The three training receipts also bind source/config/manifest and frozen-model
identity. Training entry source is main `7e2b170d857a92a9dd7d902a6c63d9379a46565f`;
later per-process Git metadata differ because unrelated main commits landed,
but the evaluation entry SHA is unchanged for all twelve processes:
`8f82e2ce61bddc89a1e45ccd8d2d975a4becb87de850dc9425937763452b3445`.
All own suite/evaluation PIDs have exited; the specified A100 was idle with
17 MiB and 0% utilization at 20:24. Other GPUs/tasks were untouched.

Every SRCC was independently recomputed from its own predictions. CLIP and
YOLO vary by less than .000017 SRCC across batch settings. DeepSeek ranges
from .5680463058 to .5703507558. No monotonic quality trend is imposed.
Historical reference scores remain separately labelled: Summary column I and
Batch results column C. Summary column C is now the plot-ready actual full558
SRCC; Batch results column U contains the same actual batch-specific metric.

Costs remain inference-only mean-power estimates under the confirmed formula.
The A100 reported a 250 W power limit in every process. Raw DeepSeek telemetry
peaks of 307.72–346.12 W are retained without clipping or an asserted cause.
The 50 ms sampler has only 10–52 tagged active samples for CLIP settings and
often reads close to idle power; these are not calibrated per-call energy
integrals. This limitation is not repaired by changing the agreed formula.


## Current frozen reevaluation update

The authorized A100 Qwen B1/B2/B4/B8 reevaluation is complete, with 558 real
predictions per setting and the first 200 videos used for costs. Source is
`../raw_remote/frozen_batch_20261013/`. All four full-test scores exactly match
the historical batch-specific values below. The run used main `4cd6fe86d`,
A100 UUID `GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55`, actual limit 250 W,
no warmup and interleaved CPU preparation/H2D/inference. Model and trained
quality rows were frozen. Own processes have exited; another user's later
GPU use is not altered.

Decode, preprocessing and H2D are audit records only and are excluded from
both final latency and energy. The final synchronized-wall inference boundary
and 338.2 W host proxy plus measured inference-active A100 power are unchanged.

| Qwen videos/call | Final time for 16 videos (ms) | Final energy for 16 videos (J) | Full558 SRCC | Cost source |
|---:|---:|---:|---:|---|
| 1 | 1188.3861464262009 | 462.1134010767563 | .7663428036522781 | Preserved nine-task table measurement |
| 2 | 751.1531734466553 | 289.25610600062987 | .7664148425389119 | New same-protocol inference |
| 4 | 554.0055650472641 | 229.76200380177073 | .7694958345192725 | New same-protocol inference |
| 8 | 504.5564776659012 | 227.74037765183186 | .7694089803486667 | New same-protocol inference |

Fresh B1 measured 1150.4916036128998 ms / 442.9714733293941 J; it is retained
only in raw evidence and the separately labelled audit, not substituted into
the final B1 table. No numerical scaling is used to force agreement. This is
a historical B1 anchor with same-protocol later B2/4/8 measurements, not four
simultaneous measurements. At B1 the ratios versus updated Ours are 115.9902
in time and 392.3898 in energy. CLIP/YOLO/DeepSeek now use the explicitly
retrained heads and new frozen costs documented above; their historical
timing-only profiles and scores remain protected but do not supply final points.

The following continuous-GPU sweep description and 81x/348x figures document
the previous audit and are not the current Qwen output. Original raw records
are kept, rather than rewritten to match the final table.

## Results and files

- `batch_cost_comparison.png/svg`: revised 16-video energy and processing time versus videos per call. No unmeasured accuracy trend is implied.
- `srcc_cost_comparison_reference.png/svg`: existing filename retained; all four electronic baselines now use real full558 scores at B. The LightGen-II physical score is separate from its component-cost proxy.
- `qwen_batch_metric_audit.png/svg`: actual SRCC recomputed from saved Qwen predictions. Full TEST and first-200 subset are separate panels; neither substitutes for the other.
- `LGVQ_temporal_multi_baseline_recalculated.xlsx`: original three-tab structure retained, with updated Ours inputs and explicit measured-at-batch SRCC columns. Original workbook is unchanged.
- `timing_energy_recalculated.csv`: all 16 baseline settings, unrounded calculation values, per-video amortized quantities and precision/source fields.
- `plot_data.csv`: 17 plot-ready rows (one optical N16 point and sixteen actual electronic B settings), explicitly joined by model and batch identity.
- `batch_metric_audit.csv`: separate historical references, actual full558/subset200 results, checkpoint origin/SHA and reference differences.
- `recalculation.json`: optical/electronic components, source SHA-256 records and calculation inputs.

Original measurements remain in the parent `raw_remote/`. The old builder
`../build_analysis.py` and its historical outputs are unchanged. Reproduction:

```powershell
python LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculate_batch_comparison.py
```

`../recalculate_batch_workbook.mjs` rebuilds the formula-driven workbook with
the Codex workspace `@oai/artifact-tool` Node dependency. If the dependency is
not on the usual module lookup path, set `CODEX_ARTIFACT_NODE_ROOT` to a
directory containing its `node_modules`. Run the Python derivation first.
Rebuilding requires the same local raw files and recovered evidence; these
data files are deliberately not included in a source-only Git commit.

## Calculation boundary

For a baseline batch of B videos, T16 = (16/B) times the mean synchronized-wall
batch time. E16 = (338.2 W + that batch's mean A100 board power) times T16 in
seconds. B = 1/2/4/8 requires 16/8/4/2 calls. The 200-video timing sample and
no-explicit-warm-up protocol are preserved; the first call remains included.
Current board power is the mean of telemetry tagged during the first-200
inference windows. Historical visual-model power used a separate sustained
pass; those old costs are no longer final points. Neither is a calibrated
per-call energy integral.

Updated LightGen-II: six optical cycles total 6.2682 ms; serial electronic
components total 3.9773695913702247 ms; sum 10.245569591370225 ms. Its energy is:

E16 = 0.0062682 * (80.358 + 41.388)
    + 0.0039773695913702247 * (41.388 + 62.842)
    = 1.1776895097085185 J.

Amortized energy is 0.0736055943567824 J/video, and task efficiency is
13.5859238518309 videos/J. The equivalent per-video service time is NOT
single-video call latency. Optical allocation remains 16 videos per field;
no LightGen-II batch-scaling performance is invented.

This uses `../../20260928_nine_task_a100_table/raw_remote/ours_narrow_200/report.json`.
Its profiler (`LightGenV2/scripts/profile_narrow_optical_electronics_a100.py`)
constructs architecture components and random inputs without loading the exact
formal PT. It is a narrow component-timing proxy, not exact-checkpoint or bench
end-to-end timing. Transfers, I/O, SLM encoding/layout, CCD crop/stack and
RMS/alpha fusion are excluded; parallel residuals are not added to the serial
critical path. Baselines use synchronized wall time, while Ours electronic
components use CUDA events. FP32 Ours and FP16/BF16 baselines are not equal
arithmetic precision; no claim of equal-precision measurement is made.

Qwen B1 in THIS sweep is 52.25563935935497 ms/video, giving 836.0902297496796 ms
and 409.8658550628422 J/16 videos. The other nine-task-table run was
74.2741341516 ms/video. Mixing just its B1 into this sweep would join different
measurements. With this sweep, Qwen ratios vs updated Ours are 81.605/51.449/
45.929/43.283 in time and 348.025/237.739/213.256/201.368 in energy for B1/2/4/8.

## Historical accuracy audit and batch theory

Original `build_analysis.py` repeats one formal full-TEST reference SRCC for
every batch; these are not independent accuracy measurements at each batch.
SRCC is a correlation coefficient, not classification accuracy or a percentage.

| Qwen videos/call | Full TEST, 558 videos | Same first 200 TEST videos |
|---:|---:|---:|
| 1 | 0.7663428036522781 | 0.659313938351 |
| 2 | 0.7664148425389119 | 0.666105630665 |
| 4 | 0.7694958345192725 | 0.667403965221 |
| 8 | 0.7694089803486667 | 0.667403965221 |

The same IDs, labels and selected frames are verified across the 200-video
settings. Full558 B1/B2 predictions are in
`LightGenV2/reports/20260914_latest_qwen_baselines_a100_FIRSTBLOCK_FINAL/evidence/T06_temporal_batch1_batch2/`.
All full558 IDs, labels and selected frames match. Existing server B1/B2/B4/B8
predictions and reports were recovered into `server_prediction_evidence/`;
its manifest gives exact source locations, file sizes and SHA-256. No inference
was run. Full558 B1/B2 differ by only 0.00007204 SRCC, but individual MOS predictions can
differ by up to 3.31146; this is not proof of negligible per-sample error. BF16,
padding, attention/kernel differences need a controlled check to attribute
the cause. No such cause is asserted here. The lower subset SRCC is a different
population, not a demonstrated regression of the full-test model.

The full558 B4/B8 runs are historical schema1 reports at Git `7d5b05e9`,
script SHA `19ec2e9500d5712aef1db98cd6866d37f5fd5e91321e724cce781bea9793dd4c`.
B1/B2 use schema2 with script SHA prefix `0ab945e1`. A read-only Git function
audit verified identical model loading/eval/dtype/SDPA and identical hidden-state,
last-valid-token pooling, native-LM projection and five-level MOS calculations.
Changes add timing events and preprocessing trace, not frame/crop/resize logic.
The metrics are usable together with this provenance; their timing fields are
NOT merged into the 200-video synchronized-wall sweep. The four full-test
scores span 0.00315303 SRCC, without a strictly monotonic trend.

For a fixed, sample-independent model in evaluation mode, batching does not
theoretically guarantee increasing or decreasing SRCC. Floating-point and
implementation differences can change results in either direction. BatchNorm
using batch statistics, active dropout, inconsistent padding/preprocessing or
cross-sample coupling can introduce larger changes and should be audited.
Increasing batch size often amortizes GPU overhead; per-batch latency and
fixed-workload throughput are distinct. Energy depends on both time and power
and need not change monotonically. The observed cost trends are measurements,
not constraints imposed by the recalculation.

CLIP/YOLO/DeepSeek timing profilers create random timing heads; they do not
load the trained head or save meaningful batch predictions. Their historical
full-TEST references are 0.7251568/0.7015822/0.5688169. Higher precision or actual
SRCC(B) cannot be reconstructed from their timing-only reports. Trained PTs,
cache identities and original run paths are documented by the T06 baseline
reports, not inferred from these timing runs.

LightGen-II reference simulation/physical SRCC is 0.8043868643/0.7977138739
for PT 5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c,
as recorded by `tasks/t06_video_quality_assessment/CURRENT_VERSION_20261004.md`.
It is not a performance result produced by the timing proxy.

## Historical missing-asset assessment (superseded by the reproduction above)

No rerun is required for the reported arithmetic update or recovered Qwen
full-test metric sweep. A fully measured multi-model SRCC-versus-batch graph
still requires CLIP/YOLO/DeepSeek B1/B2/B4/B8 using their frozen trained
checkpoints on the same 558 videos. Their existing trained prediction assets
were not found in the bounded local/server inventory; this does not establish
that they were deleted. Only missing settings need inference once the exact
trained PT and dataset contract are available.
If the exact original PTs were available, no retraining or optical capture
would have been necessary. They remained unlocated after the stronger bounded
archive search, so the user authorized the new original-protocol reproduction
documented at the top. No optical capture was performed.
Cached B1 features can verify a reference head score but cannot establish that
native B8 backbone inference yields identical features.

The original reports identify A100 but not all retain GPU UUID, power-limit
and process-audit evidence. DeepSeek B8 raw power has peaks above 300 W. This
does not by itself prove invalid data; independent physical-device/power-limit
identity should be recovered before a stronger calibrated-power claim.

The offline calculation itself does not use GPU/SDK. Qwen was reevaluated
with its original frozen head; CLIP/YOLO/DeepSeek heads were reproduced and
then independently frozen-evaluated on the specified A100. Source code was
Git-synced. Raw predictions/weights were not externally uploaded.
