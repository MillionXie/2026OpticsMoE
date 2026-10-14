# LGVQ batch comparison: frozen reevaluation and recalculation, 2026-10-13

## 2026-10-14：论文用表、计时边界和图形复核

本次只复核存档并修正展示，未启动训练、推理测速或光路采集。所有正式
SRCC、时间和能耗数值保持不变；旧图表已先可恢复备份到既有 T06 canonical
运行目录的 `plot_presentation_before_20261014.zip`。原始历史工作簿、预测、
功率采样和逐调用计时均不改写。

### 表一：Summary（可直接用于论文/画图）

LGVQ 固定划分共 2,808 个不同视频，其中 TRAIN 2,250、TEST 558。四个电子
baseline 分别在 B=1、2、4、8 下冻结推理，同一模型每个 B 均对完整 558 个
TEST 视频产生独立预测并重新计算 SRCC；总共 16 组设置、8,928 次模型—视频
评估，不是 8,928 个不同视频。主列 C 现在是每组实际 full558 SRCC，列 I 的
历史参考分数及列 J 的 first200 SRCC 不用于主图。成本取同一 TEST 顺序的前
200 个不同视频，按 16 视频工作量折算；并非仅测 16 个视频。B=1/2/4/8 对应
200/100/50/25 次成本测量调用、16/8/4/2 次折算调用，每个视频输入 4 帧。
LightGen-II 是唯一 N=16 视频/光场点，物理参考 SRCC 为 0.7977138739，沿用
总表成本 10.245569591370225 ms、1.1776895097085185 J（表中四舍五入
10.246 ms、1.178 J）；没有虚构其 GPU B=1/2/4/8 曲线。Qwen B=1 保留总表
锚点：SRCC 0.7663428036522781、1188.3861464262009 ms、462.1134010767563 J。
`plot_data.csv` 将模型、B/N、实际 full558 SRCC、16 视频时间和能耗合并为
17 行未提前四舍五入的绘图数据，附样本量、计算精度和来源口径。

### 表二：Batch results（逐 batch 时间与能耗的计算审计）

每个电子模型/B 设置在指定 A100 的独立新进程运行，模型仅加载一次；逐
batch 顺序执行 CPU 解码/裁剪/预处理、H2D、冻结前向、输出 MOS，非先准备
完全部视频、也无预处理和推理流水重叠。计时起点是第一个视觉 Transformer
block 的 forward-pre-hook（Qwen、CLIP、DeepSeek）；YOLO 对应第一个卷积
block，不能称 TF。终点是 GPU 上五级质量读出计算完成、MOS 就绪并等待 CUDA
同步之后；不含随后预测下载到 CPU，也不含模型加载、解码、预处理和 H2D。
无显式 warm-up，首次调用保留。T16=(16/B)×前 200 视频的平均 batch 同步
墙钟时间，E16=(338.2 W 主机代理功率+该 B 实测 active A100 平均功率)×T16/1000。
完整 558 TEST 在 B=1/2/4/8 下实际运行 558/279/140/70 次，B=4/8 尾批分别为
2/6 个真实视频，不补充虚假样本。计时钩子内部未再次 CUDA 同步，因此这是
实际实现的“首 block 入口至 GPU 分数就绪”的同步墙钟边界，不能声称严格
剔除了所有钩子前已排队的 embedding GPU 工作。功率为 50 ms 采样、整段
forward 活跃标签窗口的均值；与上述边界时间相乘是估算，不是每个短调用的
校准能量积分。U 列是实际 full558 SRCC，C 列仅历史参考。

### 表三：Sources（来源、精度与适用范围）

每个 B 的 SRCC 均由对应 `predictions.csv` 独立重算，已核验 558 个样本 ID、
标签，以及同一模型不同 B 的选帧一致；各模型间 ID/标签相同。CLIP 和 YOLO
的实际 SRCC 跨 B 最大变化分别约 0.00001699、0.00000660，近水平曲线不是
复制分数；DeepSeek 为 0.5680463058–0.5703507558，Qwen 为
0.7663428037–0.7694958345。不要求 batching 导致性能单调变化。Qwen 使用原
冻结质量行；原 CLIP/YOLO/DeepSeek PT 未找回，当前为原协议重训的质量头，
最高 TEST 选模应注明 development/no VAL。Qwen 新 B=1 审计成本为
1150.4916036128998 ms、442.9714733293941 J，仅保留记录，不进入最终数据；
B=1 历史锚点和后续 B=2/4/8 同协议复测不是同日测量。LightGen-II 的时间是
既有六次光学周期 6.2682 ms 加串行电子部件 3.9773695913702247 ms 的估算，
不是新实测全链路；电子部件采用 FP32 CUDA event，baseline 为 FP16/BF16
同步墙钟，并非相同数值精度/相同计时器。CSV 与 JSON 保留源文件、PT 身份、
SHA、功率限制和异常原始采样。相关限制须随论文结果说明，不应将均值乘
功率估算表述为逐视频直接能量实测。

### 图形坐标与使用说明

- `srcc_cost_comparison_reference.png/svg`：纵轴为 SRCC 系数、范围 0–1，
  不是 Accuracy (%)。能量横轴对数范围 0.7–3000 J，时间横轴对数范围
  7–5000 ms，端点均明确标注，覆盖所有数据与标注。B=1/2/4/8 文字通过
  同色细引线绑定各点；N=16 仅标 LightGen-II 光场容量，不冒充 GPU batch。
- `batch_cost_comparison.png/svg`：横轴为每次 GPU 调用/光场的视频数；电子
  曲线仅有 B=1/2/4/8，N=16 光学点独立。纵轴均为 16 视频折算成本。
- `qwen_batch_metric_audit.png/svg`：完整 558 TEST 与前 200 子集分开显示，
  均用完整 0–1 SRCC 纵轴，标出真实数值，不能混用两个样本总体。

三个工作表均附一段英文表注，可复制给论文同学。`timing_energy_recalculated.csv`
是 16 组电子设置的计算明细；`batch_metric_audit.csv` 是实际精度与历史参考的
分离审计；画主图优先用已合并的 `plot_data.csv`，避免按行号误拼两份表。

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
