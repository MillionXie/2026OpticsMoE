# LGVQ GPU utilization measurements

## CLIP B11 extension — 16 October 2026

The original 30-point/90-repeat cohort below is preserved. A new CLIP B11 point uses exactly the same native resident queue8 loop, frozen checkpoint, 4-frame inputs, 250 W A100 and three complete repeats. Means of 99.6806722689%, 99.3819095477% and 99.7231543624% pool to 99.5950782998% over 1,788 samples and 90.108401 s (14,680 forwards). The first new batch already exceeds 98%, so no B12–16 was run. This is a measured result, not a configured utilization target or best-window selection. B10 remains its original different-date observation of 94.89367655%.

All three workbooks and display CSVs append the new point; the original90 repeats remain unchanged and three B11 repeats are appended. `plot_data.xlsx` adds the separately measured full558 SRCC0.725202866664, T16=34.6894103289ms and E16=13.2737767190J. These costs use their own interleaved-run power estimate, not resident-load power. The original cost CSV/JSON remain the immutable30-point cohort; the extension's independently recomputed `verified_costs.json` is separate.

Qwen B2/4/8/10 were profiled for diagnosis only. Eight native forwards per batch contain148/206/322/380 CUDA stream synchronizations per forward and58/108/208/258 scalar-read operators per forward. Scalar-sized D2H copies correlate with native scalar/indexing operations even when inputs are already GPU resident. Profiler timing is intrusive and is not NVML utilization, SM occupancy or a replacement cost. One unprofiled repeat at each diagnostic B is retained only as a control. Formal Qwen values, including the63.133111480865225% historical B1 anchor, are unchanged. This demonstrates internal synchronization overhead, without assigning a causal percentage of the plateau or promising98% after optimization.

Raw reports, traces, identity/SHA records and independent audits reside in T06 `runs/simulation/lgvq_clip_saturation_qwen_plateau_a100_20261016/continuation01`. See the current comparison README for the numerical extension and manuscript paragraph.

## Finalized display selection — 16 October 2026

The user explicitly retains Qwen B=1 **63.133111480865225%** from
`unified_qwen_b1_5s30s01/report.json` as the presentation anchor. This historical
record uses one native resident queue8 repeat (5.060464 s warmup, 30.241928 s
measurement, 601 samples, 576 forwards). Other displayed batches use the uniform
three-repeat pooled means below. The basic resident/native queue8 method is the
same, but the historical B1 record is not a three-repeat or same-round estimate.

The newer Qwen B1 pooled **71.17536071032187%** is retained unchanged as an audit:
column Q of the utilization workbook and column K of plot_data.xlsx Measurements.
The repeated-measurement table below and raw repeats remain unchanged. Only the
selected display value, its matched historical sample counts/date/source, and
derived labels change; finalized latency, energy and SRCC do not change.
No new GPU run, graph, batch expansion or best-window selection was performed.

## Unified resident inference measurements

All six models completed the same native resident-inference protocol on 15 October
2026: **one execution mode, repeated three times per batch**. The earlier
three-mode experiments below are historical controls and are not pooled with this
round. The 30 model–batch points contain 90 repeats, 55,249 retained utilization
samples and 158,408 forwards over 2,786.797 measured seconds. Every complete repeat
is included. No CUDA Graph, maximum-window selection or retraining is used.

| Model | Video batch B | Repeat 1 (%) | Repeat 2 (%) | Repeat 3 (%) | Pooled utilization (%) | SD (pp) | Peak (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B | 1 | 71.41860 | 71.92000 | 70.18667 | 71.17536 | 0.89196 | 73 |
| Qwen3-VL-2B | 2 | 87.45714 | 87.28047 | 87.08472 | 87.27339 | 0.18629 | 89 |
| Qwen3-VL-2B | 3 | 89.24463 | 89.24544 | 89.31457 | 89.26821 | 0.04015 | 91 |
| Qwen3-VL-2B | 4 | 90.73632 | 91.06312 | 91.40667 | 91.06814 | 0.33521 | 93 |
| Qwen3-VL-2B | 6 | 91.90296 | 91.79639 | 90.46764 | 91.38420 | 0.79970 | 94 |
| Qwen3-VL-2B | 8 | 91.69672 | 91.63399 | 91.57259 | 91.63433 | 0.06206 | 96 |
| Qwen3-VL-2B | 10 | 90.03060 | 91.25325 | 90.60680 | 90.62857 | 0.61166 | 96 |
| GLM-4.6V-Flash | 1 | 97.66446 | 97.61765 | 97.77104 | 97.68360 | 0.07861 | 98 |
| GLM-4.6V-Flash | 2 | 98.09437 | 98.04290 | 98.06908 | 98.06876 | 0.02573 | 99 |
| DeepSeek-VL2-Small | 1 | 78.69040 | 78.09846 | 78.66770 | 78.48482 | 0.33539 | 83 |
| DeepSeek-VL2-Small | 2 | 90.88060 | 91.35426 | 91.76048 | 91.33134 | 0.44037 | 96 |
| DeepSeek-VL2-Small | 3 | 96.69878 | 96.77301 | 96.69878 | 96.72347 | 0.04286 | 99 |
| DeepSeek-VL2-Small | 4 | 97.32353 | 99.09687 | 99.08036 | 98.49530 | 1.01911 | 100 |
| CLIP ViT-B/32 | 1 | 38.55201 | 38.90772 | 39.34228 | 38.93400 | 0.39579 | 42 |
| CLIP ViT-B/32 | 2 | 42.39028 | 42.46901 | 42.34564 | 42.40168 | 0.06247 | 44 |
| CLIP ViT-B/32 | 3 | 48.45805 | 48.56951 | 48.80235 | 48.61006 | 0.17567 | 50 |
| CLIP ViT-B/32 | 4 | 53.32496 | 52.82915 | 53.03020 | 53.06145 | 0.24938 | 55 |
| CLIP ViT-B/32 | 6 | 68.00838 | 67.82550 | 67.68960 | 67.84125 | 0.15996 | 69 |
| CLIP ViT-B/32 | 8 | 75.45134 | 75.88926 | 76.44556 | 75.92901 | 0.49828 | 79 |
| CLIP ViT-B/32 | 9 | 89.36471 | 87.59228 | 89.46218 | 88.80571 | 1.05258 | 93 |
| CLIP ViT-B/32 | 10 | 95.72987 | 95.82689 | 93.12584 | 94.89368 | 1.53221 | 99 |
| YOLO11s | 1 | 42.37919 | 45.06208 | 45.92295 | 44.45556 | 1.84829 | 49 |
| YOLO11s | 2 | 73.19128 | 73.71477 | 73.64381 | 73.51676 | 0.28398 | 75 |
| YOLO11s | 3 | 98.96309 | 98.78655 | 97.54271 | 98.43009 | 0.77414 | 99 |
| DeepSeek-VL2-Tiny | 1 | 74.92545 | 74.19667 | 74.35667 | 74.49697 | 0.38302 | 82 |
| DeepSeek-VL2-Tiny | 2 | 84.30924 | 83.96315 | 84.07191 | 84.11453 | 0.17698 | 87 |
| DeepSeek-VL2-Tiny | 3 | 88.90563 | 88.76568 | 88.88119 | 88.85077 | 0.07475 | 95 |
| DeepSeek-VL2-Tiny | 4 | 92.14603 | 92.39557 | 92.36809 | 92.30343 | 0.13683 | 99 |
| DeepSeek-VL2-Tiny | 6 | 96.08974 | 96.15705 | 96.00480 | 96.08382 | 0.07630 | 100 |
| DeepSeek-VL2-Tiny | 8 | 97.81214 | 97.78499 | 97.74964 | 97.78221 | 0.03134 | 100 |

The 98% stopping criterion was first reached at GLM B2 (98.06876%), Small B4
(98.49530%) and YOLO B3 (98.43009%). Their immediately preceding tested batches
remain in the table. Tiny stopped at its predefined B8 cap with 97.78221%;
CLIP stopped at B10 with 94.89368%. Qwen stopped at B10 with 90.62857%; its B8
measurement was 91.63433%. No larger batches or repeat selection were added
after observing these results. Utilization need not increase strictly with batch.
These driver counters measure kernel-busy time, not SM occupancy or achieved
FLOPS relative to peak capacity. They do not establish that 98% is an overload limit.

The largest within-point spread among the three repeat means was 3.54375
percentage points (YOLO B1), below the predefined five-point warning threshold.
The reported SD is the sample standard deviation of the three repeat means,
not a confidence interval based on thousands of independent observations.
The clocks, temperature and CPU scheduling records are available for explaining
variability, but this experiment does not isolate a causal explanation for the
earlier Qwen B1 fluctuation. Its new three-repeat mean is 71.17536%; the finalized
historical 63.133111480865225% observation remains separate, not silently replaced.

### Common execution and aggregation

For every model, the first B TEST videos supply the original four frames per
video. CPU preparation and input transfer occur once per batch. The inputs then
remain on the GPU. One native forward processes B videos and produces B MOS
outputs on the GPU. Eight forwards execute serially on the same CUDA stream
before synchronization. This does not change the simultaneous batch to 8B.
No CUDA Graph, extra head, training, artificial load or precision change is used.
Native model-internal synchronization is retained. Finite-value audit checks for
GLM and Small are outside the timed loop; their native GPU arithmetic is unchanged.

Each batch is measured three times using exactly that same loop. Each repeat
has at least five seconds of actual warmup followed by a complete measured
window of at least 30 seconds. The final eight-forward group may extend the
window slightly beyond 30 seconds. All measured driver samples are retained,
including low values. No peak window or highest repeat is selected. The primary
mean is the sum of all three repeats' utilization counters divided by their
total retained sample count. The repeat means and their sample standard deviation
are also reported. A spread greater than five percentage points is flagged,
not removed or remeasured to obtain a higher value.

The A100-PCIE-40GB is restricted to UUID
`GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55` at its existing 250 W limit. Admission
requires no foreign compute process; process checks continue during execution.
Driver utilization, board power and memory are polled every 50 ms. A separate
200 ms record retains SM and memory clocks, temperature, P-state and power limit.
CPU runtime and scheduler-queue counters are recorded before and after the
measured window. These are raw observations for diagnosing variability, not
evidence that a particular cause of the earlier Qwen fluctuation is established.

The ascending batch candidates are Qwen 1/2/3/4/6/8/10, CLIP 1/2/3/4/6/8/9/10,
YOLO 1/2/3/4, Tiny 1/2/3/4/6/8, GLM 1/2/3/4/6 and Small 1/2/3/4. The first
B >= 2 with pooled utilization >= 98% ends that model's series; its preceding
measured point is retained. A native out-of-memory error instead records the
capacity boundary and ends the series without quantization, offload or fewer
frames. A driver counter cannot exceed 100%, and 98% does not define overload.

### Data records and manuscript use

The existing utilization workbook contains all 30 points. Its `Repeat measurements`
sheet and `LGVQ_temporal_gpu_utilization_repeats.csv` contain all 90 original repeats,
one row per observation. Fields are mean and peak utilization, actual measured
duration, counter sample count, forward count, same-window board power and memory,
SM clock, temperature, CPU queue-wait time, actual warmup duration, UTC+8 timestamp
and source path. Percentages remain numeric in Excel; the mean formulas use
the original counter sum divided by sample count. The summary pools all three
repeats' counters, rather than taking their highest value. Unrounded data remain
in the CSV and canonical JSON. The repeated input is the first B TEST videos,
not 558 distinct videos or a new full-dataset accuracy evaluation.

Each canonical repeat retains `telemetry.csv`, `clock_telemetry.csv`,
`forward_groups.json`, `process_audit.json` and `report.json`. The raw driver
and clock records have monotonic and wall-clock timestamps, so their windows
can be reconstructed. The same fields and aggregation apply to all six models.
All original model reports remain bound to the unchanged checkpoint, split,
input-tensor identities and native output checks. Small B1 was also checked
against its finalized native prediction; this is not an independent B>1 SRCC audit.
The dataset remains the original 2,250 TRAIN and 558 TEST split, with no training
or test-set checkpoint selection in this utilization experiment.

The canonical run is
`lgvq_utilization_execution_20261015/unified_repeat_suite01`. GLM has 35 original
files; the successful five-model continuation `remaining_native01` has 474 files,
including its command records and parent status. All 509 original files have
local per-file SHA256 backups. Independent CPU reconciliation binds 492 source
records and checks every repeat's timestamps, driver counters, eight-call groups,
process ownership, power limit and unchanged MOS outputs. The workbook has also
been recalculated, error-scanned, rendered and reimported to verify all 30 means
and 90 repeat records. The six formal cost/accuracy files remain SHA-identical.

An initial Qwen admission-order error failed before CUDA and produced no result.
The admission fix was published as main `f97048b64`; its measurement-loop AST is
identical to the original `5f73f4e71` GLM version. External YOLO activity at two
admission boundaries was excluded, not measured concurrently. After the A100
was verified idle, the unchanged continuation completed at 22:21:30, before
the original 22:55:30 deadline. The parent and final worker exited and no own
A100 compute process remained. Failed starts and historical measurements are
retained separately; they are not additional points in the table. The earlier
12-point workbook/CSV and the interim GLM-only publication are preserved under
the same canonical run's archive records.

Utilization is reported **independently** of the finalized SRCC, latency and energy.
Neither these resident forward timings nor their board-power observations replace
the published processing-cost protocol. Qwen B1 retains
1,188.3861464262009 ms and 462.1134010767563 J per 16 videos in the historical
cost table; the new utilization must not be described as a simultaneous
measurement of those historical costs.

### Manuscript description

GPU utilization was characterized on an NVIDIA A100-PCIE-40GB GPU configured
with a 250 W power limit. For each frozen model, the inputs from B fixed LGVQ
test videos, with four frames per video, were prepared once and retained in GPU
memory. Native forward inference was then repeated on one CUDA stream, retaining
outputs on the GPU and synchronizing after every eight serial calls. Each
forward processed B videos. Three identical repetitions were acquired per batch,
each comprising at least 5 s of warmup followed by at least 30 s of measurement.
The primary utilization statistic pooled all driver samples from the three complete
measurement windows. Individual repeat means, standard deviations, power,
memory use, clock frequencies and temperature were retained. Batch sizes were
increased according to predefined sequences until the pooled mean first reached
98% or the preset batch cap was reached. These GPU-resident workload measurements
were independent of the latency, energy and accuracy evaluations.

## Historical continuous inference measurements

All **12 new model–batch points** completed on 15 October 2026, using the same
three-phase continuous inference procedure on the NVIDIA A100-PCIE-40GB at
250 W. The table below supersedes the earlier five-second-warmup table for
independent utilization reporting. Every attempted point is included; earlier
measurements remain available as historical records.

| Model | Video batch B | Mean GPU utilization (%) | Peak GPU utilization (%) |
| --- | ---: | ---: | ---: |
| Qwen3-VL-2B | 1 | 55.32 | 57 |
| Qwen3-VL-2B | 8 | 91.97 | 96 |
| Qwen3-VL-2B | 10 | 92.61 | 96 |
| CLIP ViT-B/32 | 1 | 38.90 | 40 |
| CLIP ViT-B/32 | 9 | 89.39 | 93 |
| CLIP ViT-B/32 | 10 | 98.55 | 99 |
| YOLO11s | 1 | 45.28 | 47 |
| YOLO11s | 2 | 68.52 | 74 |
| YOLO11s | 3 | 99.54 | 100 |
| DeepSeek-VL2-Tiny | 1 | 74.95 | 81 |
| DeepSeek-VL2-Tiny | 4 | 90.64 | 98 |
| DeepSeek-VL2-Tiny | 8 | 96.73 | 100 |

Means and peaks both refer to the fixed seconds 5–30 of the third phase, a
25-second interval. A peak of 100% is not a mean of 100%. Qwen B8 reproduces
approximately 92% utilization without a batch of 16 or 24. CLIP B10 and YOLO B3
reach the predefined 98% reference. Qwen B10 and DeepSeek B8 do not; their
measured means are retained without expanding the batch search. Higher values
from an earlier run or another execution phase are not substituted.

### Measurement procedure

Each model–batch point runs in a fresh process with its previously frozen
checkpoint, native precision and quality head. The dataset manifest remains
the original 2,250 TRAIN and 558 TEST split. Only the first B TEST videos are
used for this utilization experiment, with four frames per video and unchanged
frame selection. Their inputs are prepared and transferred to the GPU once.
The same resident input is then reused, without repeating video decoding,
preprocessing or input transfer between forwards. This is not a new full-558 accuracy evaluation, and repeated
forwards are not additional distinct videos.

Every point executes three ordered 30-second phases:

1. Native synchronized inference with MOS returned to the CPU after each call.
2. Synchronized inference with the same numerical work and MOS retained on the GPU.
3. Native inference with GPU outputs, eight serial forwards on one CUDA stream,
   followed by a group synchronization. Timing hooks are removed for this phase.

Each forward processes B videos. The eight-call queue does not increase the
simultaneous video batch to 8B. There is a two-second settling interval before
each phase. The primary statistic, fixed in source before measurement, is the
third phase's seconds 5–30 mean. The full 0–30-second mean and both fixed windows
of the first two phases are also retained. No CUDA Graph, extra layer, training,
precision change or artificial background load is used. All models, including
Qwen B1, follow the same sequence. Unlike the earliest Qwen B1 control, this
round does not insert an additional 200-video cost audit before its three phases.

Utilization is polled every 50 ms. The reported mean is the arithmetic mean of
all driver counters whose monotonic timestamps fall inside the fixed interval,
including low and zero values. The 496–498 retained samples per point are polls,
not independent driver averaging windows. GPU utilization denotes kernel-busy
time, not SM occupancy or a fraction of peak FLOPS. The phases have a fixed order,
so differences between phases are not randomized estimates of a single cause.
Longer preceding inference did not uniformly increase all utilization values.

Initial higher batches were Qwen8, CLIP9, YOLO3 and DeepSeek4. A mean below 98%
allowed one predefined larger batch, respectively 10, 10, 4 and 8. A mean of at
least 99.5% triggered the adjacent lower batch if not already measured. This
produced the YOLO B2/B3 comparison; the counter itself cannot exceed 100%.
These thresholds describe observed kernel-busy utilization, not arithmetic
saturation or overload. All four B1 references were newly measured.

### Data and manuscript use

The existing [spreadsheet](LGVQ_temporal_gpu_utilization.xlsx) and
[CSV](LGVQ_temporal_gpu_utilization.csv) now contain these 12 points. They include
the fixed-window mean and peak, sample count, complete-window forward count,
measurement role, timestamp in UTC+8 and source location. Forward counts include
only entire eight-forward groups contained within seconds 5–30. Additional
columns retain the third phase's full mean and the first two phases' steady
means. Timestamps identify the last retained telemetry sample. Means are
calculated in Excel from the sum of raw counters divided by the retained sample
count; raw observations and unrounded values remain separate.

Utilization is measured **independently** of the previously finalized latency,
energy and SRCC. In particular, the historical Qwen B1 values of
1,188.3861464262009 ms and 462.1134010767563 J per 16 videos are unchanged.
The new 55.32% B1 utilization must not be described as a simultaneous observation
from that historical cost measurement. It also does not replace or rescale the
earlier 63.13% five-second-warmup result or the 67.88% earlier control; those used
different preceding execution sequences. All six formal cost/metric data files
remain SHA-identical to the previous protected versions.

The canonical run is `lgvq_utilization_execution_20261015/continuous_suite01`.
Its 121 original files were backed up locally with per-file SHA256 verification.
`verified_summary.json` independently reconciles all 36 phase call counts and
72 fixed telemetry windows, binds 97 source records, checks unchanged PT and
measurement-module identities, and confirms exact MOS agreement across phases.
Startup source was published main `85e6a99c7`; later unrelated HEAD changes did
not alter the measured modules. The preceding spreadsheet and CSV are archived
under `figure5_native_suite01/published_table_before_continuous.*` before in-place
replacement. All 13 owned parent/child PIDs were confirmed absent, and the A100
had no compute process at 16:37:14 +0800. The coordinated model task was notified
that it could resume; no utilization reservation or queued rerun remains.

Suggested manuscript wording: GPU utilization was evaluated independently of
latency and energy using frozen native inference on an NVIDIA A100 GPU with a
250-W power limit. For each video batch size, four-frame inputs from the first B
test videos were prepared and transferred once, then reused for three ordered
30-s resident-inference phases. We report the mean driver utilization over the
predefined seconds 5–30 of the final phase, which retained outputs on the GPU and
synchronized after every eight serial forwards on one CUDA stream. No CUDA Graph
was used. Mean utilization reached 98.55% for CLIP ViT-B/32 at B=10 and 99.54%
for YOLO11s at B=3. Qwen3-VL-2B achieved 91.97% at B=8 and 92.61% at B=10;
DeepSeek-VL2-Tiny achieved 96.73% at B=8. These values characterize sustained
kernel-busy duty rather than arithmetic occupancy or peak FLOPS.

## Historical five second warmup measurements

The unified protocol has produced **10 valid points: nine new measurements and
the retained Qwen B1 measurement**. DeepSeek's high-batch points remain pending:
the attempted B8 process was rejected before CUDA initialization when a GLM
process from another task entered the A100. The original failed suite and its
error are preserved, not relabelled as complete. The other task reported that its
running feature extraction has no safe pause/resume interface. No process was
killed, no concurrent utilization result was accepted, and no background retry
or GPU reservation remains active.

| Model | B | Mean GPU utilization (%) | Peak GPU utilization (%) | Measurement time (s) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B | 1 | 63.13 | 73 | 30.24 |
| Qwen3-VL-2B | 16 | 90.92 | 98 | 31.43 |
| Qwen3-VL-2B | 24 | 92.79 | 100 | 30.33 |
| CLIP ViT-B/32 | 1 | 39.46 | 41 | 30.04 |
| CLIP ViT-B/32 | 10 | 93.28 | 96 | 30.00 |
| CLIP ViT-B/32 | 11 | 99.64 | 100 | 30.01 |
| YOLO11s | 1 | 43.91 | 47 | 30.05 |
| YOLO11s | 3 | 97.31 | 99 | 30.00 |
| YOLO11s | 4 | 99.79 | 100 | 30.05 |
| DeepSeek-VL2-Tiny | 1 | 72.04 | 79 | 31.00 |

Qwen B1 was not remeasured. Its 14:52 measurement and all nine new measurements
use the same frozen PT per model, first B TEST videos, four frames per video,
native inference precision, one-time H2D, one stream, and synchronization after
eight serial forwards. Each forward still processes B videos. Approximately
5 s of actual inference warmup precedes a separate interval of at least 30 s.
Complete groups remain included, explaining durations slightly above 30 s. The
50 ms polling frequency is not the driver's internal utilization averaging
window. Means retain every sample inside the exact interval, including low or
zero counters. There is no CUDA Graph, training or new full-558 SRCC evaluation.

Candidate selection was fixed before measurement. Initial high batches were
Qwen16, CLIP10, YOLO4 and DeepSeek8; means below 98% trigger at most one larger
candidate (24, 11, 5 and 12, respectively). A mean of at least 99.5% triggers the
adjacent lower batch unless already measured. Thus CLIP10/11 and YOLO3/4 describe
the observed transition into near-ceiling kernel-busy utilization. These are
operational observation thresholds, not proof of arithmetic saturation or
overload. The driver counter cannot exceed 100%. Qwen24 reached 100% only at
its peak; its mean was 92.79%, below the 98% reference. No higher-utilization
historical value replaces this observation. All attempted points are retained.

[Spreadsheet](LGVQ_temporal_gpu_utilization.xlsx) and
[machine-readable table](LGVQ_temporal_gpu_utilization.csv) contain the measured
dates, roles, sample counts, forward counts and source locations. The workbook
computes each mean as the sum of retained raw driver counters divided by the
number of retained samples. The canonical run is
`lgvq_utilization_execution_20261015/figure5_native_suite01`; its
`verified_summary.json` independently reconciles all 50 source files underlying
the ten valid points, input frame counts, fixed PT/manifest identity, process
audits, invocation counts and native-output agreement. The original 66 files
from this suite, including the pre-admission failure, are SHA-backed up locally;
Qwen B1 retains its separate earlier backup. Actual source for the nine new
points was dispatched from main `ad3ddade171ceaea89b9b734fc5a06b4e20ffa92`.
Seven child processes recorded that HEAD; YOLO B3/B4 recorded the intervening
other-task commit `b3e66bfc2c908d9c6628036a30d1d9b9b0331f96`. The measured
module SHA identities stayed unchanged, including entry SHA
`9fa6a77c76f6959dbe6bad7f6ebe2b2aeda28b4761b0edaf56b22be578ec1ea5`.
The later bounded resume interface was tested but not launched because no exclusive window was
available. The six existing formal cost/metric data files remain SHA-identical
(`historical_cost_protection.json`). Utilization is independent of those
finalized latency, energy and SRCC values.

Suggested manuscript wording: GPU utilization was measured independently of
latency and energy using a common GPU-resident inference protocol. For each
batch size, the first B test videos, with four frames per video, were prepared
and transferred once and repeatedly processed by the frozen native model.
Following approximately 5 s of warmup, utilization was sampled over at least
30 s without CUDA Graph. CLIP ViT-B/32 showed mean utilization of 93.28% and
99.64% at B=10 and B=11, respectively; YOLO11s showed 97.31% and 99.79% at B=3
and B=4. Qwen3-VL-2B reached 92.79% mean utilization at B=24. These measurements
describe GPU kernel-busy duty, not SM occupancy or a percentage of peak FLOPS.
DeepSeek high-batch measurements are incomplete and are not included as final
high-load comparison points.

## User approved independent utilization measurement

The subsequently approved reporting protocol measures GPU utilization separately
from the finalized inference latency and energy. Its common execution method is
resident native inference without CUDA Graph: prepare the first B TEST videos
once, transfer their four-frame inputs once, retain MOS outputs on the GPU, and
repeat serial forwards on one stream. Synchronization follows every eight
invocations; each invocation still processes B videos, not 8B. Five seconds of
real inference warmup precede a separately timed measurement of at least 30 s.
Complete groups are retained, so actual durations can slightly exceed 30 s.
All samples in the exact recorded interval contribute to the average, including
low and zero values. The measurements under this protocol are listed above;
earlier multi-model controls below are historical, not additional new points
under this protocol. No model-specific Graph advantage is used in these results.

On 15 October 2026 at 14:52 +0800, **Qwen3-VL-2B, B=1** completed this measurement
on the NVIDIA A100-PCIE-40GB at its unchanged 250 W power limit. Mean GPU
utilization was **63.133111480865225%**, peak 73%, median 59%, and minimum 47%.
The exact post-warmup interval lasted 30.241928331553936 s and contained 601
telemetry samples and 576 forward invocations (72 groups). Warmup lasted
5.06046400219202 s; its 99 samples are retained separately, not included in the
reported mean. The resident input was the first TEST video, four frames at
positions 10, 35, 60 and 86, at the unchanged 448-pixel input resolution. This is
a repeated single-video utilization experiment, not 576 distinct videos or a
new full-dataset accuracy evaluation. Driver utilization is kernel-busy time,
not SM occupancy or percent of peak arithmetic throughput. Input residency
removes repeated preparation and H2D, but does not remove all native model
CPU dispatch or internal synchronization.

The original checkpoint SHA starts `e102adc6`, and the dataset manifest SHA
starts `607c50d2`. Both initial and final GPU-output MOS matched the original
native inference exactly (57.99810791015625, absolute difference zero). All 33
periodic process audits contained no foreign A100 compute process. The task
exited and released the A100; five original files were transferred to the same
local canonical run with per-file SHA verification. An independent calculation
from the raw telemetry timestamps reproduced the mean, peak, sample count and
576 forward invocations. Source main was `42a48af0e87c3ebd075c0d8fbdc34c7f0d87395c`,
with entry SHA `33d63dacf1dfe6411830769d233d452566465539cedbb795282732be2e46dab8`.

[Exact result and input identity](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/unified_qwen_b1_5s30s01/report.json),
[raw telemetry](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/unified_qwen_b1_5s30s01/telemetry.csv)
and [SHA transfer receipt](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/unified_qwen_b1_5s30s01/backup_receipt_unified_qwen_b1.json)
provide the retained evidence. The historical 1188.3861464262009 ms and
462.1134010767563 J per 16 videos remain unchanged. If shown beside this
utilization, they must be labelled as separate-protocol measurements, not
simultaneous observations. This approved separation supersedes the earlier
recommendation below to require every displayed utilization to share its cost
measurement run. The new B1 mean is retained even though it is below the earlier
approximately 68% controls; no highest-window or highest-run substitution is made.

Suggested manuscript wording: GPU utilization was evaluated independently under
a sustained resident-input protocol. For each batch size, the first B test
videos were prepared and transferred to the GPU once and repeatedly processed
using the frozen native model, without CUDA Graph. Following 5 s of inference
warmup, GPU utilization was sampled at 50 ms polling intervals over at least
30 s. For Qwen3-VL-2B at B=1, the mean and peak utilization were 63.13% and 73%,
respectively. These measurements characterize resident inference duty and are
reported separately from the dataset-once latency and energy measurements.

## Earlier execution controls

15 October 2026. The original Qwen B=8 resident mean of 89.22% includes a
startup transient. Its fixed seconds 1–15 mean is 91.05%. CPU preparation explains
substantial GPU waiting in the interleaved workload. The new controls below
separate this startup effect from execution changes. Performance, precision,
checkpoint, dataset and historical table values remain unchanged.

## What the existing measurements establish

The interleaved implementation prepares one batch on the CPU, copies it to the
GPU, executes inference, returns MOS values to the CPU, and then prepares the next
batch. In the original Qwen B=8 first-200 workload, preparation summed to
54.276083 s, while the selected first-vision-block-to-MOS boundaries summed to
6.405379 s. Preparation is excluded from the published inference latency, but
the GPU waits during that preparation. A driver utilization value queried during
a short inference can still average over preceding preparation. Phase labels do
not turn this device counter into an instantaneous per-kernel measurement.

The resident implementation prepares the first B distinct TEST videos once and
repeats the same model inference for approximately 15 s. It avoids repeated
decoding and H2D, but retains CUDA synchronization, timing hooks, and a small MOS
transfer back to the CPU after every call. The transferred MOS array is tiny.
The synchronization and launch cadence, rather than its byte volume alone, can
create idle gaps. Their separate causal contributions require a controlled test.

### Startup sensitivity of the original resident measurement

These are exploratory, fixed elapsed windows referenced to the first recorded
active telemetry sample. They do not recover the exact phase-transition time.
All original measurements remain intact. The original whole-active-period mean
is shown alongside the two subwindows, not replaced by whichever is higher.

| Model, B=8 | Original full mean (%) | First second mean (%) | Seconds 1–15 mean (%) | Seconds 5–15 mean (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B | 89.22 | 63.25 | 91.05 | 91.09 |
| CLIP ViT-B/32 | 71.30 | 58.65 | 72.21 | 72.49 |
| YOLO11s | 97.10 | 70.95 | 98.99 | 99.00 |
| DeepSeek-VL2-Tiny | 95.26 | 75.90 | 96.67 | 96.60 |

Qwen B=8's first resident inference boundary was 272.227 ms versus a mean
210.434 ms for subsequent calls. Removing the startup counter transient explains
why a steady observation can exceed 90% while the full 15-s mean is 89.22%.
It does not explain CLIP's entire remaining idle fraction, nor prove that every
model can reach 98% at the tested batch size. No monotonic utilization trend is
required as batch changes.

Source: canonical T06 run
`lgvq_utilization_execution_20261015/cpu_fixed01/historical_windows.json`,
including SHA256 bindings to the 76 unchanged source files in `full_remeasurement02`.

## Predeclared execution controls

The new entry `audit_utilization_execution` uses the existing frozen loaders,
PT identities, first B TEST videos, frame selection, precision and complete native
computation, including Qwen's native vocabulary projection. It performs no
training and supplies no new SRCC. Cases are fixed to Qwen B1/B8, CLIP B8/B10,
YOLO B3/B8 and DeepSeek B8. Each new process measures three fixed 30-s controls:

1. Original synchronized inference and CPU MOS output.
2. Same synchronized first-block boundary, retaining MOS on the GPU until the
   measurement ends. This isolates removal of the explicit per-call CPU return.
3. Same numerical work, retaining GPU outputs and synchronizing once per eight
   calls. Timing hooks are removed in this control. Its latency is **full-forward
   throughput**, not the historical synchronous first-block latency.

All raw counters are retained. Each control reports both its full 0–30 s mean
and a predeclared 5–30 s mean, regardless of whether the latter is higher.
The windows were fixed in source before new GPU results were collected.
Controls run in a fixed order, not randomized, so cold/warm and temporal drift
are limitations of effect-size comparisons. Qwen B1 first completes the separate
200-video audit below, so its subsequent resident controls are already warm.
These controls are not CUDA Graph capture, synthetic load, or a claim of maximum
SM occupancy. Internal model synchronization may remain even in the queued mode.

Only the specified A100 UUID at 250 W is permitted, with the shared task lock and
foreign-process guards. Interrupted evidence is excluded rather than combined
with a later successful attempt. The deadline is 12:45 +0800 to leave time for
verification and delivery before 13:00. The first attempt was interrupted during
the B1 audit by another evaluation process and is retained as `gpu_suite01`.
After both processes exited and repeated idle observations, recovery uses a
different output, `gpu_suite02`, under the same deadline and published source.

## Preserving the Qwen B1 reference

The historical total-table anchor remains 1188.386146 ms and 462.113401 J per
16 videos. It has no matched utilization record. New utilization must not be
attached to that historical cost as if measured in the same experiment.

The new Qwen B1 audit processes the same first 200 distinct TEST videos in the
original order, without explicit warmup, and retains the first call. Its timing
is the original first native visual block to GPU MOS synchronized wall boundary.
It records preparation and H2D separately, then computes

`T16 = sum(first200 boundary ms) × 16 / 200`

`E16 = (338.2 W host proxy + same-run active board mean W) × T16 / 1000`.

The audit is reported beside the historical reference, not forced to match it.
For resident and queued results, utilization, power and timing remain paired with
their own workload. A faster queued throughput cannot be relabelled as the old
batch-one latency. A higher steady utilization cannot be paired with the old
interleaved power measurement. If a new deployment method is adopted for the
paper, the changed protocol and its complete matched costs must be identified.

## Interpretation for manuscript writing

GPU utilization denotes the fraction of a driver sampling interval with one or
more kernels executing. It is not the fraction of peak FLOPS or SM occupancy.
Polling every 50 ms does not shorten the driver's internal sampling interval.
A fixed warm/steady measurement is appropriate for sustained throughput, while
the cold-inclusive first-200 protocol describes the existing inference-cost
measurement. Both are legitimate when named accurately and not combined.

Definitions: [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
Synchronization semantics: [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.6/notes/cuda.html).

## New experimental results

All seven `gpu_suite02` cases and their 21 controls completed on the fixed A100.
The table reports the predeclared seconds 5–30 mean, not a peak or the largest
observed window. The full 0–30 s mean, whole-phase mean, complete-loop timing,
energy and individual telemetry records remain available in the canonical run.

| Model | B | Original synchronized loop (%) | GPU output, same synchronization (%) | Eight-call queue (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B | 1 | 68.05 | 68.25 | 67.88 |
| Qwen3-VL-2B | 8 | 91.89 | 92.06 | 92.30 |
| CLIP ViT-B/32 | 8 | 73.47 | 73.96 | 76.09 |
| CLIP ViT-B/32 | 10 | 96.59 | 97.26 | 98.70 |
| YOLO11s | 3 | 80.30 | 98.04 | 91.19 |
| YOLO11s | 8 | 99.00 | 99.00 | 100.00 |
| DeepSeek-VL2-Tiny | 8 | 97.64 | 97.74 | 97.83 |

All retained outputs matched the original loop exactly for the measured batch.
No new full-dataset accuracy evaluation was performed. For Qwen, removing the
small MOS return alone produced only a small utilization change. The larger
YOLO B3 change was not monotonic across the three controls. Because controls
were ordered rather than randomized and repeated, it is not an isolated causal
estimate of CPU-copy cost. No model is guaranteed to reach 98% at an arbitrary
batch, nor does 100% kernel-busy imply peak FLOPS.

The new Qwen B1 first-200 audit was 1245.912659 ms and 481.923808 J per 16 videos,
with same-run active-tagged utilization 14.44%. The latency is 4.8407% above the
historical 1188.386146 ms anchor. This measured difference is retained; the old
anchor is not overwritten or rescaled. All six original measurement/table files
were independently SHA-verified unchanged. The new resident Qwen B1 loop is
approximately 784.53 ms per 16-video equivalent, illustrating why its utilization
and costs cannot be presented as matched measurements of the old interleaved
1188.39 ms workload.

`comparison_verified.json` independently reconciles the 21 whole-phase raw
counter series and call counts, first-200 timing/energy formulas, PT/source
identities, output agreement and historical six-file protection. The source
report's fixed subwindow means are retained separately from the recomputed
whole-phase means. Queue times include the complete native forward and final
group synchronization, not the original per-call first-block host boundary.

The [complete derived results and verification record](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/comparison_verified.json)
and [startup-window calculations with source SHA256](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/cpu_fixed01/historical_windows.json)
remain in the canonical task run. They include unrounded values; displayed
decimal places do not imply that the driver counters have that precision.

## CUDA Graph feasibility control

A second, separately declared experiment attempts one native full-forward
capture for Qwen B8, CLIP B10 and YOLO B3. It uses three real side-stream warmup
calls, one capture attempt and a fixed 30-s replay period. All fixed-input and
second-distinct-batch MOS values must match native computation within 0.0002.
Failed/unsupported capture is retained and not patched into another model.
This tests real CPU launch-overhead reduction, not artificial background load.

Capturing/replaying supported static work can reduce Python and driver dispatch
overhead, but operations that synchronize the CPU, such as `.item()`, are not
capture-safe. See [PyTorch CUDA Graph requirements](https://docs.pytorch.org/docs/2.6/notes/cuda.html#cuda-graphs).
These results are independent throughput controls. They cannot supply a new
utilization value for the historical B1 cost or replace complete 200-video
dataset-once measurement. New-input validation occurs outside timing; repeated
resident-input replay does not include the cost of changing videos.

Both supported captures completed and independently passed raw-timestamp window,
power, timing, call-count and two-distinct-input output verification:

| Model | B | Entire replay phase mean U (%) | Seconds 5–30 mean U (%) | Full-forward throughput time per 16 videos (ms) | Energy per 16-video equivalent (J) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP ViT-B/32 | 10 | 98.9431 | 100.00 | 8.4437 | 4.9362 |
| YOLO11s | 3 | 99.2642 | 100.00 | 53.1200 | 31.0523 |

The entire-phase time and energy are paired with the entire-phase counters, not
with a higher selected utilization window. The energy includes the same 338.2 W
host proxy. Both original and second distinct batches matched native MOS exactly.
These are measured throughput equivalents for repeated resident-input work,
excluding warmup, capture, preprocessing and video replacement. They are not new
200-distinct-video dataset-once costs or new full-558 performance results.

For Qwen B8, native full-forward capture failed in
`visual.fast_pos_embed_interpolate`, at the `torch.linspace(..., h)` call with
`operation not permitted when stream is capturing`. The later capture-end error
is secondary. No model patch, retry or substituted success is reported. The
validated approximately 92% ordinary resident result remains the applicable
outcome for unmodified Qwen.

All owned measurement processes exited and the A100 was released. The 98 original
success/failure, command, log, report and telemetry files were SHA-backed up in the
same local canonical run. `backup_receipt_execution_all.json` records the scope.

## Recommended reporting

For the existing paper table, retain its cold-inclusive inference-cost protocol
and historical Qwen B1 reference. Describe the lower interleaved counter values
as execution-duty measurements affected by CPU preparation and driver averaging.
Do not attach the approximately 92% resident value to the historical 1188.39 ms
cost as a newly matched observation.

For a sustained-load figure, use a common, predeclared warm/steady protocol for
all models, with both cold-inclusive and steady means retained. Qwen B8 is
reproducibly above 90% without changing precision or model computation. For
deployment-optimized comparisons, CUDA Graph is a demonstrated option for the
tested CLIP B10 and YOLO B3 workloads, not a universal switch for Qwen. Adopting
it as a final paper cost requires complete, consistently measured distinct-video
workloads and corresponding data-transfer costs. High utilization alone is not
evidence of better energy efficiency.
