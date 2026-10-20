# LGVQ temporal quality: frozen-model accuracy, processing time and estimated energy

## Current measurement definition: 20 October 2026, new two-GPU rental

The latest author instruction supersedes the timing boundary of the historical
sections below. New raw measurements are under
`lgvq_strict_tf_rental_dual_20261020/attempt02`; the initial deployment failure is
preserved separately and produced no formal cost observations. Only full
DeepSeek-VL2 FP32 uses two A100-SXM4-80GB GPUs. Kimi in either precision and
DeepSeek-VL2 in its original precision use one GPU. No model is trained in this run.
The rental measurements are complete: 16 conditions and all 48 trials passed
independent raw-data replay. Historical Qwen B1 is unchanged. The original laboratory
measurements retain the two completed CLIP MPS conditions in `recovery_1102`.
Its failed temporal-control condition remains excluded. The subsequent
`calibration_1140` completed CLIP temporal control; `duration_1155` has completed
all sixteen deployed-precision laboratory conditions and is continuing FP32.
At this partial update, sixteen laboratory conditions have been independently
audited. Laboratory FP32 results remain unavailable until completed and audited.

The existing eight-view workbook now contains the newly audited interval costs.
Its previous complete-wall version, SHA256
`0fb5b3efe68afb45672ebd0161c471be3a798c6879675fdc6e502a839b35912b`,
is preserved as `prior_workbook_<SHA256>.xlsx` in the new run's `presentation`
directory. Do not mix its older complete-wall observations with the new interval
observations. Both rental GPUs were verified at zero allocated memory, zero
busy time and no compute processes after completion. This is GPU release, not a
claim that the virtual machine was powered off or billing was stopped.

### Completed rental data

The table reports all three trials pooled. Time and energy are normalized to 16
video evaluations. Busy percentages are full-wall NVML observations; dual-GPU
values are the maximum pooled card value, not the allocation average. Both cards
are retained in the raw repeat table. Unrestricted execution does not guarantee
98–100% busy time. MPS 50/100 denotes resources, not a prescribed NVML percentage.

| Model | Precision | Control | Setting | Busy (%) | T16 (ms) | Estimated E16 (J) | SRCC reference |
|---|---|---|---|---:|---:|---:|---:|
| DeepSeek-VL2 | FP32, two GPUs | MPS | 100% resources | 58.399 | 29198.089 | 21286.742 | .707578 |
| DeepSeek-VL2 | FP32, two GPUs | MPS | 50% resources | 58.697 | 53905.052 | 33473.702 | .707578 |
| DeepSeek-VL2 | FP32, two GPUs | Temporal | Approximately 50% busy | 49.920 | 34130.786 | 23447.365 | .707578 |
| DeepSeek-VL2 | FP32, two GPUs | Temporal | Unrestricted | 57.953 | 29156.146 | 21260.820 | .707578 |
| DeepSeek-VL2 | Deployed BF16 | MPS | 100% resources | 86.618 | 8270.773 | 4868.513 | .716735 |
| DeepSeek-VL2 | Deployed BF16 | MPS | 50% resources | 96.202 | 12014.841 | 6497.005 | .716735 |
| DeepSeek-VL2 | Deployed BF16 | Temporal | Approximately 50% busy | 47.966 | 14175.597 | 7269.084 | .716584 |
| DeepSeek-VL2 | Deployed BF16 | Temporal | Unrestricted | 96.618 | 7104.585 | 4458.756 | .716584 |
| Kimi-VL-A3B | FP32 | MPS | 100% resources | 99.370 | 20168.059 | 11822.791 | .751930 |
| Kimi-VL-A3B | FP32 | MPS | 50% resources | 99.394 | 29073.473 | 15768.805 | .751930 |
| Kimi-VL-A3B | FP32 | Temporal | Approximately 50% busy | 49.363 | 40337.936 | 19896.130 | .751930 |
| Kimi-VL-A3B | FP32 | Temporal | Unrestricted | 99.338 | 19830.525 | 11702.072 | .751930 |
| Kimi-VL-A3B | Deployed BF16 | MPS | 100% resources | 73.498 | 3641.567 | 1850.738 | .805465 |
| Kimi-VL-A3B | Deployed BF16 | MPS | 50% resources | 81.701 | 4202.430 | 2102.708 | .805465 |
| Kimi-VL-A3B | Deployed BF16 | Temporal | Approximately 50% busy | 50.073 | 3519.439 | 1711.998 | .809021 |
| Kimi-VL-A3B | Deployed BF16 | Temporal | Unrestricted | 98.650 | 1340.097 | 822.472 | .809021 |

MPS pairs use B1. Temporal pairs use DeepSeek BF16 B2, Kimi BF16 B6, and B1
for both FP32 profiles. Each pair shares batch, weights, dtype and placement.
SRCC references at B1 and higher batches were independently evaluated, not newly
trained or measured separately at each cap. FP32 does not improve either model's
SRCC here; its backend and weight expansion are disclosed below.

Machine-readable observations and all 96 currently qualified trial records
(48 rental, 48 laboratory) are in the new run's `presentation/paired_measurements.csv`
and `paired_repeats.csv`. Each row binds report, checkpoint, accuracy source, date
and hashes. All 270 rental files (23,917,130 bytes) were imported from a SHA-bound
final archive. Eight figure panels use `strict_tf_<method>_<precision>_*` in this
original report directory; missing conditions are not interpolated.

### Laboratory partial results: deployed precision

All values below pool the three complete trials. MPS pairs use B1; temporal pairs
use fixed B10 for CLIP, B3 for YOLO and Qwen, and B2 for GLM. These batches are retained in the data,
not substituted into figure labels. The same charged-interval definition and
matched power integration are used for every row. The unrestricted CLIP point
does not reach 98% busy time and is not labelled as near-100% saturation.

| Model | Control | Setting | Busy (%) | T16 (ms) | Estimated E16 (J) | SRCC reference |
|---|---|---|---:|---:|---:|---:|
| CLIP ViT-B/32 | MPS | 100% resources | 38.961 | 93.601 | 37.730 | .725256 |
| CLIP ViT-B/32 | MPS | 50% resources | 38.443 | 97.621 | 39.773 | .725256 |
| CLIP ViT-B/32 | Temporal | Unrestricted | 81.456 | 9.112 | 5.326 | .725239 |
| CLIP ViT-B/32 | Temporal | Approximately 50% busy | 49.697 | 15.702 | 8.198 | .725239 |
| YOLO11s | MPS | 100% resources | 43.875 | 180.055 | 83.726 | .701597 |
| YOLO11s | MPS | 50% resources | 59.857 | 176.761 | 83.257 | .701597 |
| YOLO11s | Temporal | Unrestricted | 95.652 | 58.594 | 34.327 | .701594 |
| YOLO11s | Temporal | Approximately 50% busy | 51.281 | 106.908 | 54.965 | .701594 |
| Qwen3-VL-2B | MPS | 100% resources | 66.411 | 762.344 | 413.741 | .766343 |
| Qwen3-VL-2B | MPS | 50% resources | 85.413 | 855.036 | 464.361 | .766343 |
| Qwen3-VL-2B | Temporal | Unrestricted | 96.354 | 415.542 | 243.490 | .768899 |
| Qwen3-VL-2B | Temporal | Approximately 50% busy | 50.441 | 784.970 | 398.982 | .768899 |
| GLM-4.6V-Flash | MPS | 100% resources | 97.480 | 1780.846 | 1039.894 | .809262 |
| GLM-4.6V-Flash | MPS | 50% resources | 97.999 | 2571.622 | 1503.626 | .809262 |
| GLM-4.6V-Flash | Temporal | Unrestricted | 98.084 | 1630.915 | 955.118 | .810310 |
| GLM-4.6V-Flash | Temporal | Approximately 50% busy | 48.547 | 3358.155 | 1647.443 | .810310 |

MPS resource provisioning and NVML busy time are different measured quantities.
The values above are retained rather than adjusted to an expected trend. No
causal speedup is inferred from the small YOLO MPS latency difference. These are new
strict interval measurements, not the historical Qwen B1 finalized observation.

### Methods draft: timing boundary and inter-inference intervals

For each forward pass i, we recorded the model-invocation time a_i, the first
native computation-block input time s_i and the synchronized GPU-output time e_i.
For Transformer models, the hook is on the visual encoder's first Transformer
block: `vision.blocks.0` for DeepSeek-VL2, `vision_tower.encoder.blocks.0` for Kimi,
`model.visual.blocks.0` for Qwen and GLM, and `visual.transformer.resblocks.0` for
CLIP. It is not on the language decoder's first block. For YOLO, the analogous
boundary is the first native convolutional block, `model.0`; it is not described
as a Transformer. Inputs
were prepared and transferred to the GPU before measurement and reused for frozen
inference. Before recording s_i, all GPUs used by that model were synchronized to
drain GPU work preceding the first Transformer block. The output timestamp e_i
was recorded after the final GPU quality score had completed on all used GPUs.
The charged duration of a trial was

`t = sum_i(e_i - s_i) + sum_{i=1}^{m-1}(a_{i+1} - e_i)`.

Thus, each invocation-to-first-block interval was excluded, whereas the gap from
each output to the next model invocation was included. Temporal pacing waits
were already contained in these gaps and were not added a second time. Warm-up,
intervals between independent trials and any idle tail after the last output
were excluded. We additionally retained output-to-next-first-block gaps and the
continuous first-block-to-last-output span as audit quantities, not as interchangeable
cost definitions. Here “next input” means entry into the model with prepared
GPU-resident inputs; this protocol does not measure new-video decoding or host-to-device
transfer. Eight consecutive forwards each had their own input/output synchronization;
they were not an eightfold larger concurrent video batch.

Three complete trials were retained for each condition. Temporal-control pairs
used the same batch size within each model/precision profile, with calibrated
idle intervals enabled only for the restricted condition. Calibration was frozen
before the three trials. MPS pairs used batch size one with nominal active-thread
limits of 100% or 50%, set before CUDA-context creation. The exposed multiprocessor
counts were audited separately from measured GPU busy time. MPS provisioning is
not a target of 50% NVIDIA Management Library (NVML) busy time.

The laboratory continuation adds a bounded pre-measurement validation of the
pacing calibration. After the original 60-s short-feedback calibration, up to
three complete instrumented rehearsal windows target 50 ± 2 percentage points.
The duty is then frozen for all three formal trials, whose acceptance criterion
remains 50 ± 5 percentage points. Rehearsal calls, waits and raw telemetry are
retained separately and do not enter formal costs. No formal trial is selected
or automatically repeated. A failed calibration or trial remains in its original
run. The first laboratory attempt's power-support failure and the subsequent
out-of-band temporal trial are not silently replaced or promoted.

Every retained trial has at least 30 charged seconds. In `duration_1155`, the loop
additionally requires at least 35 actual charged seconds, rather than relying
only on a loop count extrapolated from cold warm-up. The bound is tested against
actual first-block intervals and internal waits, with a hard limit of 100,000
groups. This stopping-rule correction does not add frontend time or an idle tail
to a trial. The preceding short YOLO trial remains excluded, while completed
CLIP trials are reused in place with their original report and identity hashes.
Calibration version and required duration are retained in the workbook and raw
processed-data records. The older rental execution is not relabelled as using
the later laboratory calibration/stopping implementation.

### Methods draft: energy, normalization and device allocation

For each trial, sampled board power was integrated over the same union of
first-block-to-output intervals and output-to-next-invocation gaps used for time.
The integral used zero-order-held raw NVML power readings; a support gap exceeding
250 ms invalidated the measurement. The energy estimate was
`E = 338.2*t + sum_g integral_charged_intervals P_g(u) du`, with time in seconds,
power in watts and energy in joules. The sum includes every GPU used by the model;
the retained 338.2-W host proxy is added once, not once per GPU. This is an
interval-matched estimated system energy, not newly measured host wall power or
a high-bandwidth per-kernel power measurement. All three trials contributed to
`T16 = 16000*sum(t)/sum(n)` and `E16 = 16*sum(E)/sum(n)`, where n counts completed
video evaluations rather than distinct videos. These are 16-video throughput
equivalents, not a claim that sixteen distinct videos were submitted together.

The dual-GPU DeepSeek-VL2 FP32 configuration uses module-level model placement:
decoder blocks 0–14 are on GPU0 and blocks 15–29 on GPU1, with complete blocks
preserved. There is no CPU/disk offload or quantization. Both GPUs' board energies
are included. This capacity-oriented layer placement is not tensor parallelism,
and unrestricted execution does not establish simultaneous near-100% busy time
on both GPUs. Both per-card busy rates and their allocation average are retained;
the dual-GPU temporal controller uses the maximum pooled per-card busy rate.

### Methods draft: utilization and accuracy provenance

GPU busy time was calculated from all NVML observations within each full trial
wall-clock window, without removing frontend intervals, selecting favorable
subwindows or selecting the highest repeat. It is distinct from the non-contiguous
charged-time intervals used for cost. A 50-ms polling interval does not turn the
driver's averaged busy measurement into instantaneous SM occupancy. Three-trial
pooled utilization is sample-count weighted. Resource limits, actual busy rates,
precision, GPU count, timestamps and source hashes are retained together.

Accuracy references comprise 558 distinct videos from the original fixed TEST
partition, with the original four frames, central 65% crop, head weights and
model assets. Full DeepSeek-VL2 FP32 receives a new 558-video evaluation; the other
profiles reference individually identified frozen evaluations at the matching
batch and precision. These are not newly measured accuracy values at each
resource cap. Original head selection used highest TEST SRCC without a validation
partition, so the reported correlations are development estimates. Casting
deployment weights to FP32 does not recover information lost in their original
lower-precision representation. The FP32 profile uses its separately disclosed
official eager-attention runtime, rather than being described as an otherwise
identical attention-kernel experiment.

### 中文留底及术语

本次主时间为逐次首TF→输出，加输出→下一模型调用入口；每次调用入口→首TF排除并
单列，限流等待已包含在间隔内，不重复相加。独立三轮之间的时间不计入。
GPU busy（GPU忙碌率）指完整墙钟NVML实测；MPS resource cap（MPS资源上限）指执行
资源配置；charged time（计入时间）指上述区间之和，三者不得混为同一统计量。
租赁16条件均已完成并重算，只有完整版DS FP32双卡；其他租赁条件单卡。
实验室目前只有CLIP两项MPS条件合格。时间限流三轮48.05/43.73/42.35%，
合并44.5867%，未到45%下限，失败原始记录保留，不进入合格50%图表。
此前399ms功率采样缺口另属一次失败，250ms是既定数据质量阈值，非硬件或期刊标准。
实验室FP32尚无新的完整558视频结果，不从原精度或租赁平台借用数值。

## Historical timing audit hold: 20 October 2026, 08:00

The author clarified that image-to-first-block work must be excluded on **every**
forward, while internal pacing waits must be included. Code inspection found that
the complete-window measurements below exclude the frontend only at the first
forward of the trial; frontend work on subsequent forwards is inside that window.
The 12 conditions in the archived complete-wall workbook remain valid
records of that older complete-wall protocol, **not results of the newly clarified
per-forward timing boundary**. Do not cite them as such or silently subtract an
unmeasured frontend estimate. All raw data and historical Qwen B1 remain protected.

The laboratory entry now uses the separately identified
`per_forward_first_block_plus_internal_wait_v1` profile under its existing run.
Every forward drains frontend GPU work before its first-block timestamp, then
synchronizes its final GPU MOS output. Actual internal pacing waits are added;
pre-block work, warm-up and an idle tail after the final output are excluded.
Board energy is estimated by integrating the original power readings over exactly
those charged intervals, with the same 338.2-W host proxy. The full-wall NVML busy
rate is retained separately, without masking frontend samples to inflate it.
Individual synchronization changes execution scheduling relative to queue8;
previous busy rates and calibration duties therefore cannot be copied. This does
not change model weights or numerical precision. No new GPU result has yet been
produced under this profile: the laboratory A100 was externally occupied at 07:57
(92% busy, 11,419 MiB). The following Methods and numbers describe the prior protocol.

## Historical 20 October 2026 complete-window resource-control comparison

The revised experiment separates temporal duty control from GPU execution-resource
provisioning. The archived complete-wall workbook contains eight views: the two controls, original deployed
precision or FP32, and processing time or estimated energy. Raw repeat inputs and
recalculating normalization formulas are retained in the same workbook. Missing
measurements remain unavailable rather than being filled with historical pure-cost
values. This release is still partial: all eight native-precision DeepSeek/Kimi
conditions and four Kimi FP32 cost conditions are complete and locally SHA-verified.
The Kimi FP32 full558 accuracy evaluation is also complete and independently
recomputed locally; its development SRCC is 0.7519302260. The original 40GB A100
remained occupied by another user throughout this window. CPU-only admission expired
without any GPU measurement; the other four models remain unavailable under the new
matched-window protocol. The full DeepSeek FP32
weights alone require 109,920,536,992 bytes, exceeding the single 80GB device. The
author has decided not to add a two-GPU rental. Historical Qwen B1 is unchanged.

### Methods draft: temporal duty control

To measure the cost of reduced GPU busy time without introducing CPU input-preparation
gaps, we prepared the first B videos from the fixed LGVQ TEST partition and transferred
their original four-frame inputs to the GPU once. These inputs were reused, but each
forward pass performed the frozen model computation and produced a new GPU quality
score. Frames were sampled at normalized positions 0.10, 0.37, 0.63 and 0.90 using
the original central 65% crop. The80GB profiles retained384×384-pixel inputs for
DeepSeek-VL2 and448×448-pixel inputs for Kimi. Eight forwards were issued sequentially on the same CUDA stream before
synchronization. Each forward processed B videos, not 8B. Batch size was fixed between
the restricted and unrestricted conditions; the original-precision 80GB experiments
used B=2 for DeepSeek-VL2 and B=6 for Kimi-VL-A3B. A previously calibrated duty setting
controlled idle intervals between completed inference groups. The unrestricted
condition used the same model, precision, inputs and batch size without imposed idle
intervals. Three complete measurement windows were retained for each condition,
following at least 5 s of warm-up per window. Each formal window lasted at least 30 s.
Timing began when the first native Transformer block received its input and ended
when the final GPU quality output completed and synchronized. All internal imposed
waits were included; no wait after the final output was included. Actual GPU busy
time and board power were sampled within this same window. Calibration and warm-up
were excluded from the reported costs. Target acceptance was 50±5 percentage points;
no repeat or favorable subwindow was selected.

### Methods draft: GPU execution-resource provisioning

We separately evaluated execution-resource restriction using NVIDIA Multi-Process
Service (MPS). Batch size was fixed at one video for both the nominal 100% and 50%
active-thread limits. Each limit was set before the client CUDA context was created.
The 100% and 50% settings exposed 108 and 54 multiprocessors, respectively, through the
CUDA device attribute on the 80GB A100. This attribute verifies the resource limit;
it does not establish a dedicated physical partition. No temporal idle intervals
were imposed in this comparison. The resident-input protocol, first-block-to-final-output
boundary, three complete repeats and same-window power measurement were otherwise
retained. Measured GPU busy time was reported independently of the nominal resource
limit. A 50% MPS setting was therefore not labelled as 50% GPU busy time. MPS does not
halve all memory-system resources, so neither busy time nor processing time is
assumed to scale linearly with its active-thread limit. A private MPS control
directory was used without changing global compute mode, MIG configuration, GPU
power limit or clocks. Only the experiment's own client and server were permitted
on the target GPU during a valid measurement.

### Normalization, power, accuracy and precision

For formal window r with duration t_r in seconds and n_r completed video evaluations,
we calculated T16=16,000×sum(t_r)/sum(n_r) in milliseconds and
E16=16×sum[(338.2+P_r)×t_r]/sum(n_r) in joules, where P_r is the mean board power in
watts sampled within that same complete window. All three windows contributed to
the sums. The 338.2-W host term is the retained power proxy, not a new wall-power
measurement; E16 is consequently an estimated system energy. The first B videos are
reused, so n_r counts video evaluations rather than unique videos. These 16-video
equivalents describe normalized steady-state processing time and energy, not a
single batch of 16 videos or CPU-inclusive service latency. Loading, video decoding,
preprocessing and host-to-device transfer occur outside the resident measurement.
GPU busy time is the sampled fraction of time with kernel execution, not achieved
FLOPS or streaming-multiprocessor occupancy. These costs characterize the specified
fixed resident inputs, not the latency distribution across all 558 distinct videos.

Cost windows do not constitute a new 558-video accuracy test. Any displayed temporal
SRCC is explicitly bound to an independent frozen full 558-video evaluation at the
same batch and precision. Native-precision SRCC is not copied into FP32 results.
The original 2,250-training/558-test split contains 2,808 videos in total. No weights
were trained in the present measurements. The frozen heads were originally selected
by maximum TEST SRCC without a validation partition; reported SRCC values are therefore
development estimates, not selection-independent generalization estimates. Here,
independent evaluation means execution separate from cost windows, not a new test set.
FP32 uses FP32 floating-point parameters and inputs with autocast and TF32 disabled.
For Kimi, the official eager-attention backend replaces FlashAttention2 because the
latter does not support this FP32 execution. This comparison therefore changes both
dtype and attention backend, and is not interpreted as a dtype-only causal effect.
Casting previously deployed rounded weights to FP32 does not restore discarded
weight precision. The pending original-platform FP32 profiles also retain explicit
execution differences: Qwen uses native SDPA rather than the audited BF16 B3 execution
adapter, and CLIP/YOLO omit their FP16 feature-storage round trip. Those profiles must
not be described as changing only a tensor dtype. The 80GB A100-SXM4 at 400 W and original 40GB A100-PCIE at 250 W are
identified separately. Optical approximately 50% spatial use is retained only as an
optical reference, not asserted to be equivalent to either GPU control. The two
GPU controls are sensitivity analyses with different interventions, not interchangeable
definitions of fair optical-versus-electronic resource utilization.

### Audited native-precision 80GB data

| Model | Control | Setting | Batch | Actual GPU busy (%) | Time /16 (ms) | Estimated energy /16 (J) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek-VL2 | Temporal duty | Unrestricted | 2 | 96.553112 | 7143.255698 | 4597.936589 |
| DeepSeek-VL2 | Temporal duty | Approximately50% busy | 2 | 49.884321 | 13834.573833 | 7298.126373 |
| Kimi-VL-A3B | Temporal duty | Unrestricted | 6 | 98.662351 | 2252.162317 | 1313.407062 |
| Kimi-VL-A3B | Temporal duty | Approximately50% busy | 6 | 49.591940 | 4447.047202 | 2202.869899 |
| DeepSeek-VL2 | MPS | 100% resource cap | 1 | 87.903719 | 8201.859339 | 4923.280878 |
| DeepSeek-VL2 | MPS | 50% resource cap | 1 | 96.066944 | 12064.261718 | 6693.214902 |
| Kimi-VL-A3B | MPS | 100% resource cap | 1 | 74.430678 | 4535.837685 | 2367.904791 |
| Kimi-VL-A3B | MPS | 50% resource cap | 1 | 83.346545 | 5057.918594 | 2594.947504 |

### Audited Kimi FP32 cost data

| Control | Setting | Batch | Actual GPU busy (%) | Time /16 (ms) | Estimated energy /16 (J) |
| --- | --- | ---: | ---: | ---: | ---: |
| Temporal duty | Unrestricted | 1 | 99.243275 | 21166.494751 | 12654.723130 |
| Temporal duty | Approximately50% busy | 1 | 49.826226 | 42013.489035 | 21135.254283 |
| MPS | 100% resource cap | 1 | 99.210962 | 21180.647301 | 12653.536935 |
| MPS | 50% resource cap | 1 | 99.421402 | 30196.481787 | 16721.370793 |

The FP32 unrestricted-duty and100%-MPS arms use the same B1 workload and runtime.
Their pooled times differ by approximately0.067%. This is a consistency check for
this workload, not a universal claim that MPS has zero overhead. The nominal50%
MPS setting still shows99.42% time busy and must not be relabelled50% time busy.

### Terminology and evidence for author review

| Term | Meaning used in this release | Distinction retained |
| --- | --- | --- |
| GPU busy time | Driver-sampled kernel-busy percentage | Not SM occupancy or resource fraction |
| MPS resource cap | Nominal CUDA active-thread execution-resource limit | Not a target nvidia-smi percentage |
| T16 and E16 | Throughput-normalized cost per16 video evaluations | Not16 different videos in one forward |
| SRCC reference | Independent frozen full558-video accuracy | Not measured by repeating the cost inputs |

Evidence is retained in the canonical `lgvq_busy_resource_20261020` run, per-point
identity/report files, all three raw windows, telemetry, group timestamps, process
audits and SHA-verified transfer receipts. The table generator rejects duplicate
point identities and recomputes time, energy, utilization and terminal-wait boundaries
from raw data. The Methods draft describes directly implemented operations; it makes
no claim that either GPU restriction reproduces optical spatial utilization. Pending
40GB measurements remain missing evidence, not estimated results. The independent
FP32 reference is not a separate full558 test under each MPS cap. Kimi FP32 SRCC
0.7519302260 is lower than the frozen native B1 reference 0.8054652719; the data do
not support a claim that FP32 improves this head's predictive performance. No
checkpoint was retuned or replaced in response to that result.

## Earlier 20 October handover: interpretation hold on the old50%-busy cost comparison

The author has now confirmed that imposed throttling waits must be included in the
cost of a restricted workload. The18October pure-inference costs below are preserved
historical measurements, **not matched-window50%-busy service costs**. Do not use
their near-equal paced/native times to claim that temporal throttling has no cost.
The displayed busy percentages came from a separate GPU-resident repeating workload;
the cost runs interleaved distinct-video CPU preparation and measured only each
first-block-to-GPU-output interval, explicitly excluding the inserted waits.

Offline raw-time reconciliation found, for Qwen at the same B3 setting,534.142060ms
pure inference plus537.136698ms imposed waiting per16 normalized video evaluations,
or1071.278759ms combined. The corresponding unrestricted pure reference is526.553994ms.
The combined audit still excludes decode/preprocessing/H2D gaps and includes all
recurrent cycle waits: it is not a new measured end-to-end service latency or energy.
Waiting energy must not be obtained by multiplying pure active board power by this
longer time. QwenB1 historical finalized data remain unchanged and are not part of
the new50%-versus-unrestricted experiment.

Full raw reconciliation is retained under the canonical
`lgvq_rental_precision_20261019/presentation/busy50_boundary_audit.json` andCSV.
Further pure-cost queue stages were held while the author selected temporal duty
control versus MPS resource provisioning. The ongoing independent FP32 utilization
stage was allowed to finish; resource partitioning had not yet been enabled at that
handover. The earlier40GB laboratory wait expired without measurements. The latest
two-method authorization and renewed CPU-only idle guard are described above.

NVIDIA defines GPU utilization as the fraction of a sampling period with a kernel
executing, not the fraction of physical compute units in use
([nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)).
Temporal duty control preserves the resources of each inference and inserts idle
intervals; MPS active-thread provisioning instead limits execution resources
available to a CUDA client and need not reduce the busy-time counter to50%
([NVIDIA MPS provisioning](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)).
Neither is automatically equivalent to50% optical spatial use. Within a model,
batch size should remain fixed between the restricted and unrestricted conditions;
changing both batch and resource control would confound their effects. Actual
busy time must be reported rather than adjusted by batch scanning to force a label.

Everything below this notice describes the18October historical pure-cost release,
not the revised publication-ready50% comparison.

## Historical compact figure sets — inference boundary of 18 October 2026

The two compact plotting workbooks and four figures now use the same dataset-once inference-cost definition. The former resident-service costs, including the9.177-ms CLIP value, are not used in either current set. Ten prescribed cost points completed: six CLIP native repeats and four paced-setting points. Two interrupted partial runs are retained but excluded. All139 original remote files and launch controls were SHA-verified locally. No additional training, precision change, frame reduction, CUDA Graph or power-limit change was made. Finalized Qwen B1 remains63.133111480865225% utilization,1188.3861464262009ms and462.1134010767563J.

### Methods for both figure sets

We evaluated temporal quality assessment on the fixed558-video LGVQ TEST partition. The frozen quality heads and original four-frame input protocol were retained. Each new point evaluated all558 distinct videos; inference costs were calculated from the first200 videos in the same fixed order. Batches were split at the200-video cost boundary and558-video test boundary, retaining partial batches rather than dropping or padding them. Each point used a fresh process, with no explicit model-forward warmup; its first inference was included. Timing started at the input to the first native backbone block, or its equality-audited equivalent for optimized Qwen B3, and ended when the GPU MOS output had completed and synchronized. YOLO used its first convolutional block. CPU video decoding, cropping, preprocessing, host-to-device transfer, model loading and inter-inference pacing waits were recorded separately and excluded. These are inference-boundary costs, not CPU-inclusive end-to-end service times.

For synchronized inference times t_i in milliseconds covering exactly200 videos, T16=(sum t_i)×16/200. Estimated E16=(338.2W host-power proxy+same-cost-run mean active board power)×T16/1000. Board power came from numeric50-ms telemetry samples tagged active during the corresponding first200 inference calls. Neither independent resident-utilization power nor the GPU power limit was substituted. The338.2-W term is a retained host proxy, not a new wall-power measurement. CLIP B1 and B10 each had three prescribed cold-process repeats. Their displayed times and energies are arithmetic means of all three paired estimates; energy was not reconstructed by multiplying the mean power by the mean time. Each repeat used the same200 unique cost videos, so three repeats represent600 evaluations, not600 different videos.

GPU utilization was measured independently to reduce CPU input-preparation gaps. The first B TEST videos were prepared and transferred once, then reused in native resident inference, with eight sequential forwards on the same stream before synchronization. Each call still processed B videos, not8B. The pre-existing three complete repeats used at least5s warmup and at least30s measurement each, pooling all50-ms samples rather than selecting a favorable interval. Finalized Qwen B1 retains its separately identified historical reference. Approximately50% labels refer to these independent resident measurements. Their frozen duty settings were applied between calls in the new distinct-video cost runs, but those cost runs need not themselves average50% GPU busy because their input-preparation gaps differ. Driver-window kernel-busy time is not SM occupancy or achieved FLOPS.

Qwen, GLM, CLIP and YOLO costs use the original A100-PCIE-40GB at250W. DeepSeek-VL2 full and Kimi retain their locally re-audited first200 costs on disclosed A100-SXM4-80GB/400W sessions and were not remeasured. Their retained conventional timing rows, telemetry and558-video identities satisfy the same arithmetic boundary, but they are not a common-hardware causal comparison. No compliant paced dataset-once costs exist for those two models; the corresponding cells remain unavailable. Optical approximately50% denotes author-estimated spatial use, not an equivalent GPU time-busy constraint. The optical component-cost reference remains unchanged.

### Figure description: batch1 and near-saturation

Temporal-quality SRCC is plotted against inference latency and estimated energy normalized to16 videos. Each electronic model is represented by its B1 reference and one predeclared higher-batch configuration. Percent labels show independently measured GPU kernel-busy time only; batch sizes remain in the source tables. A model keeps the same color and symbol across its points, with connectors retained. The Qwen dashed connector spans finalized native B1 and equivalence-audited optimized B3, while the DeepSeek dashed connector spans different physical80GB GPU sessions. These links are descriptive rather than batch-only effects. CLIP uses all three new native cost repeats. The latency and energy panels share identical accuracy and utilization sources.

| Model | Condition | Video batch | Temporal SRCC | Independent GPU busy (%) | Inference time /16 (ms) | Estimated energy /16 (J) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGen-II | Optical reference | n.a. | 0.797714 | ~50 spatial (not GPU busy) | 10.245570 | 1.177690 |
| Qwen3-VL-2B | Finalized B1 | 1 | 0.766343 | 63.133111 | 1188.386146 | 462.113401 |
| Qwen3-VL-2B | Native reference | 3 | 0.768899 | 97.135509 | 526.553994 | 209.058386 |
| GLM-4.6V-Flash | B1 reference | 1 | 0.809262 | 96.881933 | 1810.626785 | 805.188484 |
| GLM-4.6V-Flash | Native reference | 2 | 0.810310 | 98.035320 | 1631.116714 | 775.784796 |
| CLIP ViT-B/32 | B1 reference | 1 | 0.725256 | 39.591620 | 171.119088 | 64.656916 |
| CLIP ViT-B/32 | Native reference | 10 | 0.725239 | 98.613878 | 33.753106 | 12.740406 |
| YOLO11s | B1 reference | 1 | 0.701597 | 44.997767 | 367.670324 | 140.508523 |
| YOLO11s | Native reference | 3 | 0.701594 | 98.653609 | 151.266861 | 57.283617 |
| DeepSeek-VL2 | B1 reference | 1 | 0.716735 | 88.049155 | 8360.056700 | 4727.746492 |
| DeepSeek-VL2 | Native reference | 2 | 0.716584 | 96.539183 | 7189.600891 | 4372.864338 |
| Kimi-VL-A3B-Instruct | B1 reference | 1 | 0.805465 | 73.987439 | 3745.677059 | 1803.738445 |
| Kimi-VL-A3B-Instruct | Native reference | 6 | 0.809021 | 98.392966 | 1380.524297 | 734.780588 |

### Figure description: approximately50% setting and near-saturation

The second pair of panels uses the same inference boundary as the first pair. Open symbols identify paced settings; filled symbols identify native references. Every native endpoint uses exactly the same cost values as the corresponding endpoint in the first set. All four new paced-setting points have558-video SRCC evaluations and first200 paired costs. Their imposed waits are excluded from formal time and energy and listed separately below. Consequently, the panels do not show the complete service-time or service-energy penalty of throttling, nor a controlled causal effect of utilization. Similar or reversed pure-inference costs, such as CLIP31.061844ms versus the native three-repeat mean33.753106ms, must not be interpreted as evidence that throttling accelerates the model. Cold-first-call and run-to-run variation remain in this no-warmup protocol. DeepSeek full and Kimi show only their valid native endpoints; missing paced costs are not replaced by resident-service estimates.

| Model | Condition | Video batch | Temporal SRCC | Independent GPU busy (%) | Inference time /16 (ms) | Estimated energy /16 (J) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGen-II | Optical reference | n.a. | 0.797714 | ~50 spatial (not GPU busy) | 10.245570 | 1.177690 |
| Qwen3-VL-2B | Paced setting | 3 | 0.768899 | 46.345564 | 534.142060 | 213.350874 |
| Qwen3-VL-2B | Native reference | 3 | 0.768899 | 97.135509 | 526.553994 | 209.058386 |
| GLM-4.6V-Flash | Paced setting | 2 | 0.810310 | 53.236225 | 1631.684928 | 782.776978 |
| GLM-4.6V-Flash | Native reference | 2 | 0.810310 | 98.035320 | 1631.116714 | 775.784796 |
| CLIP ViT-B/32 | Paced setting | 10 | 0.725239 | 49.771684 | 31.061844 | 11.744483 |
| CLIP ViT-B/32 | Native reference | 10 | 0.725239 | 98.613878 | 33.753106 | 12.740406 |
| YOLO11s | Paced setting | 3 | 0.701594 | 49.619632 | 157.906160 | 59.726234 |
| YOLO11s | Native reference | 3 | 0.701594 | 98.653609 | 151.266861 | 57.283617 |
| DeepSeek-VL2 | Paced cost unavailable | 2 | Not measured | Not measured | Not measured | Not measured |
| DeepSeek-VL2 | Native reference | 2 | 0.716584 | 96.539183 | 7189.600891 | 4372.864338 |
| Kimi-VL-A3B-Instruct | Paced cost unavailable | 6 | Not measured | Not measured | Not measured | Not measured |
| Kimi-VL-A3B-Instruct | Native reference | 6 | 0.809021 | 98.392966 | 1380.524297 | 734.780588 |

### CLIP power check and all native repeats

| CLIP condition / repeat | Time /16 (ms) | Estimated energy /16 (J) | Sampled active board power (W) | Active samples |
| --- | ---: | ---: | ---: | ---: |
| b01_r01 | 171.096227 | 64.610640 | 39.427500 | 44 |
| b01_r02 | 171.251271 | 64.704672 | 39.634694 | 49 |
| b01_r03 | 171.009766 | 64.655437 | 39.880377 | 53 |
| b10_r01 | 34.692538 | 13.106067 | 39.577692 | 13 |
| b10_r02 | 29.769459 | 11.231099 | 39.069167 | 12 |
| b10_r03 | 36.797320 | 13.884053 | 39.111538 | 13 |

CLIP B1 active means were39.4275–39.8804W with44–53 tagged samples per repeat. B10 means were39.0692–39.5777W with only12–13 samples, and the paced B10 mean was39.9W with12 samples. These readings are close to recorded idle values. Reproducible sampled readings do not establish accurate instantaneous compute power: short calls and driver averaging windows limit attribution. The resulting energy values remain estimates under the author-approved formula, not calibrated board-energy or whole-system energy integrals. The higher continuous-resident board power is retained in its own history and is not mixed into these cost rows. First-call times and raw sample counts are exposed in both workbooks.

### Waiting excluded from formal costs

| Paced setting | Pure inference /16 (ms) | Excluded waiting /16 (ms) | Estimated pure-inference energy /16 (J) |
| --- | ---: | ---: | ---: |
| qwen b03_r01 | 534.142060 | 537.136698 | 213.350874 |
| glm b02_r01 | 1631.684928 | 1375.071713 | 782.776978 |
| clip b10_r01 | 31.061844 | 25.244689 | 11.744483 |
| yolo b03_r01 | 157.906160 | 135.049286 | 59.726234 |

Waiting above is the measured interval following first200 inference calls, normalized by16/200. It is not included in T16 or E16. Full558 waiting and per-call requested/actual intervals remain in the raw JSON and workbook measurements. Native requested pacing is zero; tiny recorded native intervals are bookkeeping overhead, not imposed idle cycles.

### Files and terminology

- First set: [plotting Excel](LGVQ_batch1_near_saturation_plot_data.xlsx), [source CSV](LGVQ_batch1_near_saturation_plot_data.csv), [latency](batch1_near_saturation_latency.svg), [energy](batch1_near_saturation_energy.svg), [paired preview](batch1_near_saturation_pair.png).
- Second set: [plotting Excel](LGVQ_busy50_near_saturation_plot_data.xlsx), [source CSV](LGVQ_busy50_near_saturation_plot_data.csv), [latency](busy50_near_saturation_latency.svg), [energy](busy50_near_saturation_energy.svg), [paired preview](busy50_near_saturation_pair.png).

Each panel is78×88mm with600-dpi PNG plus SVG/PDF, giving156mm for a two-panel Word row. Labels avoid markers, model connectors and other annotation leaders; legends are checked against axis titles and notes. The actual cost positions are not jittered to separate close endpoints. No DS Tiny/Small or rental Qwen control is shown.

| Canonical term | Definition used here |
| --- | --- |
| Inference latency | First-backbone-block input to synchronized GPU MOS output |
| Estimated inference energy | Retained338.2-W host proxy plus same-cost-run sampled active board mean, multiplied by inference time |
| Independent GPU utilization | Driver-window kernel-busy fraction in the separate resident protocol |
| Spatial use | Author-estimated optical spatial occupancy, not GPU time-busy |

Reproduction uses the T06 run `lgvq_unified_inference_recheck_20261018`: original seven complete points, GLM under `resume03_20261018`, and final CLIP/YOLO under `resume02_20261018`. Failed GLM34-video and CLIP30-video attempts remain excluded. Source, split, frozen-head and raw-data SHAs, all-repeat arithmetic, waiting audit, formula checks and figure geometry receipts are retained there. Local CRLF versus server LF differences were checked against the exact measured Git blobs, without changing runtime sources. Earlier expanded comparison files and resident-service records below are historical and were not overwritten.

## Historical expanded comparison — 16–18 October 2026

### Supplementary platforms and clean Kimi follow-up (18 October 2026)

The paper comparison retains all 15 original displayed rows and four supplementary points: DeepSeek-VL2 B=1/2 and Kimi-VL-A3B-Instruct B=1/6. The rental Qwen B=1 control is excluded from the paper tables and figures at the author's request; its raw data and historical repeats remain available for audit. Hollow symbols identify A100-SXM4-80GB /400W, whereas the original GPU cohort uses A100-PCIE-40GB /250W. These are descriptive deployment comparisons, not a common-hardware model ranking. Energy uses recorded active board power, never the nominal power limit.

Kimi B=1 and B=6 were measured on the same physical GPU, GPU-ecac2c49-306a-9ec5-1572-8fcd2d232165, with unchanged frozen head SHA256 e6bf55b3b7ff318d37c973b67c4a6ce1062e4760ac91c2bdc748628988022586. Each point has all 558 TEST predictions and its own first-200 timing/power samples. New B=1 predictions match all 558 historical B=1 predictions exactly. The previous Kimi B=1/B=2 session is superseded for display, not deleted. DeepSeek full B=1/B=2 retains its historical distinct-card sessions (GPU-780326a6-79bc-7a45-36f6-683c5b278bee and GPU-d06c95d0-7066-9768-0f6e-fc1ee877f9f9); its dashed connector is descriptive, not a strict same-card batch ablation. The 338.2-W host proxy is retained and is not a measured rental-host wall-power value.

**Independent native utilization.** One Kimi model process was loaded once and measured sequentially at B=1 through B=6. Inputs retain four frames and the deployed precision. Each point has three identical complete queue8 repeats (at least 5 s warmup and 30 s measurement); all 50-ms driver samples are pooled. Each forward processes B videos, not 8B. B=6 is the first measured point strictly above 98%. B=2–5 have utilization-only evidence, not new full558 accuracy or dataset-once costs. No intermediate accuracy or cost is interpolated.

| B | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) | Full558 accuracy/cost |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 73.98686371 | 74.05901639 | 73.91666667 | 73.98743856 | Measured |
| 2 | 85.90572391 | 85.68581081 | 85.89597315 | 85.82940516 | Not measured |
| 3 | 91.94607843 | 91.98522167 | 91.89542484 | 91.94217130 | Not measured |
| 4 | 95.82450832 | 95.67170953 | 95.75114504 | 95.74911482 | Not measured |
| 5 | 97.95976155 | 97.94378698 | 97.73481481 | 97.87932740 | Not measured |
| 6 | 98.33129771 | 98.43226788 | 98.41538462 | 98.39296636 | Measured |

**Supplementary paced resident service: completed original 40GB suite.** We measured the six frozen original-platform models at their predeclared batch pairs on A100-PCIE-40GB at 250 W: Qwen B=1/3, GLM B=1/2, DeepSeek-VL2-Small B=1/3, CLIP B=1/10, YOLO B=1/3 and DeepSeek-VL2-Tiny B=1/8. Kimi B=1/6 retains the separate same-card A100-SXM4-80GB/400W measurements. On the original 40GB platform, a fresh process was used for each model–batch point, retaining one loaded model through calibration and all three repeats. At each point, only the first B TEST videos, each retaining the original four frames, were prepared and transferred once. Subsequent forwards reused these resident inputs and each processed B videos. Queue8 denotes eight sequential forwards before synchronization, not 8B simultaneously processed videos.

The original 40GB controller calibrated on complete compute-plus-imposed-idle groups for at least 30 s, then froze the duty fraction for all three repeats. Each repeat included at least 5 s of warmup and a complete measurement window of at least 30 s. All 50-ms driver samples in the three complete windows were pooled. The target was 50%, with a tolerance of ±5 percentage points of kernel-busy time, not SM occupancy or optical spatial occupancy. The target was not reached at CLIP ViT-B/32 B=1 (39.20%). These observations are retained without rounding up or adding unrelated GPU work. CLIP B=1 reached duty1 with zero imposed idle time, so its lower busy fraction is a recorded native-execution outcome, not a50% observation. Kimi used its separately disclosed last-completed-group feedback controller rather than the original 40GB aggregation across groups since the preceding update. No weights, numerical precision, frames, power limits or clocks were changed.

For a model batch B, normalized service T16 equals 16000×the sum of complete measured durations in seconds divided by B×the total completed forward calls. Mean board power is calculated within each complete window and duration-weighted across repeats. Estimated service E16=(338.2 W host-power proxy+same-window board mean)×service T16/1000, with T16 in milliseconds. Every imposed wait is included. These sampled-power service estimates are not calibrated whole-system energy-meter integrals. Repeating the first B inputs is not an evaluation of 558 distinct videos: the displayed full558 SRCC is linked to the separately measured conventional point, and paced runs check repeated-input MOS parity with an absolute MOS tolerance of 0.0002. Different input populations and warmup boundaries mean these resident service costs must not replace, or be interpreted as a causal slowdown ratio against, the first-200 dataset-once inference costs. The finalized Qwen B=1 conventional reference remains unchanged.

The first four 40GB points were preserved with 45 bound evidence hashes. After explicit authorization, only the remaining eight were measured in attempt03_remaining_20261018 using the published telemetry repair. The repair retains raw stdout, rejects unavailable numeric fields and saves available evidence before propagating failure; it does not impute or skip missing observations. All 12 original points passed independent raw-window, waiting-time and cost replay. The failed Small attempt remains excluded and archived. Full-precision service data and per-point source paths are retained in [LGVQ_paced_service.csv](LGVQ_paced_service.csv), alongside the updated Paced service worksheet.

| Model | B | Pooled GPU busy (%) | Repeat1 /2 /3 (%) | Service T16 (ms) | Estimated service E16 (J) | Same-window board mean (W) | Platform |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Qwen3-VL-2B | 1 | 47.615257 | 47.501672 / 47.470297 / 47.872727 | 1110.800360 | 556.365482 | 162.669015 | A100-PCIE-40GB /250W |
| Qwen3-VL-2B | 3 | 46.345564 | 46.352846 / 45.591973 / 47.073409 | 870.135809 | 434.314149 | 160.933749 | A100-PCIE-40GB /250W |
| GLM-4.6V-Flash | 1 | 53.221480 | 52.442211 / 54.342237 / 52.878536 | 3379.293990 | 1710.364761 | 167.930797 | A100-PCIE-40GB /250W |
| GLM-4.6V-Flash | 2 | 53.236225 | 53.850329 / 52.437908 / 53.424144 | 3084.170063 | 1557.531845 | 166.808418 | A100-PCIE-40GB /250W |
| DeepSeek-VL2-Small | 1 | 49.221130 | 49.425185 / 49.135493 / 49.104258 | 9786.105060 | 4809.512431 | 153.263396 | A100-PCIE-40GB /250W |
| DeepSeek-VL2-Small | 3 | 49.745808 | 49.833333 / 49.868700 / 49.535714 | 8474.758845 | 4142.332645 | 150.584722 | A100-PCIE-40GB /250W |
| CLIP ViT-B/32 | 1 | 39.198883 | 39.460637 / 39.145729 / 38.989933 | 96.456147 | 39.531713 | 71.641304 | A100-PCIE-40GB /250W |
| CLIP ViT-B/32 | 10 | 49.771684 | 49.561139 / 49.534454 / 50.220168 | 17.134323 | 8.809635 | 175.951326 | A100-PCIE-40GB /250W |
| YOLO11s | 1 | 46.226131 | 45.954774 / 46.434564 / 46.289298 | 171.426264 | 81.124415 | 135.032121 | A100-PCIE-40GB /250W |
| YOLO11s | 3 | 49.619632 | 48.015101 / 50.491639 / 50.345576 | 111.433660 | 56.504892 | 168.872027 | A100-PCIE-40GB /250W |
| DeepSeek-VL2-Tiny | 1 | 47.186571 | 47.117550 / 47.619601 / 46.828151 | 3840.556503 | 1875.690184 | 150.190207 | A100-PCIE-40GB /250W |
| DeepSeek-VL2-Tiny | 8 | 49.939394 | 49.906528 / 49.810164 / 50.101493 | 2820.744458 | 1383.449762 | 152.255546 | A100-PCIE-40GB /250W |
| Kimi-VL-A3B-Instruct | 1 | 50.668156 | 51.027875 / 50.149766 / 50.886957 | 6670.007739 | 3246.963150 | 148.600507 | A100-SXM4-80GB /400W |
| Kimi-VL-A3B-Instruct | 6 | 49.840841 | 50.160875 / 49.557125 / 49.805161 | 4490.205712 | 2233.461114 | 159.207303 | A100-SXM4-80GB /400W |

The 120 delivered Kimi raw files passed length and SHA256 checks, and utilization, costs and predictions were independently replayed on CPU in verified_kimi_clean_extension.json. The delivery and original audit remain under the canonical rental run. Previously TEST-selected quality heads retain the development/no-independent-VAL interpretation; no head or backbone was retrained for this follow-up. The model process has exited and its GPU was released.

**Figure interpretation.** The two panels show independently evaluated temporal SRCC against normalized inference latency and estimated energy for the same configurations. Labels give B and separately measured resident utilization. The Kimi solid connector links same-card B=1/B=6; the DeepSeek full dashed connector spans different sessions. Neither the platform differences nor the sampled-power estimates permit attribution solely to model architecture. Paced service costs are intentionally absent from these conventional inference panels.



On 17 October, independent GPU utilization at YOLO B=3 and CLIP B=10 was remeasured using one loaded model process for each ascending batch sequence. All 11 predecessor/target batches and 33 complete repeats were retained. Only the two requested target utilizations replace the corresponding displayed annotations, regardless of direction. Accuracy, inference latency and energy were not remeasured in this utilization-only run: they retain their separately verified same-checkpoint, same-batch sources. Prior fresh-process repeats remain distinct historical records and are never pooled with this run. All other displayed utilization values, including finalized Qwen B=1, remain unchanged. The main display now caps YOLO at B=3 and CLIP at B=10 rather than substituting B=4/B=11. The main figures and the three original workbooks now show 18 electronic points plus one unchanged LightGen-II reference: B=1 and one or two near-saturation settings per model. Intermediate candidates remain in the raw evidence and performance audit, not in the main figures. Numerical source: [plot_data.csv](plot_data.csv); colored comparison: [plot_data.xlsx](plot_data.xlsx). Unrounded values, parameter definitions, checkpoint hashes and distinct cost/utilization sources are retained there. Previous presentation files are preserved with SHA receipts in T06's canonical run, under `prior_report_archive`.

### Same-process native reproduction (17 October 2026)

YOLO was loaded once for B=1,2,3. CLIP was loaded once for B=1,2,3,4,6,8,9,10. Every batch retains all three complete repeats. The terminal targets B3/B10 were required regardless of whether an earlier batch reached98%. No additional B4/B11 target or retry was substituted.

| Model | B | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) | All three ≥98% |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO11s | 1 | 46.84255 | 47.14047 | 46.84396 | 46.94249023 | False |
| YOLO11s | 2 | 72.26510 | 72.52843 | 72.66611 | 72.48659218 | False |
| YOLO11s | 3 | 98.66779 | 98.42689 | 98.86577 | 98.65360940 | True |
| CLIP ViT-B/32 | 1 | 39.44799 | 39.33501 | 39.26466 | 39.34916201 | False |
| CLIP ViT-B/32 | 2 | 44.35966 | 44.37584 | 43.96812 | 44.23447118 | False |
| CLIP ViT-B/32 | 3 | 50.63652 | 50.69799 | 50.32383 | 50.55282281 | False |
| CLIP ViT-B/32 | 4 | 54.41709 | 54.03344 | 53.87605 | 54.10881696 | False |
| CLIP ViT-B/32 | 6 | 69.00168 | 69.04020 | 68.82077 | 68.95418994 | False |
| CLIP ViT-B/32 | 8 | 79.06868 | 79.79530 | 80.40436 | 79.75572946 | False |
| CLIP ViT-B/32 | 9 | 94.16443 | 93.79362 | 94.96980 | 94.30928412 | False |
| CLIP ViT-B/32 | 10 | 98.40604 | 98.80537 | 98.63025 | 98.61387801 | True |

The measured loop is AST-identical to the original: resident first-B four-frame inputs prepared once, native same-stream queue8, each repeat ≥5s warmup plus a complete ≥30s window, all50ms driver samples pooled. All repeat process guards use one PID per model. Source/config/PT/manifest hashes and target input tensor/reference-output identities were verified. No Graph, weight/precision change, clocks/power setting change, training or highest-repeat selection was introduced.

The earlier98.76% CLIP B10 native control was interleaved with a candidate implementation after previous batches. This run uses only the prescribed all-native ascending plan and does not claim an exact reproduction of that interleaved history. The same-process profile is disclosed rather than silently pooled with the previous fresh-process round. One three-repeat run supports within-run reproducibility, not a guarantee for every subsequent run or machine state.

This is utilization-only. Original per-batch accuracy, timing and energy remain from their separately identified verified cost runs. New resident power is never used to recompute those costs. The finalized historical Qwen B1 is neither executed nor changed. The previous fresh-process observations (including YOLO B3 at87.15912898%) remain preserved, not deleted or relabeled as invalid.

All11 batch points and33 complete repeats are retained in the canonical sequential run and raw-repeat table with source roles. Only requested YOLO B3/CLIP B10 replace their displayed utilization annotations. Other B1 measurements are predecessor audit, not replacements of finalized references.


The display rule was fixed before the remaining models completed: keep B=1 and at most two predeclared higher batches with pooled resident utilization ≥95%; if none qualifies, retain the largest predeclared batch and disclose the unmet target. The 98% line is a reference, not a selectable utilization setting or an overload boundary. No point was selected by SRCC, latency or energy. The original 40GB cohort uses its retained A100 UUID and 250-W limit; supplementary 80GB measurements use the separately disclosed 400-W platform and sessions; no training, precision change, CUDA Graph, artificial load or highest-window selection was introduced.

### Methods for the latency and energy figures

The LGVQ split contains 2,250 training videos and 558 TEST videos (manifest SHA256 `607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc`). Each fixed checkpoint was evaluated once on all 558 TEST videos at each measured batch size to obtain that point's SRCC and PLCC. Inputs retain the original four-frame sampling, 65% crop, model-specific resolution and deployed BF16/FP16 backbone plus FP32 readout. B denotes videos per forward, not frames or the normalization workload. Fixed checkpoints previously selected on TEST retain their development/no-independent-VAL interpretation; these measurements introduce no new training or selection.

Accuracy and costs are generated by one dataset-once process per point. The first 200 TEST videos define inference costs, with a separate batch boundary at 200 and 558; incomplete tails are retained, not padded, dropped or charged as full batches. There is no explicit cost warmup, so the first inference is included. The timed boundary begins at the first native backbone block (first convolution for YOLO, first Transformer block for the others) and ends after GPU MOS output synchronization. For optimized Qwen, an equivalent boundary hook is attached to its first visual block. CPU video decoding, preprocessing and host-to-device transfer are recorded only for audit, not included in the displayed inference latency. Thus these are inference-boundary comparisons, not full video-file-to-answer latency.

For synchronized first-200 wall times t_i in milliseconds, T16=(Σt_i)×16/200. Estimated E16=(338.2 W host-power proxy + same-cost-run mean active A100 board power)×T16/1000. The 16-video values are normalized workloads, not a claim that each GPU call uses B=16. Costs are paired with their own recorded active board-power samples; the board power or utilization of the separate resident experiment is never substituted. E16 is the user-approved sampled-power estimate, not a calibrated whole-system energy-meter integral. Driver averaging windows and sparse active samples for short calls limit instantaneous power attribution.

Independent utilization is measured separately to reduce external CPU waiting: the first B TEST videos are prepared and copied to GPU once; each forward still processes B videos. Eight model forwards are queued on the same CUDA stream before synchronization (queue8 does not mean 8B concurrent videos). Each point has three identical repeats, each with at least 5 s warmup and a complete ≥30 s measurement window. All 50-ms telemetry samples within all three complete windows are pooled; no high-utilization interval or best repeat is selected. NVML reports driver-window kernel-busy time, not SM occupancy or achieved FLOPS. Cost-run and resident-utilization dates, clocks, power, memory and CPU scheduling records remain distinct.

### Qwen execution optimization and protected references

Qwen B=3/4 use CPU integer metadata, equivalent indexing and grouped independent equal-length visual-attention calls to reduce host/device synchronization and launch overhead. Learned weights, architecture, numerical precision and attention connectivity are unchanged. This is an execution implementation change and must be disclosed; it is not an unmodified off-the-shelf forward implementation. Both points passed exact comparisons of hidden states, full vocabulary logits, quality logits and MOS on all 558 TEST videos, with the fresh cold-cost predictions also equal. This establishes equivalence for the tested inputs and profiles, not for every possible input or batch. Qwen's differences against historical B=1 include both batch and execution optimization and must not be described as a pure batch-scaling ablation.

† The finalized40GB Qwen B=1 point is not replaced: SRCC=0.7663428036522781, T16=1188.3861464262009 ms, E16=462.1134010767563 J, and independent resident utilization=63.133111480865225% are the finalized historical references. Its utilization comes from one retained 5-s/30-s native run, not this round's three-repeat mean. The separate historical three-repeat audit of 71.17536071032187% is preserved and is not displayed as the selected B=1 value. The historical B=1 point is not connected to the optimized Qwen points in the figures. LightGen-II retains physical SRCC=0.7977138739, T16=10.245569591370225 ms and estimated E16=1.1776895097085185 J; N=16 denotes videos encoded in the optical field, not a GPU batch. Optical cost remains the approved component estimate, not a newly measured GPU-style process.

### Complete data used in both figures

| Model | B / optical N | Resident GPU utilization (%) | Full558 SRCC | T16 (ms) | Estimated E16 (J) | LightGen-II time advantage (×) | LightGen-II energy-efficiency advantage (×) | Platform / limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LightGen-II | N=16 | — | 0.79771387 | 10.245570 | 1.177690 | 1.000 | 1.000 | Optical system |
| Qwen3-VL-2B † | 1 | 63.13 | 0.76634280 | 1188.386146 | 462.113401 | 115.990 | 392.390 | A100-PCIE-40GB / 250W |
| Qwen3-VL-2B | 3 | 97.14 | 0.76889942 | 526.553994 | 209.058386 | 51.393 | 177.516 | A100-PCIE-40GB / 250W |
| Qwen3-VL-2B | 4 | 97.06 | 0.76949583 | 481.665483 | 203.091550 | 47.012 | 172.449 | A100-PCIE-40GB / 250W |
| GLM-4.6V-Flash | 1 | 96.88 | 0.80926196 | 1810.626785 | 805.188484 | 176.723 | 683.702 | A100-PCIE-40GB / 250W |
| GLM-4.6V-Flash | 2 | 98.04 | 0.81030997 | 1631.116714 | 775.784796 | 159.202 | 658.735 | A100-PCIE-40GB / 250W |
| DeepSeek-VL2-Small | 1 | 77.65 | 0.70007977 | 6323.384420 | 3190.408718 | 617.182 | 2709.041 | A100-PCIE-40GB / 250W |
| DeepSeek-VL2-Small | 3 | 95.85 | 0.70065163 | 4454.141151 | 2483.077033 | 434.738 | 2108.431 | A100-PCIE-40GB / 250W |
| DeepSeek-VL2-Small | 4 | 98.58 | 0.69753655 | 4277.833742 | 2390.636129 | 417.530 | 2029.938 | A100-PCIE-40GB / 250W |
| CLIP ViT-B/32 | 1 | 39.59 | 0.72525581 | 167.503529 | 63.381562 | 16.349 | 53.819 | A100-PCIE-40GB / 250W |
| CLIP ViT-B/32 | 10 | 98.61 | 0.72523882 | 30.817941 | 11.816369 | 3.008 | 10.034 | A100-PCIE-40GB / 250W |
| YOLO11s | 1 | 45.00 | 0.70159701 | 367.670324 | 140.508523 | 35.886 | 119.309 | A100-PCIE-40GB / 250W |
| YOLO11s | 3 | 98.65 | 0.70159400 | 151.266861 | 57.283617 | 14.764 | 48.641 | A100-PCIE-40GB / 250W |
| DeepSeek-VL2-Tiny | 1 | 72.41 | 0.56846690 | 2962.932000 | 1348.909189 | 289.192 | 1145.386 | A100-PCIE-40GB / 250W |
| DeepSeek-VL2-Tiny | 8 | 96.96 | 0.56982445 | 1560.458820 | 827.419736 | 152.306 | 702.579 | A100-PCIE-40GB / 250W |
| DeepSeek-VL2 | 1 | 88.05 | 0.71673526 | 8360.056700 | 4727.746492 | 815.968 | 4014.425 | A100-SXM4-80GB / 400W |
| DeepSeek-VL2 | 2 | 96.54 | 0.71658420 | 7189.600891 | 4372.864338 | 701.728 | 3713.088 | A100-SXM4-80GB / 400W |
| Kimi-VL-A3B-Instruct | 1 | 73.99 | 0.80546527 | 3745.677059 | 1803.738445 | 365.590 | 1531.591 | A100-SXM4-80GB / 400W |
| Kimi-VL-A3B-Instruct | 6 | 98.39 | 0.80902087 | 1380.524297 | 734.780588 | 134.744 | 623.917 | A100-SXM4-80GB / 400W |

LightGen-II time advantage is baseline T16 / LightGen-II T16; energy-efficiency advantage is baseline E16 / LightGen-II E16 for the same normalized 16-video workload, not TOPS/W. Relative-to-B1 improvements are additionally available in the workbook. Model parameter sizes retain their declared nominal/loaded-total definitions, with MoE active parameters listed separately; they are not inferred from memory usage or optical degrees of freedom.

### Manuscript-ready descriptions of the two panels

**Latency panel.** Temporal-quality assessment accuracy is plotted against inference latency normalized to 16 LGVQ videos. Accuracy is evaluated independently at each model–batch configuration on all 558 test videos; GPU latency is derived from synchronized inference-boundary timings of the first 200 videos. Original-platform models retain their selected B=1 and high-batch points; the80GB supplementary cohort retains DeepSeek B1/B2 and adds same-card Kimi B1/B6. The label at each point gives video batch size and independently measured resident GPU utilization. LightGen-II is shown using its retained experimental reference. The finalized native Qwen B=1 reference is distinguished from the equivalence-validated optimized execution used at B=3/4. Lower latency at comparable SRCC indicates a more favorable inference trade-off; no monotonic accuracy trend with batch is assumed.

**Energy panel.** The same per-configuration temporal SRCC is plotted against estimated energy normalized to 16 videos. For each new GPU point, energy combines its synchronized inference timing with its own active A100 power samples and the fixed 338.2-W host-power proxy. Resident-utilization measurements are used only for annotation and do not supply the power in this calculation. The LightGen-II and finalized Qwen B=1 references retain their approved historical cost estimates. The logarithmic horizontal axis covers every displayed point. These results compare estimated inference energy under the declared boundaries, rather than wall-meter energy for decoding and all host processing.

Current figures: [two-panel comparison](srcc_cost_comparison_reference.png), [latency](latency_comparison_reference.png), [estimated energy](energy_comparison_reference.png). Their editable SVG counterparts and [independent utilization data](LGVQ_temporal_gpu_utilization.csv) are in this same directory. Reproduce the evidence audit with T06 `audit_selected_execution.py`, initial workbook authoring with `reports/build_selected_execution.mjs --build` (refuses to overwrite a changed archive), saved-workbook verification with `--verify`, formatting-only updates with `--polish`, and the figures with `reports/plot_selected_execution.py`. Add `--yolo-clip-sequential` to the audit, workbook and figure commands; original, recheck and current verified-source SHA values are recorded in the presentation receipt. Do not import the old overwrite-on-import `build_analysis.py`.

---

# Historical notes — superseded presentation, protected evidence

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

[Archived prior README](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/figure5_kimi_refresh_20261018/prior_report_archive/README.md).
