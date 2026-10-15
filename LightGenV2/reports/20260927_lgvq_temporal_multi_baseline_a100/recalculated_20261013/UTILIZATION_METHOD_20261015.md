# LGVQ GPU utilization: execution gaps and steady-state measurement

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
