# LGVQ batch comparison: offline recalculation, 2026-10-13

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
in time and 392.3898 in energy. CLIP/YOLO/DeepSeek remain historical cost plus
fixed performance references: exact original trained temporal heads have not
been located, so no new batch-specific performance is claimed for them.

The following continuous-GPU sweep description and 81x/348x figures document
the previous audit and are not the current Qwen output. Original raw records
are kept, rather than rewritten to match the final table.

## Results and files

- `batch_cost_comparison.png/svg`: revised 16-video energy and processing time versus videos per call. No unmeasured accuracy trend is implied.
- `srcc_cost_comparison_reference.png/svg`: original quality-versus-cost layout with corrected inputs. Filled Qwen symbols use real full558 scores at B; open/dashed CLIP/YOLO/DeepSeek symbols repeat formal reference scores explicitly, not purported batch measurements. The LightGen-II physical score is separate from its component-cost proxy.
- `qwen_batch_metric_audit.png/svg`: actual SRCC recomputed from saved Qwen predictions. Full TEST and first-200 subset are separate panels; neither substitutes for the other.
- `LGVQ_temporal_multi_baseline_recalculated.xlsx`: original three-tab structure retained, with updated Ours inputs and explicit measured-at-batch SRCC columns. Original workbook is unchanged.
- `timing_energy_recalculated.csv`: all 16 baseline settings, unrounded calculation values, per-video amortized quantities and precision/source fields.
- `batch_metric_audit.csv`: separate full-test references, actual full558/subset200 batch results and missing-data status.
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
The historical visual-model power pass is a separate sustained inference measurement,
not a time integral of each timed call. These are mean-power duration estimates.

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

## Accuracy audit and theory

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

## What needs a new evaluation

No rerun is required for the reported arithmetic update or recovered Qwen
full-test metric sweep. A fully measured multi-model SRCC-versus-batch graph
still requires CLIP/YOLO/DeepSeek B1/B2/B4/B8 using their frozen trained
checkpoints on the same 558 videos. Their existing trained prediction assets
were not found in the bounded local/server inventory; this does not establish
that they were deleted. Only missing settings need inference once the exact
trained PT and dataset contract are available.
No retraining or new optical capture is necessary for these electronic baselines.
Cached B1 features can verify a reference head score but cannot establish that
native B8 backbone inference yields identical features.

The original reports identify A100 but not all retain GPU UUID, power-limit
and process-audit evidence. DeepSeek B8 raw power has peaks above 300 W. This
does not by itself prove invalid data; independent physical-device/power-limit
identity should be recovered before a stronger calibrated-power claim.

The offline calculation itself does not use GPU/SDK. Its updated Qwen source
was actually reevaluated on A100 as stated above. Source code was Git-synced;
raw predictions/weights were not externally uploaded.
