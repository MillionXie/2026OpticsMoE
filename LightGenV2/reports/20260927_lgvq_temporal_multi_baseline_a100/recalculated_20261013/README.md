# LGVQ temporal quality: selected-batch accuracy, latency and estimated energy

## Current comparison — 16–17 October 2026

The independently audited 16-point selected round is retained, with 4 prescribed YOLO/CLIP recheck points from 17 October replacing the corresponding older points regardless of increase or decrease. Both old and new raw repeats are retained separately. All other models, including the historical Qwen B1, are unchanged. The no-explicit-warmup cost protocol still includes each first call. CLIP B10 first-call latency changed from534.974873 to197.230 ms (rounded), which explains most of the lower first200 normalized cost in this round; the timing boundary was not shortened. These cold-first-call costs retain initialization/cache-state variability and are not a steady-state throughput measurement. The main figures and the three original workbooks now show 15 electronic points plus one unchanged LightGen-II reference: B=1 and one or two near-saturation settings per model. Intermediate candidates remain in the raw evidence and performance audit, not in the main figures. Numerical source: [plot_data.csv](plot_data.csv); colored comparison: [plot_data.xlsx](plot_data.xlsx). Unrounded values, parameter definitions, checkpoint hashes and distinct cost/utilization sources are retained there. Previous presentation files are preserved with SHA receipts in T06's canonical run, under `prior_report_archive`.

### YOLO/CLIP reproducibility check (17 October 2026)

The inference-loop/statistics modules have identical SHA256 across the compared native runs: resident inputs prepared once, same-stream queue8, three complete ≥5-s warmups and ≥30-s measurement windows, all 50-ms samples pooled. YOLO B3 sample IDs, frames, resident tensor identity and reference outputs also match.

The full execution histories were not identical. The old YOLO B3 followed B1/B2 in one loaded process. The old high CLIP B10 native control followed earlier batches/controls in one process. The recent selected run and this prescribed recheck start a fresh process for each point/phase. Clock, temperature and launch/host execution state therefore need not match despite an identical inner loop. The prior statement “completely identical” applied only to the loop/statistics, not this process history.

| Native record | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) |
| --- | ---: | ---: | ---: | ---: |
| YOLO old sequential B3 | 98.96309 | 98.78655 | 97.54271 | 98.43009 |
| YOLO selected fresh B3 | 94.15217 | 95.33779 | 96.64824 | 95.37869 |
| CLIP old sequential B10 | 95.72987 | 95.82689 | 93.12584 | 94.89368 |
| CLIP native control after earlier work B10 | 98.96309 | 98.88926 | 98.41779 | 98.75671 |
| CLIP selected fresh B10 | 95.73658 | 97.80168 | 98.17617 | 97.23783 |
| YOLO11s prescribed recheck B3 | 81.82915 | 97.34338 | 82.30486 | 87.15913 |
| YOLO11s prescribed recheck B4 | 99.77685 | 99.73950 | 99.10906 | 99.54169 |
| CLIP ViT-B/32 prescribed recheck B10 | 97.22185 | 95.74664 | 97.60840 | 96.85834 |
| CLIP ViT-B/32 prescribed recheck B11 | 98.66611 | 98.83221 | 98.71477 | 98.73770 |

The recheck uses the unchanged fresh-process method, not selected high windows or longer conditioning. If YOLO3/CLIP10 pooled means remain below98%, only the predeclared adjacent YOLO4/CLIP11 is added. Latest complete prescribed results replace the corresponding old displayed point regardless of direction. Earlier data remain historical and are never pooled across rounds.

Clock/temperature co-variation is diagnostic evidence, not an isolated causal attribution. The recheck YOLO B3 repeat2 was near97% while repeats1/3 were near82%; main-thread run-queue waiting was only about14ms per30s, so neither pure thermal warmup nor CPU scheduling delay alone explains the spread. Host launch service, CPU/GPU frequency and driver-window effects remain possible contributors. We do not claim to have eliminated all environmental variance. The GPU utilization counter is kernel-busy time, not compute occupancy and not a batch-linear quantity.

The per-repeat clocks, temperatures, forward throughput and CPU scheduling evidence are retained in T06 canonical recheck run `method_comparison.csv` with source SHA receipts. Precision, weights, original four frames, clock/power settings and finalized Qwen B1 are unchanged. Utilization is independent of the newly paired first200 latency/power cost measurements.


The display rule was fixed before the remaining models completed: keep B=1 and at most two predeclared higher batches with pooled resident utilization ≥95%; if none qualifies, retain the largest predeclared batch and disclose the unmet target. The 98% line is a reference, not a selectable utilization setting or an overload boundary. No point was selected by SRCC, latency or energy. All new measurements use the same A100 UUID and 250-W power limit; no training, precision change, CUDA Graph, artificial load or highest-window selection was introduced.

### Methods for the latency and energy figures

The LGVQ split contains 2,250 training videos and 558 TEST videos (manifest SHA256 `607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc`). Each fixed checkpoint was evaluated once on all 558 TEST videos at each measured batch size to obtain that point's SRCC and PLCC. Inputs retain the original four-frame sampling, 65% crop, model-specific resolution and deployed BF16/FP16 backbone plus FP32 readout. B denotes videos per forward, not frames or the normalization workload. Fixed checkpoints previously selected on TEST retain their development/no-independent-VAL interpretation; these measurements introduce no new training or selection.

Accuracy and costs are generated by one dataset-once process per point. The first 200 TEST videos define inference costs, with a separate batch boundary at 200 and 558; incomplete tails are retained, not padded, dropped or charged as full batches. There is no explicit cost warmup, so the first inference is included. The timed boundary begins at the first native backbone block (first convolution for YOLO, first Transformer block for the others) and ends after GPU MOS output synchronization. For optimized Qwen, an equivalent boundary hook is attached to its first visual block. CPU video decoding, preprocessing and host-to-device transfer are recorded only for audit, not included in the displayed inference latency. Thus these are inference-boundary comparisons, not full video-file-to-answer latency.

For synchronized first-200 wall times t_i in milliseconds, T16=(Σt_i)×16/200. Estimated E16=(338.2 W host-power proxy + same-cost-run mean active A100 board power)×T16/1000. The 16-video values are normalized workloads, not a claim that each GPU call uses B=16. Costs are paired with their own recorded active board-power samples; the board power or utilization of the separate resident experiment is never substituted. E16 is the user-approved sampled-power estimate, not a calibrated whole-system energy-meter integral. Driver averaging windows and sparse active samples for short calls limit instantaneous power attribution.

Independent utilization is measured separately to reduce external CPU waiting: the first B TEST videos are prepared and copied to GPU once; each forward still processes B videos. Eight model forwards are queued on the same CUDA stream before synchronization (queue8 does not mean 8B concurrent videos). Each point has three identical repeats, each with at least 5 s warmup and a complete ≥30 s measurement window. All 50-ms telemetry samples within all three complete windows are pooled; no high-utilization interval or best repeat is selected. NVML reports driver-window kernel-busy time, not SM occupancy or achieved FLOPS. Cost-run and resident-utilization dates, clocks, power, memory and CPU scheduling records remain distinct.

### Qwen execution optimization and protected references

Qwen B=3/4 use CPU integer metadata, equivalent indexing and grouped independent equal-length visual-attention calls to reduce host/device synchronization and launch overhead. Learned weights, architecture, numerical precision and attention connectivity are unchanged. This is an execution implementation change and must be disclosed; it is not an unmodified off-the-shelf forward implementation. Both points passed exact comparisons of hidden states, full vocabulary logits, quality logits and MOS on all 558 TEST videos, with the fresh cold-cost predictions also equal. This establishes equivalence for the tested inputs and profiles, not for every possible input or batch. Qwen's differences against historical B=1 include both batch and execution optimization and must not be described as a pure batch-scaling ablation.

† Qwen B=1 is not reexecuted: SRCC=0.7663428036522781, T16=1188.3861464262009 ms, E16=462.1134010767563 J, and independent resident utilization=63.133111480865225% are the finalized historical references. Its utilization comes from one retained 5-s/30-s native run, not this round's three-repeat mean. The separate historical three-repeat audit of 71.17536071032187% is preserved and is not displayed as the selected B=1 value. The historical B=1 point is not connected to the optimized Qwen points in the figures. LightGen-II retains physical SRCC=0.7977138739, T16=10.245569591370225 ms and estimated E16=1.1776895097085185 J; N=16 denotes videos encoded in the optical field, not a GPU batch. Optical cost remains the approved component estimate, not a newly measured GPU-style process.

### Complete data used in both figures

| Model | B / optical N | Resident GPU utilization (%) | Full558 SRCC | T16 (ms) | Estimated E16 (J) | LightGen-II time advantage (×) | LightGen-II energy-efficiency advantage (×) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGen-II | N=16 | — | 0.79771387 | 10.245570 | 1.177690 | 1.000 | 1.000 |
| Qwen3-VL-2B † | 1 | 63.13 | 0.76634280 | 1188.386146 | 462.113401 | 115.990 | 392.390 |
| Qwen3-VL-2B | 3 | 97.14 | 0.76889942 | 526.553994 | 209.058386 | 51.393 | 177.516 |
| Qwen3-VL-2B | 4 | 97.06 | 0.76949583 | 481.665483 | 203.091550 | 47.012 | 172.449 |
| GLM-4.6V-Flash | 1 | 96.88 | 0.80926196 | 1810.626785 | 805.188484 | 176.723 | 683.702 |
| GLM-4.6V-Flash | 2 | 98.04 | 0.81030997 | 1631.116714 | 775.784796 | 159.202 | 658.735 |
| DeepSeek-VL2-Small | 1 | 77.65 | 0.70007977 | 6323.384420 | 3190.408718 | 617.182 | 2709.041 |
| DeepSeek-VL2-Small | 3 | 95.85 | 0.70065163 | 4454.141151 | 2483.077033 | 434.738 | 2108.431 |
| DeepSeek-VL2-Small | 4 | 98.58 | 0.69753655 | 4277.833742 | 2390.636129 | 417.530 | 2029.938 |
| CLIP ViT-B/32 | 1 | 39.59 | 0.72525581 | 167.503529 | 63.381562 | 16.349 | 53.819 |
| CLIP ViT-B/32 | 10 | 96.86 | 0.72523882 | 30.817941 | 11.816369 | 3.008 | 10.034 |
| CLIP ViT-B/32 | 11 | 98.74 | 0.72520287 | 30.721120 | 11.756414 | 2.998 | 9.983 |
| YOLO11s | 1 | 45.00 | 0.70159701 | 367.670324 | 140.508523 | 35.886 | 119.309 |
| YOLO11s | 4 | 99.54 | 0.70159166 | 133.947075 | 51.452286 | 13.074 | 43.689 |
| DeepSeek-VL2-Tiny | 1 | 72.41 | 0.56846690 | 2962.932000 | 1348.909189 | 289.192 | 1145.386 |
| DeepSeek-VL2-Tiny | 8 | 96.96 | 0.56982445 | 1560.458820 | 827.419736 | 152.306 | 702.579 |

LightGen-II time advantage is baseline T16 / LightGen-II T16; energy-efficiency advantage is baseline E16 / LightGen-II E16 for the same normalized 16-video workload, not TOPS/W. Relative-to-B1 improvements are additionally available in the workbook. Model parameter sizes retain their declared nominal/loaded-total definitions, with MoE active parameters listed separately; they are not inferred from memory usage or optical degrees of freedom.

### Manuscript-ready descriptions of the two panels

**Latency panel.** Temporal-quality assessment accuracy is plotted against inference latency normalized to 16 LGVQ videos. Accuracy is evaluated independently at each model–batch configuration on all 558 test videos; GPU latency is derived from synchronized inference-boundary timings of the first 200 videos. Each electronic model is represented by B=1 and one or two predeclared near-saturation configurations. The label at each point gives video batch size and independently measured resident GPU utilization. LightGen-II is shown using its retained experimental reference. The finalized native Qwen B=1 reference is distinguished from the equivalence-validated optimized execution used at B=3/4. Lower latency at comparable SRCC indicates a more favorable inference trade-off; no monotonic accuracy trend with batch is assumed.

**Energy panel.** The same per-configuration temporal SRCC is plotted against estimated energy normalized to 16 videos. For each new GPU point, energy combines its synchronized inference timing with its own active A100 power samples and the fixed 338.2-W host-power proxy. Resident-utilization measurements are used only for annotation and do not supply the power in this calculation. The LightGen-II and finalized Qwen B=1 references retain their approved historical cost estimates. The logarithmic horizontal axis covers every displayed point. These results compare estimated inference energy under the declared boundaries, rather than wall-meter energy for decoding and all host processing.

Current figures: [two-panel comparison](srcc_cost_comparison_reference.png), [latency](latency_comparison_reference.png), [estimated energy](energy_comparison_reference.png). Their editable SVG counterparts and [independent utilization data](LGVQ_temporal_gpu_utilization.csv) are in this same directory. Reproduce the evidence audit with T06 `audit_selected_execution.py`, initial workbook authoring with `reports/build_selected_execution.mjs --build` (refuses to overwrite a changed archive), saved-workbook verification with `--verify`, formatting-only updates with `--polish`, and the figures with `reports/plot_selected_execution.py`. For this recheck add `--yolo-clip-recheck` to the audit, workbook and figure commands; both original and recheck verified-source SHA values are recorded in the new presentation receipt. Do not import the old overwrite-on-import `build_analysis.py`.

---

# Historical notes — superseded presentation, protected evidence

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

[Archived prior README](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/prior_report_archive/README.md).
