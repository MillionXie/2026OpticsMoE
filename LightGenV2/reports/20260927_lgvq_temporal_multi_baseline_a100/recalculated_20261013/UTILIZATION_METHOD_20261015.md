# LGVQ GPU utilization: current selected-execution comparison

The independently audited 16-point selected round is retained, with 4 prescribed YOLO/CLIP recheck points from 17 October replacing the corresponding older points regardless of increase or decrease. Both old and new raw repeats are retained separately. All other models, including the historical Qwen B1, are unchanged. The no-explicit-warmup cost protocol still includes each first call. CLIP B10 first-call latency changed from534.974873 to197.230 ms (rounded), which explains most of the lower first200 normalized cost in this round; the timing boundary was not shortened. These cold-first-call costs retain initialization/cache-state variability and are not a steady-state throughput measurement. Current point means and repeats are shown below. Superseded repeats remain separately labeled in the raw-repeat workbook/CSV, never pooled with a later round. Main figures use B=1 plus at most two predeclared higher batches with pooled resident utilization ≥95% (largest tested batch if none qualifies). All other points remain audit evidence. The three workbooks and CSV identify displayed versus audit-only repeats. See the current comparison README for paired inference-cost data and the two figure descriptions.

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


The measurement method is unchanged: first B TEST videos prepared/H2D once, original four frames, original precision, same-stream queue8, three identical ≥5-s warmup/≥30-s full measurement repeats, all 50-ms samples pooled. Each call processes B videos, not 8B. Driver-window utilization is kernel-busy time, not SM occupancy or a configured percentage. There is no CUDA Graph, synthetic load, best-window selection, clock change or new training. Qwen B3/4 change execution grouping/indexing under full558 exact-equivalence checks; the other models use native execution.

Qwen B1 stays at the finalized 63.133111480865225% historical one-repeat value and was not reexecuted. The earlier 71.17536071032187% three-repeat audit remains a separate historical record. Neither is a new same-round result. The finalized Qwen B1 costs are unchanged. Resident utilization and power are independent of the dataset-once cost experiment; resident power must not enter the cost formula.

| Model | B | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) | Full-window samples | Duration (s) | Execution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-VL-2B | 3 | 96.75125 | 97.34328 | 97.30858 | 97.13550885 | 1808 | 91.103 | equivalence-validated optimized Qwen execution |
| Qwen3-VL-2B | 4 | 97.28947 | 96.92333 | 96.95688 | 97.05742684 | 1811 | 91.509 | equivalence-validated optimized Qwen execution |
| GLM-4.6V-Flash | 1 | 96.75327 | 97.01163 | 96.88303 | 96.88193300 | 1821 | 91.929 | native |
| GLM-4.6V-Flash | 2 | 97.94649 | 98.06612 | 98.09195 | 98.03532009 | 1812 | 91.563 | native |
| DeepSeek-VL2-Small | 1 | 77.98769 | 77.46901 | 77.45819 | 77.64823848 | 1845 | 93.000 | native |
| DeepSeek-VL2-Small | 3 | 95.62250 | 95.90701 | 96.02588 | 95.85270133 | 1962 | 98.909 | native |
| DeepSeek-VL2-Small | 4 | 98.86646 | 98.49025 | 98.38187 | 98.57778889 | 1999 | 100.938 | native |
| CLIP ViT-B/32 | 1 | 39.29481 | 39.78188 | 39.69849 | 39.59162011 | 1790 | 90.071 | native |
| CLIP ViT-B/32 | 9 | 93.04698 | 92.07718 | 92.57383 | 92.56599553 | 1788 | 90.084 | native |
| CLIP ViT-B/32 | 10 | 97.22185 | 95.74664 | 97.60840 | 96.85834267 | 1786 | 90.072 | native |
| CLIP ViT-B/32 | 11 | 98.66611 | 98.83221 | 98.71477 | 98.73769575 | 1788 | 90.050 | native |
| YOLO11s | 1 | 44.99664 | 44.81773 | 45.17923 | 44.99776661 | 1791 | 90.154 | native |
| YOLO11s | 2 | 66.59564 | 67.72148 | 67.96315 | 67.42705422 | 1789 | 90.133 | native |
| YOLO11s | 3 | 81.82915 | 97.34338 | 82.30486 | 87.15912898 | 1791 | 90.166 | native |
| YOLO11s | 4 | 99.77685 | 99.73950 | 99.10906 | 99.54168998 | 1787 | 90.090 | native |
| DeepSeek-VL2-Tiny | 1 | 72.54739 | 72.56169 | 72.10235 | 72.40679825 | 1824 | 92.090 | native |
| DeepSeek-VL2-Tiny | 6 | 94.83495 | 94.98234 | 94.88217 | 94.89994650 | 1869 | 94.335 | native |
| DeepSeek-VL2-Tiny | 8 | 96.85465 | 97.06657 | 96.96403 | 96.96190935 | 2074 | 104.612 | native |

Full source hashes, raw telemetry, forward counts, clocks, temperatures and process guards are independently audited in T06 `runs/simulation/lgvq_selected_execution_a100_20261016/verified_selected_execution.json` and `runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/verified_selected_execution.json`. Raw reports and their transfer SHA receipts are retained alongside them. The 98% line is a reporting reference, not an overload limit; results below it are not rounded upward.

---

# Historical utilization notes — preserved, not the current display

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

[Archived prior utilization methods](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/prior_report_archive/UTILIZATION_METHOD_20261015.md).
