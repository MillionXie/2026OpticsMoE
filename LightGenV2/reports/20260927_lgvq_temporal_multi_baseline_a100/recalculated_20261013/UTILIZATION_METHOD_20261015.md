# LGVQ GPU utilization: current selected-execution comparison

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

**Separate paced service measurements.** Kimi B=1 and B=6 were also measured at approximately 50% driver-window kernel-busy time. This is a temporal duty-cycle constraint, not an equivalence to optical spatial occupancy. The first B videos are resident and repeated; complete-group feedback includes both compute and imposed idle time during a ≥30-s calibration, then freezes the duty for three complete repeats. Service T16 = total complete-window duration × 16 / completed video evaluations; estimated service E16 = (338.2 W + duration-weighted same-window mean board power) × service T16/1000 (with T16 in milliseconds). All imposed waits are included. These are not the dataset-once inference costs plotted in the main panels, and their resident board power must not enter the first-200 cost formula. Arithmetic telemetry means retain the limitations of sampled driver windows, rather than representing a calibrated energy integral. This rental controller uses the last complete compute-plus-idle group for feedback; the paused original40GB v2 run aggregates completed groups since the preceding update. They must not be described as bit-identical controller profiles.

| B | Busy mean (%) | Repeat1 / 2 / 3 (%) | Full-window board mean (W) | Service T16 (ms) | Estimated service E16 (J) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 50.66815642 | 51.02787456 / 50.14976599 / 50.88695652 | 148.60050712 | 6670.00773935 | 3246.96315000 |
| 6 | 49.84084084 | 50.16087516 / 49.55712452 / 49.80516129 | 159.20730310 | 4490.20571213 | 2233.46111364 |

Paced data are retained in [LGVQ_kimi_paced_service.csv](LGVQ_kimi_paced_service.csv) and the separate Paced service worksheet; the six-batch native trend is retained in [LGVQ_kimi_native_utilization_trend.csv](LGVQ_kimi_native_utilization_trend.csv). The original40GB paced experiment is incomplete after a telemetry-reader failure: four points passed independent raw audit, eight remain uncompleted. Neither failed telemetry nor inferred values enter these Kimi tables. Completion of this Kimi follow-up does not imply completion of that separate suite.

The 120 delivered raw files passed length and SHA256 checks, and utilization, costs and predictions were independently replayed on CPU in verified_kimi_clean_extension.json. The delivery and original audit remain under the canonical rental run. Previously TEST-selected quality heads retain the development/no-independent-VAL interpretation; no head or backbone was retrained for this follow-up. The model process has exited and its GPU was released.

**Figure interpretation.** The two panels show independently evaluated temporal SRCC against normalized inference latency and estimated energy for the same configurations. Labels give B and separately measured resident utilization. The Kimi solid connector links same-card B=1/B=6; the DeepSeek full dashed connector spans different sessions. Neither the platform differences nor the sampled-power estimates permit attribution solely to model architecture. Paced service costs are intentionally absent from these conventional inference panels.

On 17 October, independent GPU utilization at YOLO B=3 and CLIP B=10 was remeasured using one loaded model process for each ascending batch sequence. All 11 predecessor/target batches and 33 complete repeats were retained. Only the two requested target utilizations replace the corresponding displayed annotations, regardless of direction. Accuracy, inference latency and energy were not remeasured in this utilization-only run: they retain their separately verified same-checkpoint, same-batch sources. Prior fresh-process repeats remain distinct historical records and are never pooled with this run. All other displayed utilization values, including finalized Qwen B=1, remain unchanged. The main display now caps YOLO at B=3 and CLIP at B=10 rather than substituting B=4/B=11. Current point means and repeats are shown below. Superseded repeats remain separately labeled in the raw-repeat workbook/CSV, never pooled with a later round. Main figures use B=1 plus at most two predeclared higher batches with pooled resident utilization ≥95% (largest tested batch if none qualifies). All other points remain audit evidence. The three workbooks and CSV identify displayed versus audit-only repeats. See the current comparison README for paired inference-cost data and the two figure descriptions.

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


The measurement method is unchanged: first B TEST videos prepared/H2D once, original four frames, original precision, same-stream queue8, three identical ≥5-s warmup/≥30-s full measurement repeats, all 50-ms samples pooled. Each call processes B videos, not 8B. Driver-window utilization is kernel-busy time, not SM occupancy or a configured percentage. There is no CUDA Graph, synthetic load, best-window selection, clock change or new training. Qwen B3/4 change execution grouping/indexing under full558 exact-equivalence checks; the other models use native execution.

The original40GB Qwen B1 stays at the finalized 63.133111480865225% historical one-repeat value and was not reexecuted. The earlier 71.17536071032187% three-repeat audit remains a separate historical record. Neither is a new same-round result. The finalized Qwen B1 costs are unchanged. Resident utilization and power are independent of the dataset-once cost experiment; resident power must not enter the cost formula.

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
| CLIP ViT-B/32 | 10 | 98.40604 | 98.80537 | 98.63025 | 98.61387801 | 1787 | 90.075 | native |
| CLIP ViT-B/32 | 11 | 98.66611 | 98.83221 | 98.71477 | 98.73769575 | 1788 | 90.050 | native |
| YOLO11s | 1 | 44.99664 | 44.81773 | 45.17923 | 44.99776661 | 1791 | 90.154 | native |
| YOLO11s | 2 | 66.59564 | 67.72148 | 67.96315 | 67.42705422 | 1789 | 90.133 | native |
| YOLO11s | 3 | 98.66779 | 98.42689 | 98.86577 | 98.65360940 | 1787 | 90.043 | native |
| YOLO11s | 4 | 99.77685 | 99.73950 | 99.10906 | 99.54168998 | 1787 | 90.090 | native |
| DeepSeek-VL2-Tiny | 1 | 72.54739 | 72.56169 | 72.10235 | 72.40679825 | 1824 | 92.090 | native |
| DeepSeek-VL2-Tiny | 6 | 94.83495 | 94.98234 | 94.88217 | 94.89994650 | 1869 | 94.335 | native |
| DeepSeek-VL2-Tiny | 8 | 96.85465 | 97.06657 | 96.96403 | 96.96190935 | 2074 | 104.612 | native |
| DeepSeek-VL2 | 1 | 88.14900 | 88.12769 | 87.87117 | 88.04915515 | 1953 | 97.935 | Native frozen; supplementary A100-SXM4-80GB /400W |
| DeepSeek-VL2 | 2 | 96.80226 | 96.46554 | 96.35112 | 96.53918348 | 2131 | 106.922 | Native frozen; supplementary A100-SXM4-80GB /400W |
| Kimi-VL-A3B-Instruct | 1 | 73.98686 | 74.05902 | 73.91667 | 73.98743856 | 1831 | 95.819 | Native frozen; clean same-card A100-SXM4-80GB /400W; utilization B1 to6 in one model process |
| Kimi-VL-A3B-Instruct | 6 | 98.33130 | 98.43227 | 98.41538 | 98.39296636 | 1962 | 102.812 | Native frozen; clean same-card A100-SXM4-80GB /400W; utilization B1 to6 in one model process |

Full source hashes, raw telemetry, forward counts, clocks, temperatures and process guards are independently audited in T06 `runs/simulation/lgvq_selected_execution_a100_20261016/verified_selected_execution.json` and `runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/verified_selected_execution.json` and `runs/simulation/lgvq_yolo_clip_sequential_a100_20261017/verified_sequential.json`. Raw reports and their transfer SHA receipts are retained alongside them. The 98% line is a reporting reference, not an overload limit; results below it are not rounded upward.

---

# Historical utilization notes — preserved, not the current display

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

[Archived prior utilization methods](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/figure5_kimi_refresh_20261018/prior_report_archive/UTILIZATION_METHOD_20261015.md).
