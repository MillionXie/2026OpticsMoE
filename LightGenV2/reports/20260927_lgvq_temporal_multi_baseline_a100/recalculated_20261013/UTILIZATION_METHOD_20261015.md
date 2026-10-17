# LGVQ GPU utilization: current selected-execution comparison

### Supplementary A100-SXM4-80GB platform (17 October 2026)

Five complete supplementary points were added while retaining all 15 original displayed rows: DeepSeek-VL2 B=1/2, Kimi-VL-A3B-Instruct B=1/2 and native Qwen3-VL-2B B=1. The 80-GB supplementary series use hollow symbols and A100-SXM4-80GB at a 400-W power limit. The daggered historical 40-GB native Qwen B=1 is also an open circle and has a separate legend entry. The original measurements use A100-PCIE-40GB at 250 W, with the historical native Qwen B=1 separately marked. Results across these platforms are descriptive deployment comparisons, not common-hardware model rankings. Energy calculations use measured active board power, not the power limit.

Rental B=1 measurements for DeepSeek and Kimi used physical GPU UUID GPU-780326a6-79bc-7a45-36f6-683c5b278bee. Their B=2 measurements and the new Qwen B=1 control used GPU-d06c95d0-7066-9768-0f6e-fc1ee877f9f9 after the instance was restarted. Dashed connectors guide the eye only. These B=1 and B=2 results do not constitute a strict same-card batch-only ablation. No batch beyond the authorized B=2 was tested. DeepSeek B=2 utilization was 96.54% and Kimi B=2 utilization was 86.11%. Neither reached the 98% reference.

Each of the five points includes independent predictions for all 558 TEST videos and costs measured over the first 200 videos. No explicit warmup is used for cost measurement, and the first inference is included. Timing starts at the input to the first native visual block and ends at synchronized GPU MOS output. Video decoding, preprocessing and host-to-device transfer are excluded. The 338.2-W host proxy is retained and is not a wall-power measurement of the rental host. All 15 resident-utilization repeats use native queue8 execution, at least 5 s of warmup, a complete measurement window of at least 30 s, and pooling of every 50-ms sample in those windows. Each model-batch point uses a separate process. This differs from the cross-batch process lifecycle of the latest YOLO B=3 and CLIP B=10 reproduction.

The 80-GB Qwen control uses the same official revision 89644892e4d85e24eaac8bacfd4f463576704203, quality-head SHA e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2, native SDPA and original BF16/FP32 precision. Input shapes, dtypes, tensor hashes and the prompt match for the first fixed TEST video. All 558 native MOS outputs match the earlier native B=1 outputs exactly. The new T16 of 789.168774 ms and estimated E16 of 324.899880 J are reported separately from the protected 40-GB historical values of 1188.386146 ms and 462.113401 J. The observed ratios are 1.50587 and 1.42233, respectively. Different hosts, dates and Python versions (3.12.13 versus 3.11.15) prevent attribution solely to GPU form factor. The finalized 40-GB utilization of 63.13311148% is unchanged. The independent 80-GB mean is 74.38873551%.

Raw point sources, all repeats and hardware UUIDs are retained in the canonical rental run and verified_figure5_extension.json, bound to followup_delivery_20261017.json and the full CPU audit. Prior presentation files have a separate SHA archive. Kimi and DeepSeek heads were previously selected by highest TEST SRCC, so their scores remain development/no-independent-VAL metrics. No head was retrained for this followup.

**Supplementary figure description.** The latency and energy panels retain the original 40-GB measurements and add five hollow 80-GB points. Each point uses accuracy evaluated on all 558 TEST videos and costs measured over the first 200 videos. Marker fill indicates the platform. Labels show video batch size and independently measured resident GPU utilization. Dashed connectors between rental sessions are descriptive only. The observed accuracy-cost positions do not isolate model architecture, GPU form factor, host, date or batch effects. The cross-device Qwen B=1 control is not connected to the optimized B=3/4 series on the original platform.

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
| Kimi-VL-A3B-Instruct | 1 | 74.23344 | 74.13701 | 74.09748 | 74.15590551 | 1905 | 95.580 | Native frozen; supplementary A100-SXM4-80GB /400W |
| Kimi-VL-A3B-Instruct | 2 | 86.13614 | 86.20908 | 85.97893 | 86.10804970 | 1851 | 92.876 | Native frozen; supplementary A100-SXM4-80GB /400W |
| Qwen3-VL-2B [80GB] | 1 | 74.53322 | 74.41459 | 74.21947 | 74.38873551 | 1811 | 90.819 | Native frozen; supplementary A100-SXM4-80GB /400W |

Full source hashes, raw telemetry, forward counts, clocks, temperatures and process guards are independently audited in T06 `runs/simulation/lgvq_selected_execution_a100_20261016/verified_selected_execution.json` and `runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/verified_selected_execution.json` and `runs/simulation/lgvq_yolo_clip_sequential_a100_20261017/verified_sequential.json`. Raw reports and their transfer SHA receipts are retained alongside them. The 98% line is a reporting reference, not an overload limit; results below it are not rounded upward.

---

# Historical utilization notes — preserved, not the current display

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

[Archived prior utilization methods](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/prior_report_archive/UTILIZATION_METHOD_20261015.md).
