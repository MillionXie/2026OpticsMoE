# LGVQ GPU utilization: current selected-execution comparison

Sixteen new model–batch points were remeasured on 16–17 October 2026; all 48 complete repeats are retained below. Main figures use B=1 plus at most two predeclared higher batches with pooled resident utilization ≥95% (largest tested batch if none qualifies). All other points remain audit evidence. The three workbooks and CSV identify displayed versus audit-only repeats. See the current comparison README for paired inference-cost data and the two figure descriptions.

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
| CLIP ViT-B/32 | 10 | 95.73658 | 97.80168 | 98.17617 | 97.23782876 | 1787 | 90.063 | native |
| YOLO11s | 1 | 44.99664 | 44.81773 | 45.17923 | 44.99776661 | 1791 | 90.154 | native |
| YOLO11s | 2 | 66.59564 | 67.72148 | 67.96315 | 67.42705422 | 1789 | 90.133 | native |
| YOLO11s | 3 | 94.15217 | 95.33779 | 96.64824 | 95.37869492 | 1793 | 90.241 | native |
| DeepSeek-VL2-Tiny | 1 | 72.54739 | 72.56169 | 72.10235 | 72.40679825 | 1824 | 92.090 | native |
| DeepSeek-VL2-Tiny | 6 | 94.83495 | 94.98234 | 94.88217 | 94.89994650 | 1869 | 94.335 | native |
| DeepSeek-VL2-Tiny | 8 | 96.85465 | 97.06657 | 96.96403 | 96.96190935 | 2074 | 104.612 | native |

Full source hashes, raw telemetry, forward counts, clocks, temperatures and process guards are independently audited in T06 `runs/simulation/lgvq_selected_execution_a100_20261016/verified_selected_execution.json`. Raw reports and their transfer SHA receipts are retained alongside it. The 98% line is a reporting reference, not an overload limit; results below it are not rounded upward.

---

# Historical utilization notes — preserved, not the current display

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

[Archived prior utilization methods](../../../tasks/t06_video_quality_assessment/runs/simulation/lgvq_selected_execution_a100_20261016/prior_report_archive/UTILIZATION_METHOD_20261015.md).
