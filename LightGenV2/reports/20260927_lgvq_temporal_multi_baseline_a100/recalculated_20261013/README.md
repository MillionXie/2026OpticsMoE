# LGVQ temporal quality: selected-batch accuracy, latency and estimated energy

## Current comparison — 16–18 October 2026

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
