# A100 nine-metric timing and energy table — 2026-09-28

This directory is the single current location for the nine-row comparison table. The current workbook is `A100_nine_task_energy_efficiency_COMPLETE_20260928.xlsx`; the older same-date workbook without `COMPLETE` was open/locked during regeneration and must not be used. Earlier workbooks remain archived under `20260927_demo_energy_efficiency_a100/archive_superseded_20260927/`.

## Uniform timing scope

- Device: physical GPU 6, NVIDIA A100-PCIE-40GB; every formal run was audited with no foreign compute process.
- Samples: 200 measured calls; no explicit warm-up; the first call is retained.
- Ours electronic time: CUDA Event mean of router arithmetic, CCD nonlinearity/learned readout, learned bridge where present, and the actual task head. Parallel residual branches, host/device transfer, file I/O, SLM layout/encoding, CCD crop/stack, and RMS/alpha fusion are not included.
- Baseline time: synchronized wall mean from the first transformer block input through the final task output.
- Physical time: one optical pass is `0.714 + 0.300 + 0.0307 = 1.0447 ms`; multimodal tasks use six passes, while LSP and SALICON use three.
- LGVQ temporal is a matched workload of 16 videos. Its baseline per-video time is multiplied by 16; Ours already outputs 16 videos in parallel.

## Energy equations

Fixed measured powers:

- optical devices = 80.358 W;
- Ours control chassis = 41.388 W;
- Ours A100 board = 62.842 W;
- baseline chassis = 338.2 W;
- baseline A100 board = the task-specific `nvidia-smi` active mean.

Equations:

`E_ours = t_optical × (80.358 + 41.388) + t_electronic × (41.388 + 62.842)`

`E_baseline = t_baseline × (338.2 + P_A100,task)`

Times in the energy equations are in seconds.

## 2026-09-28 corrections

- ABO image-to-image now uses the current R@1 = 0.8225 checkpoint and the same narrow timing scope as the other ABO tasks. Its formal table value is 7.9384 ms, not the earlier 12.7124 ms broad-pipeline candidate. The broad candidate included the merger, 1600-gallery similarity, stable sorting, and other work outside the table scope.
- `raw_remote/formal_03_narrow/report.json` is retained as an independent identity/component audit. It starts several components independently and therefore charges multiple first-use CUDA initialization spikes; its summed 10.8691 ms is not the unified table statistic. The table uses the single unified 7-task process and the same sequential measurement order/protocol for every existing task.
- ABO clean text-to-image uses six optical passes. Its narrow electronic time is 8.5051 ms and total time is 14.7733 ms. The heavier time is from the current 256×256 RGB neural decoder task head, not from using the wrong GPU.
- ABO text-to-image retrieval (row 04) is now complete. The Ours result is bound to the exact 10 cm checkpoint with R@1 = 0.8800 (SHA-256 `8a96132d8ba68c671426d4e0b6064e87b5b3943505c0ad8b2aa694c879c20883`) and uses six optical passes. Its narrow CUDA electronic mean is 1.4004 ms, total latency is 7.6686 ms, and energy is 0.9091 J.
- The row-04 frozen Qwen baseline uses 100 official title queries against the precomputed 2400-image held-out gallery. Across 200 timed calls (the 100-query set repeated twice), R@1 is 0.8200, synchronized-wall mean is 32.4666 ms, active A100 board power is 65.0821 W, and energy is 13.0932 J. The resulting speedup is 4.2337× and energy-efficiency gain is 14.4025×.

## Evidence

- `raw_remote/ours_narrow_200/`: the unified 7-task A100 run, including 8,400 per-call rows.
- `raw_remote/formal_03_narrow/`: current ABO image-to-image identity and independent scope audit.
- `raw_remote/formal_04_ours/`: row-04 checkpoint identity, complete per-component timings, and occurrence contract.
- `raw_remote/formal_04_baseline/`: row-04 per-query wall/CUDA timings, raw 10 ms A100 telemetry, performance provenance, and gallery embedding hash.
- `raw_remote/formal_08_narrow/`: current ABO clean text-to-image narrow component audit.
- `raw_remote/formal_08_fullaudit/`: matched Qwen baseline timing, board power, checkpoint identity, and test manifest.

The workbook uses colors plus a textual `Basis/status` column: blue = measured, green = fixed hardware input, orange = formula-derived, and yellow = performance.
