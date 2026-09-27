# A100 demo energy-efficiency audit (2026-09-27)

This directory is the clean paper-facing audit for the numbered task list. It does not replace or modify older reports.

## Measurement status

| No. | Task | Status |
|---|---|---|
| 01a | LGVQ temporal quality | complete |
| 01b | LGVQ spatial quality | complete |
| 02 | ABO image-to-text retrieval | complete |
| 03 | ABO image-to-image retrieval | complete |
| 04 | ABO text-to-image retrieval | reserved; model still being revised |
| 05 | LSP keypoint detection | complete; exact bound Ours metric is 0.7347857143 |
| 06 | SALICON saliency | complete |
| 07 | OpenMoji semantic interaction, new layout | complete; baseline is 0.8120 |
| 08 | ABO clean text-to-image generation | reserved; model still being revised |

## Fixed protocol

- Performance is the identity-checked full-test result of the named checkpoint/run.
- Baseline timing and A100 board power use the first 200 test samples in dataset order. Ours uses 200 consecutive shape-matched, GPU-resident invocations of each formal electronic kernel.
- There is no explicit warm-up and no discarded measurement; item/call 1 is included in every aggregate.
- Baseline latency is synchronized wall time from the first native Qwen block to the final task output.
- Ours electronic latency is the sum of serialized CUDA-event kernels: post-CCD nonlinearity and learned readout, optical-router arithmetic, required bridge/merger, and task head. CCD crop/stack, layout, SLM encoding, transfers, I/O and generic full-field tensor fusion are excluded. Parallel residual latency is recorded separately and excluded from the critical path.
- Multimodal tasks use six physical optical passes; vision-only LSP and SALICON use three. One pass is 1.0447 ms = 0.714 ms SLM refresh + 0.300 ms exposure + 0.0307 ms camera readout.
- One real MAC is reported as two operations (2 OP). Complex propagation arithmetic is reported explicitly and never hidden inside an undocumented FLOP convention.

## Fixed power and energy equations

- Optical devices: 80.358 W.
- Ours control chassis: 41.388 W.
- Ours A100 board: 62.842 W.
- Baseline chassis: 338.2 W.
- Baseline A100 board power: task-specific mean `nvidia-smi power.draw` during the continuous-inference power pass.

For Ours:

`E = t_optical * (80.358 + 41.388) + t_electronic * (41.388 + 62.842)`

For Qwen3-VL baseline:

`E = t_baseline * (338.2 + P_A100)`

Times are converted from milliseconds to seconds before multiplying by watts.

## Directory contract

- `ours/` and `baseline/`: numbered task folders with raw timing, power, source identity, and exact code snapshots.
- `reserved/`: placeholders for 04 and 08; no fabricated metrics or borrowed timings.
- `A100_demo_energy_efficiency_audit.xlsx`: final workbook, including raw 200-sample timing and power sheets.
- `A100_demo_energy_efficiency_methods.docx`: concise Word methods note.
- `calculated_summary.json/csv`: machine-readable results and all derived quantities.
- `raw_remote/`: immutable downloaded server outputs and logs.

The rejected first LGVQ power pass accidentally sampled physical GPU 0 and is retained only under `raw_remote/audit_rejected_wrong_gpu0_power`. All formal LGVQ power values were remeasured from physical GPU 6 (A100).
