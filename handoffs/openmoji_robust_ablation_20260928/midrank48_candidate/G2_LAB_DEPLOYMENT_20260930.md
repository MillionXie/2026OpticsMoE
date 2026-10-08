# OpenMoji rank-48 G2 (no-trick) lab deployment

- Isolated bench project: `E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Midrank48_G2_SHS_20260930`
- Checkpoint: `weights/g2_rank48_no_trick_best.pt`, SHA-256 `373ab8968bdaf9b41e70b18b23f2d474fa8e8f284e983aab5d429a0f99906a5e`.
- Same rank-48 electronic head as the prior G5 comparison; G2 has no CCD-noise, coherent-DC, or training-grid robustness tricks. Normal simulation TEST changed-cell accuracy: 0.9135. This is not a physical result.
- Same 1000 TEST identities, six optical stages, 2000 us / Gain X4 / wait 240 ms, existing ROI and orientation, bounded-amplitude BMP. No G5 CCD is reused.
- Strict checkpoint loading and six-layer ideal-detector replay selftest passed with zero bridge/replay error in `runs/g2_rank48_selftest`.
- Four-sample six-stage pilot succeeded: 24/24 CCD with signal; maximum sampled saturation fraction 0.0240. The shared 1% saturation abort was too strict for this candidate; only this isolated G2 capture uses a 15% abort while logging each frame's saturation, with physical settings unchanged.
- Full capture task `OpenMoji_Rank48_G2_Full_0930` completed successfully (task result 0). Run directory: `runs/g2_rank48_full1000_e2000`. Each of the six stages has 1000 PNGs and 1000 JSON receipts (6000 CCD images total); the report status is `complete` and no traceback was found.
- Fixed same-checkpoint TEST1000 Changed-cell accuracy: simulation 0.9120, physical 0.6060. Physical operation breakdown: add 0.300, replace 0.524, move 0.608, remove 0.992 (250 each). Full report copied locally to `g2_rank48_physical_full_report_20260930.json`.
- For comparison, rank-48 G5 physical pre-tune 0.6350 and VAL-selected decoder-tuned 0.7815 (normal simulation about 0.9000). Thus G2 is 2.9 percentage points below G5 before tuning, but not near the hoped-for 0.55. Do not tune G2 on TEST results.

## Independent TRAIN capture and decoder adaptation

- Copied the previously audited original-TRAIN1000 source set into the isolated G2 project; `capture_train.jsonl` SHA matches its G5-side source copy. Split audit: 800 FIT / 200 validation, zero TEST identity overlap and zero TEST source-image overlap.
- Started exclusive interactive task `OpenMoji_Rank48_G2_Train_0930` at `runs/g2_rank48_train1000_e2000` for 1000 TRAIN images × six complete stages. The G2 model, phase masks and 2000-us/Gain-X4/wait-240-ms contract remain fixed. Do not mix G5's TRAIN CCD with G2's.
- Prepared isolated `run_tune.cmd` and G2-specific `lab_tune_decoder.py`: adapt only `shared_readout.decoder`, select epoch using the 200 validation identities and replay already-captured TEST CCD once after selection. Do not start until all 6000 TRAIN CCD/receipts are audited and the capture task has released the devices.

## Completed TRAIN and tuning result

- Independent TRAIN1000 capture completed normally: six stages × 1000 PNG and 1000 JSON receipts = 6000 CCD. Source split audit remains 800 FIT/200 VAL and zero TEST-source overlap.
- Offline decoder-only task completed normally. TRAIN VAL selected epoch 29 at Changed-cell accuracy 0.8600; upstream/protected checkpoint parameters are unchanged. The fixed replay on the previously captured TEST1000 CCD improved physical Changed-cell accuracy **0.6060 → 0.7780**. Operation breakdown after tuning: add 0.556, replace 0.840, move 0.732, remove 0.984.
- Best checkpoint SHA-256 `abfb856abe43d4001b533205da81697fa3017dc365a89c9ec0276580dc37f288`, verified against the local downloaded `g2_rank48_decoder_best_20260930.pt`. Local `g2_rank48_decoder_report_20260930.json` retains the full report; the isolated bench retains best/last, raw TEST/TRAIN CCD, per-frame receipts and cached decoder inputs.
- This G2 tuned result remains below rank-48 G5's VAL-selected tuned 0.7815 by 0.35 percentage point. Neither TEST set was used to select the reported VAL-selected checkpoints.
