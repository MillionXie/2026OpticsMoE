# G2 rank-64: TRAIN2000 physical adaptation

Completed 2026-10-02. Same no-trick G2 architecture, optical phases, alpha and upstream frozen throughout adaptation. Only the original 30,162-parameter decoder trained; no added modules.

| Same-group comparison | Changed-cell accuracy |
| --- | ---: |
| Original normal simulation | 93.90% |
| Original direct six-layer physical TEST | 61.85% |
| Previous TRAIN1000 decoder adaptation | 87.55% |
| Previous adaptation plus existing bias calibration | 87.80% |
| New TRAIN2000 decoder adaptation, selected e115 | 90.15% |
| New adaptation plus existing edit-bias calibration | 90.45% |

Direct drop is32.05 percentage points,34.13% relative. Best recovered result is3.45 percentage points below original simulation,3.674% relative. The5% recovery criterion passes;2% floor92.022% and1% floor92.961% do not. Original simulation benchmark is not replaced by an adapted checkpoint's potentially changed simulation score.

User-authorized TEST selection every5epochs and calibration are development metrics, not independent generalization. Gradients used independent TRAIN2000 only; no TEST gradients or VAL selection. Added TRAIN1000 source/ID overlaps with original TRAIN1000 and TEST are zero. All six new layers have1000 raw PNG+receipts,6000 fresh CCD; original TRAIN6000 and TEST6000 preserved. New receipts minp99=48, max clipping2.3437%; original2000us/GainX4/wait240/phase/ROI/BMP contract unchanged. No G5 CCD reused.

Both saved full checkpoints strictly reload under default CPU inference, original .6185 baseline matches, upper protection SHA f6fe1517b54a4c4e50100b05b0e30701b3f42b4bc0b2c2d6d227cbc7639bfe9a unchanged. Uncalibrated new best preserved-cell98.1901%, sceneexact57.5%; calibrated best preserved-cell98.0256%, sceneexact56.3%. Calibration trades some scene exactness for changed-cell accuracy; keep both.

Local backups in this folder:

- rank64_g2_train2000_best_20261002.pt — SHA c822d430e24ffa971636c71abb1ef6daa872b189e7ad3f5768d6ba039b138e36
- rank64_g2_train2000_last_20261002.pt — SHA 44c74647c2962cb4af16d12f38638d4d4c641a5f859cc45a9da4828319b14fd0
- rank64_g2_train2000_bias_best_20261002.pt — SHA 1cbc3d2574827272dafee7102a4a402ace7eab3f373ea177cbc02e66d90fe6db
- rank64_g2_train2000_report_20261002.json / history / samples / strict_report
- rank64_g2_train2000_bias_report_20261002.json
- rank64_g2_extra_train_physical_report_20261002.json

All three local PT hashes match remote reports. Existing bench retains raw CCD, receipts, best/last, caches, full per-sample reports, calibration outputs and existing source. Runtime group configuration reused existing lab_tune2000 with a process-local copied G2 mapping; no source transfer or extra project copy. Capture and training tasks finished successfully and GPU/SDK released. This supervision round is paused after delivery; ABO and G5 remain sealed, external upload paused.
