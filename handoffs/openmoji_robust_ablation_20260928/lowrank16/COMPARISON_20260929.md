# OpenMoji robust-stage optical comparison

Metric: changed-cell accuracy on the same original 1,000-image TEST manifest
(`c86765b9a6643be05aa0a91e1060457e214f01930bf483540a4c4e845dc15789`).
Each row uses its own checkpoint, six complete optical layers and 6,000
physical CCD frames. Simulation is the same bench code/data evaluation, not a
physical measurement. SHS settings: 2,000 us, Gain X4, 240 ms wait, verified
ROI/orientation; bounded BMP is `round(255*a)` without peak rescaling.

| Group | Training change | Same-bench simulation | Physical optics | CCD | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---|
| G2 | Base weights, direct deployment | 0.8965 | 0.5345 | 6000 | `98fd922676b4faa7b5935e408f6aa31a0517b82f44b826c690ca4115518174c0` |
| G3 | + CCD-noise training | 0.8990 | 0.5435 | 6000 | `ae1525dc02306d9fd31bf9e0b675f279731b0543d228f3b2e3c48aecf011bb27` |
| G4 | + coherent DC 30% training | 0.8900 | 0.5880 | 6000 | `6234f977cacac2f78fbcba7ddebd9888d9c802bbba58249be9822dc21241c62f` |
| G5 | + interpolation during training | 0.8885 | 0.5860 | 6000 | `7a4b9ba5bbcf546e4af87ad3f03d99fb53e2ec8d7236b2c6fdbcfb65dc56154a` |
| G5 + physical decoder adaptation | Same G5 optics; last electronic decoder tuned on separate TRAIN CCD | not evaluated as a new simulation checkpoint | 0.6710 | original 6000 TEST reused; 6000 TRAIN captured | `cdc9ce0a45c55932a5d06250750093b9cfd7d3e0da4e1c9f36ba06adce121c82` |

The user clarified that the conditional G5 decoder adaptation compares G5
physical accuracy against the base *simulation* accuracy, not G2 physical.
The relative-five-percent floor is therefore `0.95 × 0.8965 = 0.851675`.
G5 physical 0.5860 was below it. An independent original TRAIN1000 was
captured in six full layers (6,000 CCD), with 800 fit / 200 validation and
no TEST source-image overlap. The frozen-optics decoder-only tune selected
epoch 13 on validation (changed-cell 0.7700); one fixed replay of the
original TEST CCD improved 0.5860 to 0.6710. This remains below 0.851675.
TRAIN was captured at the same 2,000 us/Gain X4/240 ms contract; some bright
TRAIN frames clipped, so their saturation fraction is recorded per receipt.
The TEST capture and its guard were unchanged. Do not describe the result as
successful transfer or tune further on TEST.

Per-group reports: `g2_physical_full1000_20260929_report.json`,
`g3_physical_full1000_20260929_report.json`,
`g4_physical_full1000_20260929_report.json`,
`g5_physical_full1000_20260929_report.json`. Complete raw CCD and receipt pairs
remain on the bench under
`E:/code/guest/2026OpticsMoE/OpenMoji_RobustAblation_SHS_20260929/runs/`.
The adaptation reports are `g5_train1000_physical_20260929_report.json` and
`g5_decoder_tuned_20260929_report.json`; best/last decoder checkpoints and
cached features are on the bench under `runs/g5_decoder_train1000_20260929`.
The selected best checkpoint is also backed up locally as
`g5_decoder_tuned_best_20260929.pt` with matching SHA-256.
