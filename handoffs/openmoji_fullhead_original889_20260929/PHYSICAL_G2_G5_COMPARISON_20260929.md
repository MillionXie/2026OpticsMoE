# OpenMoji full-head G2/G5 same-scene optical comparison

Both groups used the user's new original-standard-electronic-head checkpoints,
the same ordered 1000 normal TEST samples, and the same six-stage SHS contract:
17 µm input, 10 cm propagation in the model, bounded `tanh(abs/.5)` amplitude,
`round(255*a)` BMP without per-batch peak rescale, 2000 µs, Gain_X4,
240 ms, and the established ROI/orientation. Each group has 1000 native CCD
PNG files in each of six stages, 6000 total; both scheduled capture jobs are
Ready and their reports say complete. No electronic fine-tuning was used.

| Group | Weight SHA-256 | Same-weight clean sim TEST | Physical TEST Changed-cell | Add | Replace | Move | Remove |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 direct deploy (G1 weight) | `579b19e78befe93cd258ce02308235f1a9176316f37bc42fc2e1b6ff06d38e9f` | 0.8880 | **0.8915** | 0.760 | 0.864 | 0.942 | 1.000 |
| G5 CCD + DC30% + grid | `8ba828c8168db6cc67b2721c65fa6db57efadc6d091e3a7204b79a49a17b3e52` | 0.9435 | **0.8740** | 0.784 | 0.764 | 0.952 | 0.996 |

G5's physical 0.8740 is 0.0140 below G1/G2's simulation 0.8880, or 1.58%
relative, so it meets the requested within-5% target without fine-tuning.
However, G2 is 0.0175 **better** than G5 in this same-session physical test.
This experiment therefore does not support claiming that G5's robust tricks
improved physical Changed-cell accuracy. The G2 weight is the stronger
physical candidate on this one bench/session; further independent repetition
would be needed for a stability claim.

The user chose "达标即止，优先交付" after seeing the G5 result. Therefore no G5
TRAIN acquisition or decoder fine-tune was run; the G5 original checkpoint is
preserved. G3/G4 were not physically tested under this time-limited request.

Local machine-readable reports, each with a row for every sample:

- `g2_physical_full1000_report.json`
- `g5_physical_full1000_report.json`

Native CCD, receipts, phases, inputs, predictions, and the source reports
remain on the SHS bench in
`E:/code/guest/2026OpticsMoE/OpenMoji_FullHead_Robust_SHS_20260929/runs/g2_full1000`
and `.../runs/g5_full1000`, respectively. The two reports each contain 1000
unique sample IDs in identical order. Old dark pilot runs are retained apart
from these full runs. The manual/automatic SLM-display brightness discrepancy
and its successful bounded retest are documented in `PHYSICAL_PROBE_20260929.md`;
its precise cause is not conclusively established. Mild saturation was logged
per frame and accepted by the user without changing exposure or gain.
