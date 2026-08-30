# Early-robust trade-off results

## Audit scope

Both variants resume the same pinned Stage-A EMA checkpoint at selected epoch 3.
They reset optimizer state, activate all perturbations from absolute epoch 4, and
run 32 continuation epochs. The 200-query sealed test is evaluated exactly once
after train-loss checkpoint selection.

## Simulation results

| Variant | Top-1 | Top-3 | MRR | Selected epoch | Optical gates | Phase delta RMS |
|---|---:|---:|---:|---:|---:|---:|
| accuracy-first, 0–5% coherent leakage | 85.0% | 94.5% | 0.9063 | 33 | 0.626% | 0.701 rad |
| balanced, 2–8% coherent leakage | 85.0% | 94.5% | 0.9067 | 33 | 1.024% | 0.851 rad |
| previous strong-noise | 82.0% | 93.5% | 0.8871 | existing | 1.52% | 0.439 rad at its selected run |
| pure electronic reference | 87.0% | 96.0% | 0.9159 | existing | n/a | n/a |

The gate percentage is the learned residual mixing coefficient, not measured
optical power. The coherent-leakage percentages in the configuration are
intensity fractions and are square-rooted before field mixing.

## Hardware decision policy

1. Acquire and four-stage fine-tune accuracy-first using only frames captured
   with the verified 3500-us LUT.
2. After every layer, report sealed-test Top-1 plus PCC, SSIM, gain-aligned NMAE,
   and saturation fraction. Do not select epochs on the sealed test.
3. Continue while geometry/orientation contracts pass and the cumulative Top-1
   remains compatible with a final 78% target.
4. Acquire balanced only if accuracy-first is overly sensitive to light-path
   variation or a second robustness point is needed for the paper.
5. Keep the previous strong-noise model as the robust-heavy third point; no new
   training is required.

No simulation result guarantees the final hardware accuracy. The new LUT,
homography/orientation contract, normalization parity, and per-layer local
development-set checkpoint selection are all required to interpret the hardware
result.

## Engineering acceptance bands for the 78% objective

These are go/no-go bands for scheduling scarce laboratory time, not claimed
predictions:

| Completed measured stages | Healthy target band | Stop and diagnose below |
|---|---:|---:|
| simulation only | 85% observed | n/a |
| vision expert | 82–85% | 80% |
| + vision global | 80–84% | 79% |
| + language expert | 79–83% | 78% |
| + language global | 78–82% | 78% final requirement |

If PCC/SSIM and orientation are poor, do not spend epochs compensating for a
geometry or LUT error. If agreement metrics are healthy but development and
sealed-test retrieval fall together, then the model/noise schedule is the
appropriate lever. The 85% simulation checkpoint leaves seven percentage
points of margin, compared with only four points for the former 82% model.
