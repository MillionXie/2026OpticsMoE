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

## Pinned artifacts

| Artifact | SHA-256 |
|---|---|
| accuracy-first EMA checkpoint | `8cc41ac48ef66385e612be33f6cdb4c7be4675e62daaf66712f77e5b36b8a4fe1` |
| accuracy-first `vision_expert.bmp` | `14fa29ec902ffc394b30a481b86e87b4f70c930f2e654bb813ca990ffbfa2bc8f` |
| balanced EMA checkpoint | `7a5c891cc8eba030a674100511a003de340d28a72d919efb897d56ae91d02f86c` |
| balanced `vision_expert.bmp` | `0dc71fe420d485c82e3d966b3d77486bb490cd315c2af738ccd185ebadcc53fdb` |

Each first-stage session contains exactly 210 manifest rows, 210 compact PNGs,
and 210 ready-to-play 1024×1024 BMPs. The delta ZIP is 128 MB and passed
`unzip -t`; its SHA-256 is
`1452f343f0abca4fc0b420c4234f863e634c3860b312df0af46ff873b5caa1bf0`.

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
