# Saved classification-window and confidence sanity check

Read already published audited selected-EMA confusion matrices and4521 saved
train/val prediction rows (three arms,1246+261 each). No model construction/forward,
dataset image arrays/test labels or inference, GPU/device or new calibration fit.
Saved prediction CSV SHA recorded; argmax labels/counts/CM match the existing audit.
This is additional descriptive analysis, not a repeated full scientific fit audit.
[Exact receipt](saved_window_confidence_cpu.json).

All20 training confusion-matrix rows have a unique diagonal maximum in each arm.
Minimum per-class training recalls are72.1519/56.9620/65.8228%; minimum diagonal
count advantages over any wrong column are27/19/29. Therefore identity labeling is
the unique training-accuracy-maximizing global one-to-one detector-label permutation:
for every row i, CM[i,i] > CM[i,j] for all j!=i, so identity attains every row maximum
simultaneously. No alternative permutation was fit/adopted or evaluated on test.
This supports that training learned all class/window associations; it does not rule
out other geometric defects, distribution shift or sample-specific failure. A
validation-only label remapping would change the frozen contract and is not a fix.

| Case | Split | Accuracy | Mean max probability | ECE10 | Mean max probability on wrong predictions |
|---|---|---:|---:|---:|---:|
| L4_rho03_mixup02 | train | 92.7769% | 55.3067% | 0.375005 | 32.0855% |
| L4_rho03_mixup02 | val | 63.9847% | 46.9464% | 0.188813 | 31.8263% |
| L4_rho03_mixup04 | train | 90.4494% | 50.9611% | 0.395345 | 31.3487% |
| L4_rho03_mixup04 | val | 62.0690% | 43.7350% | 0.183340 | 30.2214% |
| L4_rho03_mixup06 | train | 91.6533% | 49.4042% | 0.422641 | 30.0875% |
| L4_rho03_mixup06 | val | 64.3678% | 42.8000% | 0.232414 | 29.3157% |

ECE10 uses fixed bins[0,.1),...,[.9,1] and sum over bins of fraction * absolute
(mean correctness - mean maximum probability). It is descriptive and bin-dependent,
not an eligibility threshold or proof of causality. Lower aggregate confidence than
accuracy does not prove every bin is underconfident. The scores are normalized
optical detector energies, not independently calibrated physical probabilities.

These data show poor generalization can coexist with soft probability distributions.
For any common positive temperature, p_new[k] proportional to p[k]^(1/T) is monotone
and preserves every argmax. Such confidence sharpening may change NLL/calibration
but cannot improve overall hard classification accuracy on these saved predictions.
No temperature, threshold, class weighting or new selection rule was applied/fitted.
Do not change the validation-NLL first-tie rule or use test to calibrate/select.

The already proposed main-phase capacity comparison remains unapproved/unstarted.
This sanity check neither passes prior eligibility nor grants new GPU budget. GPU2/5/6
remain released. Keep class mappings/data/ABO/old runs and prior claims protected.
