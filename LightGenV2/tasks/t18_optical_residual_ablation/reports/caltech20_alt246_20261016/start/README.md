# Caltech20 single-router alternating 2/4/6: actual start

This is a new simulation architecture/task, not the old accepted Caltech10 model.
Run `caltech20_alt246_biasfree_rmslearned_linear_s17_equal100_20261016`. Actual training source: `649ef57e580f8ee490210f417b7b55468d316b8c`.
Source closure:168 files, current/original verified; reports later change HEAD only.
CPU campaign PID at launch:896402; GPU producer PIDs are historical,
and must be identified from current /proc commands/UUIDs rather than reused blindly.

## Architecture checked against actual implementations

The old Kather reproduction calls `prompt.routing` once, before expert/global
cycles. New main depths are E-G / E-G-E-G / E-G-E-G-E-G, plus one router.
Kather nine-expert geometry is not claimed identical to current four224 experts,
478 global and224 router; only the alternating order/router count were verified.

ABO reference:`tasks/t07_abo_image_retrieval/standalone/model.py:rms/alpha_value/fuse`.
Both electronic and optical branches are divided by detached per-sample RMS;
learned `(1-alpha)*electronic+alpha*optical` is renormalized and electronic
reference RMS restored. Alpha is bounded(.7001,.95), initially.8, learned at
every E/G. This is a normalized feature coefficient, not a fixed probability
mixture, global optical-power fraction, or guaranteed product of layer weights.
ABO's own original alpha bounds/results are not changed or reused as T18 scores.

Each main stage has the same tiny1channel3x3 bias-free electronic residual in
both arms. Terminal pooled16x16/nonaffineLN/Linear256to20 is learned from start.

| Main depth | Optical phase parameters | Electronic parameters | Terminal linear |
|---|---:|---:|---:|
|2|479364|5160|5140|
|4|908552|5180|5140|
|6|1337740|5200|5140|

All phase, convolution, alpha and linear parameters are jointly trained. Only
optical coherent rho differs(.3/0) within a pair; electronics are not removed
from the no-optical-residual control. No large electronic classifier or pretrained
front-end. Future physical deployment must replay this same graph/state.

## Actual admission and engineering correction

CPU proof SHA:`e7dcc89bdf2dd883e58c7f47f410641b879c64827760d927cf01cf20290aa9fe`.12 real structural forwards, no optimizer
updates or TEST arrays. Extracted actual ABO functions exactly matched new fusion
values AND gradients for two shapes (both maxabs0). Black20 probabilities sum1,
true capture trace/router count, alpha bounds, paired init and NumPy linear
arithmetic checked.

The first source/run b33850f7 failed its GPU admission before any optimizer/test.
The final convolution bias had a null surrogate direction under detached RMS
and terminal LayerNorm, producing about1e-10 gradients and unstable relative
comparisons. Failure/source/claims/logs remain untouched. Corrective649ef57e
removes convolution biases; a separate CPU toy diagnosis records the old null
bias and nonzero weight directions. No numerical thresholds were relaxed.
This is a new registered correction, not acceptance/re-entry of the failed run.

Actual four producers passed their OWN all20/macro16/tail12 micro1vs4 tests,
including every phase/electronic/alpha/linear gradient finite and nonzero.

| Case | Actual own gate | Probability maxabs | Maximum gradient relative L2 |
|---|---|---:|---:|
| L4_rho03 | passed | 1.4901161e-08 | 1.0562391e-05 |
| L4_rho0 | passed | 1.1175871e-08 | 8.5429465e-06 |
| L6_rho03 | passed | 1.1175871e-08 | 2.1351132e-05 |
| L6_rho0 | passed | 7.4505806e-09 | 5.1483465e-05 |

## Fixed baseline and next decision

Original Caltech20 cache/split:1372 TRAIN/283 VAL/2064 TEST. No new class screening,
preparation or role reassignment. Cache SHA a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1.
Prior class screening used validation and some Caltech test data were already
observed; all further test feedback is development, not unseen generalization.

Both arms fresh seed17/full100;86 Adam/E,8600 updates, effective16/micro4/tail12;
paired same init/order/affineMix. AdamW/cosine/EMA/MixUp/LS/TV/capture identical.
One child/card0/2/4/5, at most4,2GiB+800MiB spare; A100 forbidden, foreign PIDs
untouched. Four/six pairs currently occupy four cards; two-layer pair queued.
This is a start snapshot, not a completed100 or an accuracy achievement.

After actual100 and producer exit, independent saved source/state/EMA/schedule/
order/feature-pooling/linear/ID-CM-NLL science must pass. Only then selected models
each test once; no-residual100 controls then freeze. Residual-only extra tuning
requires actual curve/error diagnosis and a newly frozen continuation profile.
Targets are L4 residual TEST>.8, L6 residual TEST>L4 and preferably paired
gaps>10pp. None is guaranteed; do not weaken controls, rescreen classes or select
highest-test epochs. First minimum validation EMA balancedNLL remains selection.

No scheduled monitor was created; follow-up is through the active chat.
