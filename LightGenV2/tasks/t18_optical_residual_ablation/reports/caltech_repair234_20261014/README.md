# Composite physical repair / bounded L4 residual validation

Actual start 2026-10-14 04:05:09 Asia/Shanghai; CPU parent3210879, source commit46011b823ff4cffc11c4f5943c62e31501c2a4c6.
Run `caltech_revised20_s17_mainL4rho03_RGB_dense_global_regstudy150_gpu256_20261014`.
This is the three-candidate repair/regularization validation, not formal six-arm training.

| Candidate | Physical GPU | Actual child at start | Current verified snapshot |
|---|---|---|---|
| MixUp.2 |6 A100 |3211094 |E2 CPU reviewed |
| MixUp.4 |2 3090 |3211169 |E2 CPU reviewed |
| MixUp.6 |5 3090 |3211235 |E2 CPU reviewed |

Parent CUDA invisible; each child only its recorded GPU UUID, Torch2GiB+800MiB margin.
No GPU1 used. Own actual memory roughly678/678/872MiB in first-epoch snapshot;
foreign processes untouched. PID/epoch snapshots are historical, inspect current state.
Three actual fresh production numerical gates passed;126 source current/original
Git blobs exact, original121 unchanged. Linux Torch2.6.0+cu124/Python3.11.15.
Initial state identical; each E2 has156 Adam steps, scheduler2, all11 phase gradients
finite nonzero, every order/affine/MixUp hash including macro tail14 verified by CPU.
No model forward in saved partial review, no test arrays/inference/claims.

New20 train1246/val261 only. Fixed RGB/dense4/preserve478 composite, actual8bit
joint amplitude/phase remains inside fits. Six depth/rho actualGPU preflights passed
20-class probability repeat error0, maxgradientrelative5.86e-7/mincos>.9999999999998.
18 synthetic random/dim/black nativeBMP roundtrips exact, probability error0,
3/4/5captures respectively, black all-stage amplitudes zero. Fresh checks are not
trained-weight or physical-device acceptance. Old224-only exporter incompatible;
selected weights require matching new export and fresh hardware calibration.

First epoch including train endpoint measured55.38/56.19/58.23seconds. Sparse initial
throughput suggests this validation roughly2.5-3hours if all150 epochs run, about
06:40-07:15 Beijing; estimates require later logs and may change. Early stop allowed
only by predeclared rules. Formal six fits and full delivery are subsequent work;
09:00 report is actual progress, not a guarantee of completed75% acceptance.

All nine prior same-budget diagnostics audited and preserved. More MixUp alone
failed to restore learning; RGB/dense/global information factors had much larger
finite-budget effects. Their composition and new categories change the task; do
not add factor gains or assert one cause. At E1 validation was only1.53-1.92%; these
are startup metrics, not evidence that the repair passed. Do not hide them or call
small initial gap overfitting solved. Candidate eligibility uses selected valacc>=65%,
macro>=60%, all selected valrecalls>=20%, no selectedtrain/valzero, persistent-zero
limit, gap<=15pp, complete routing variation and trainedBMP checks. No eligible
candidate means no forced formal run. Independent2/3validation is still required.

[Predeclared scientific protocol](../../CALTECH_REPAIR234_PROTOCOL_20261014.md),
[actual start/UUID/source proof](production_start_health.json),
[first measured epochs](first_epoch_health.json),
[actual numerical/nativeBMP receipt](numerical_native_bmp_receipt.json),
[saved partial CPU review](partial_CPU_receipt.json).

The first checker invocation failed before creating an output because repository
imports were not initialized. The import-only entrypoint fix46011b823 was Git
published before successful checks; model, science configuration and gates did not
change. Original private error log retained. No failed scientific training retried.

Matched independent478-global/dense4 exporter now implemented and all six fresh
static plus18dynamic CPU native checks passed; selected trained-weight verification
still pending. [Contract and actual receipt](matched_export.md).
