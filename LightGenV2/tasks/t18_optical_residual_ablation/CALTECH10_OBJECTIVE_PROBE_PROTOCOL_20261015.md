# Fixed TRAIN objective-gradient and photometric probe (2026-10-15)

Human ongoing expert adjustment authorization, fixed10/data/inference/frozen100.
Responsibility only this protocol and caltech10_objective_probe.py;209=207+2.
207 measured teacher existing-affine TRAIN98.4-98.8% and actualKDgate89.93%,
disproving near-disabled teacher signal. L2clean TRAIN79.1%, bonsai34.83%,
maxprob45.56% still underfit. L4train99.6/val87.26 suggests heldout generalization,
not poor robustness to the same original affine views. Old selected models,
source203/205/207, failed205/claims/weights untouched; no old helper rerun.

Before selecting one next strategy, on idle physicalGPU4 exact-self guard207,
shared2GiB+800MiB/max4nonA100/one child per card; no foreign process changes:
L2saved validation-selected E1018, teacher acceptedL4E488 frozen/no-grad.
Two fixed TRAIN macro20 groups: first2 examples per10classes, first10bonsai and
first10airplanes. Original deterministic epoch1041 affine/MixUp; retain config.
Separately measure CE, actual .5*T2 gatedKD, capture, smoothness gradient norms
and cosines; predeclared focal-gamma1.5 weightedCE candidate for hard-example
emphasis, not a radius/weight sweep or update. No Adam/model state changes.
Also measure saved teacher on all761TRAIN under neutral and TWO fixed modest
photometric transforms: RGB gains[1.2,1,.8]/contrast1.2, gains[.8,1,1.2]/contrast.8,
clamp to byte input before unchanged optical encoding. Report whole/perclass
recall/confidence, no validation/test arrays, no threshold changes. This is
training-only sensitivity evidence, not a new model accuracy or generalization
claim. Exact source/parent IDs and modeldigests, one claim/out/failure-no-retry.
Do not train/test from this diagnostic module; later explicit fixed strategy,
its own gates/science needed. All raw results private, source through Git only,
exact2paths serial, foreignT06/bench/shared README untouched.
