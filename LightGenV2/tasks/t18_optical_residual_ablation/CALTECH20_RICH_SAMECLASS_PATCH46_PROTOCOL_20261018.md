# TRAIN-only same-class donor patch augmentation, retained L4/L6

Continuous human authorization: actively reduce overfitting in L4/L6, existing
weights only, same inference/roles, L2 FINAL untouched, use free shared GPUs up
to5, no timers or lab interference. Source468=protected463+five new leaves.

Hypothesis: TRAIN≈95.6% vs VALID≈78.9% persists after many mild geometry/noise
methods. Add within-class appearance variation without changing any label or
deployable graph. This tests a data-coverage regularizer, not a guarantee or a
claim that a single diagnosed cause explains overfitting. Keep calibrated KD.25
and every other459 method setting/parent for a matched augmentation comparison.

Per TRAIN macro, RNG SeedSequence[17,localE,batch,20261018,468], probability.5:
each image receives one 10–25% rectangle (ratio.9–1.1) of ORIGINAL150RGB from a
different TRAIN image of the SAME class. No VAL/TEST donor. Apply only before
the existing augmented-view crop/color/affine/encoding/MixUp. The second RGB
access used for the clean view returns the exact original images; assert two
accesses. Source tensor never mutated. Existing common MixUp labels, public
teacher probabilities/gates and class prototypes remain exactly aligned; they
are evaluated on the untouched clean view. No incorrect cross-class CutMix
label-area assumption, feature/teacher/inference expansion or new parameters.

Warm the same source459 a2244/4f9d6 parents; no phase/alpha/head reset. Five LOCAL
epochs, one TEST at5 and reused incumbent0,2GiB+800MiB, best/last only,1800fitsec
guard, same old source459 independent numerical thresholds and current-teacher
calibration. Actual numeric gate sees these augmented inputs and all21/30
gradients on all20/macro16/tail5 micro1/2, no optimizer/TEST. Independent CPU
synthetic check proves donor class, source preservation, exact unchanged clean
view, rectangle-only replacement; no real image/forward/optimizer calls.

Every macro saves donor/box-plan SHA. Own independent saved science retains the
full source/config/PT/local order/affine/crop/LR/Adam/EMA/phase/prototype and
calibrated raw teacher-target hashes plus IDlabelheadCMNLL/TESTselection checks;
independently rebuilds all within-TRAIN donors/boxes/active flags from labels and
RNG. Unique own claims/logs/out/actualrc0/exit/SCIcompletedSHA, no retry or relaxed
thresholds. Explicit TEST-informed development and extra residual/pretraining
budget, not unseen generalization/equal-budget optical residual causality.

GPU preference free4090 indices3/0/1/4 only. Queue rather than duplicate a card;
global≤5 one per card, noA100/foreignkill/environment change. Never modify running
459/463 sources, frozen controls, old partial/failures/claims, hardware e7c/5c8
or any CCD. Old best remains on lower/tied actual TEST correct count. Source only
Git exact5paths, runtime config/identities saved; raw/PT/IDs/logs private.
