# Ordinary per-image CE versus class-weighted CE, retained L4/L6

Latest human optimization scope: preserve L2 FINAL e7c/5c8, focus4/6 overall
accuracy and try to increase gaps, fixed rich20/graph/controls, no timers or lab
interference. Source472=protected468+four leaves; only class_weight_exponent
.5→0 changes from459, same .25 calibrated publicKD and all other inputs/methods.

Evidence: TRAIN class supports68..640 differ by~9.4x. Existing sqrt-inverse
weights give a rare-class example~3.07x the loss weight of a common-class example.
The human's requested metric is overall correct count, not uniform-class macro.
Teacher itself was prepared using ordinary per-image CE and meets90% VALID.
One fixed ordinary-CE comparison tests objective alignment; no claim this alone
solves overfitting, no class/support/test changes or removal of difficult classes.
Keep crop/MixUp/feature-dropout/JS/phase jitter/prototype/publicKD/EMA regularizers.

This may trade rare-class recall/macro for overall accuracy. Save and report
all20 confusion/support/recall and macro, including zero/poor classes; never call
an overall gain universal improvement or independent generalization. No control
training/test rerun, no suppression of control accuracy. Extra residual tuning,
teacher knowledge, unequal alpha constraints remain confounds to causal claims.

Warm exactly the same accepted parents as459: a2244 TEST82.00%,4f9d6 TEST81.02%,
no fresh weights or alpha reset, five LOCAL epochs, test onlyat5 plus cached
incumbent0. Same single-router EG4/6,5140head/electronics/physical quantization,
alpha .4001..95; no new inference parameters or phase structure. No same-class
patch in this matched ordinary-CE comparison; that is the separate468 method.

Independent CPU proof preserves current TRAIN teacher scales/targets, phase
semantics and sources, proves all actual class weights exactly1 and analytical
ordinary soft-target CE derivative;0 real forward/optimizer/TEST images. Actual
own unchanged micro1/2 all20/macro16/tail5 numeric gates,3e-6/.02/.999,21/30
finite/nonzero gradients. Full independent saved state/order/affine/crop/LR/
Adam/EMA/phase/prototype/teacher target hashes/IDlabelheadCMNLL/selection science,
original rc0/exits and exclusive claims/completed SHA. Preserve any failure;
no retry, threshold relaxation, old source mutation or checkpoint deletion.

Prefer free shared4090 GPUs1/4 then0/3, global≤5 one/card2GiB+800MiB, noA100 or
foreignkill. Two468 jobs+two472≤4. Best/last only,1800fitsec guard. No captured
version replacement, no lab queue/SDK/CCD actions. Exact own4paths Git source;
rawJSON/PT/IDs/logs private. Global earliest best survives ties/regressions.
