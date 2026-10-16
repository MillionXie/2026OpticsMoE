# Six-layer refinement of the SAME linear readout

Human authorizes targeted residual optimization on the fixed originalCaltech20.
Baseline six arms100 and the first20 extra regularization epochs all truly
completed their own science/test receipts and CPU/GPU exits. No old stage is
re-entered or evidence overwritten. Genreg selected2/6 retain baselineE100;
selected4E112 improves test83.2364->83.4787 and macro62.8181->64.4978.
Six remains83.0426, below the non-weakened four reference by9 correct samples.

Next hypothesis: a fast full-batch regularized solution of the existing linear
readout can adjust decision boundaries without more expensive optical training.
This is not temperature-only calibration, an added network or a larger head.
Only the existing20x256 matrix+bias20=5140parameters can change. Every phase,
tiny electronic convolution and learned alpha remains byte-identical to the
accepted parent six-layerE100 EMA (actual previous optical budget120, selected
optical epoch100). Inference remains one router, E-Gx3, same RMS fusion and
bounded alpha>.7, same pooling/nonaffineLN/Linear256to20.

## Training only

Use already independently audited clean TRAIN1372/VAL283 feature caches for
this exact frozen parent. No TEST features/images are materialized while fit
or admission runs. Compute feature mean/std from TRAIN only, std floor.1.
Coordinates z=(F-mu)/sigma improve conditioning. Optimize uW/uB with CPUfloat64
fullbatch L-BFGS (max_iter1/strongWolfe/history20 per call), weighted smoothed
CE(LS.05) plus lambda||uW||^2/2. Lambda is a deterministic TRAIN-only rule:
clip(2*.2*initial weightedCE/||uW0||^2,1e-4,1e-2), not a validation/test grid.
Initialize uW0=Wparent*sigma,uB0=bparent+Wparent@mu. New optimizer is explicit;
no claim of resuming Adam in this head-only stage.

At most20 actual parameter-changing head updates, stopping earlier if the
optimizer makes no change. Record true LBFGS iteration/function-evaluation
counts, not fabricated epochs. HeadEMA.8. Fold each candidate back exactly:
W=uW/sigma,b=uB-W@mu, thenfloat32. No mu/sigma module/buffer/extra layer is
added to deployment. Frozen optical epochs do not increase in this stage.
Select first minimum validation balancedNLL over parent and headEMA updates,
with differences<=1e-6 treated as numerical ties retaining the earlier model.
Never select maximum TEST accuracy or change classes/roles/controls.

## Independent proof and saved audit

New CPU admission checks actual parent-cache probabilities, exact algebraic
folding, all20-class weighted-smoothed CE analytic gradients with both linear
tensors finite/nonzero,5140count,TRAIN-only statistics and no inference extras.
0optimizer updates/0optical model forwards/0TEST arrays in admission.
Source/config and parent PT/cache SHA identities are locked before fitting.

Save every raw/EMA head update,actual TRAIN/VAL probabilities,loss,closure
count,scaler statistics and optimizer state. Own exclusive CPU saved audit
independently checks scaler/ridge rule,objective,folding,EMA recurrence,
first-minimum selection,original IDs/labels/support/CM/macro/NLL,only W/b changed,
and identical non-head digest. Only pure immutable saved-math helpers are
reused, not any old audit dispatch or claim. Failures persist; no retry or
threshold relaxation. No new optical forward is required for cached features.

After genuine fit exit rc0 and own science/completedSHA, selected unchanged
parent genuinely reuses its old TEST. A new head evaluates the matching frozen
TEST feature cache exactly once,0new optical forwards/1head batch; this still
counts as one new selected-model TEST evaluation. Own independent saved TEST
audit checks probabilities/IDs/CM/NLL and frozen optics. Do not pretend the
whole development has only tested once. CPU campaign exit code is not invented.

Four reference is the acceptedE112 model83.47868217%, never lowered to make six
win. Keep no-residual100 tests frozen. Report six>four and preferred gap>10pp
separately; success is not guaranteed. If the chosen head lowers TEST versus
its accepted parent, reject this head method family and retain that parent,
reporting the negative method result. Do not choose a favorable intermediate
head step. This is explicit TEST-informed family acceptance/development,
not independent unseen generalization or equal-budget depth/residual causality.

CPU-only tiny linear fit, no GPU/A100/SDK/physical acquisition, no new categories,
no new electronics, no camera-data deletion, no automation. Original20 cache,
all old failures/claims/weights/histories/plots and accepted two/four remain
protected. Future physical fine-tuning loads this SAME trained linear head.
